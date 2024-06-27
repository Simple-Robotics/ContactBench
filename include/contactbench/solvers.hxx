#pragma once

#include "contactbench/bindings/context.hh"
#include "contactbench/friction-constraint.hpp"
#include "contactbench/macros.hpp"
#include "contactbench/solvers.hpp"
#include <Eigen/src/Core/util/Constants.h>
#include <cassert>
#include <iostream>
#include <pinocchio/utils/static-if.hpp>
#include <proxsuite/proxqp/dense/compute_ECJ.hpp>
#include <proxsuite/proxqp/status.hpp>

namespace contactbench {

using Eigen::Infinity;
using Eigen::LLT;
using Eigen::Map;
using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::MatrixBase;
using Eigen::MatrixXd;
using Eigen::Ref;
using Eigen::Vector2d;
using Eigen::Vector3d;
using Eigen::VectorXd;

//
//
// utils functions
//
//

template <typename T, typename MatIn, typename MatIn2, typename MatIn3,
          typename MatIn4, typename MatOut>
void iterativeRefinement(
    const MatrixBase<MatIn> &A, const MatrixBase<MatIn2> &B,
    const MatrixBase<MatOut> &X_out, const MatrixBase<MatIn3> &ATA,
    const MatrixBase<MatIn4> &ATB,
    Eigen::LLT<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> &ATA_llt,
    const T epsilon, const int maxIter, T rho) {
  CONTACTBENCH_UNUSED(epsilon);
  auto &X_cc = X_out.const_cast_derived();
  auto &ATA_cc = ATA.const_cast_derived();
  auto &ATB_cc = ATB.const_cast_derived();
  X_cc.setZero();
  ATA_cc.noalias() = A.transpose() * A;
  ATA_cc.diagonal().array() += rho;
  ATB_cc.noalias() = A.transpose() * B;
  ATA_llt.compute(ATA_cc);
  T stop;
  CONTACTBENCH_UNUSED(stop);
  for (int i = 0; i < maxIter; i++) {
    X_cc *= rho;
    X_cc += ATB_cc;
    ATA_llt.solveInPlace(X_cc);
  }
}

template <typename T, typename MatIn, typename VecIn, typename VecIn2,
          typename VecIn3>
T computeLargestEigenValue(const MatrixBase<MatIn> &G,
                           const MatrixBase<VecIn> &Gvpow,
                           const MatrixBase<VecIn2> &vpow,
                           const MatrixBase<VecIn3> &err_vpow, const T epsilon,
                           const int maxIter) {

  auto &Gvpow_ = Gvpow.const_cast_derived();
  auto &vpow_ = vpow.const_cast_derived();
  auto &err_vpow_ = err_vpow.const_cast_derived();
  return proxsuite::proxqp::dense::power_iteration<T>(
      G, Gvpow_, vpow_, err_vpow_, epsilon, maxIter);
}

template <typename T, typename MatIn, typename VecIn, typename VecIn2,
          typename VecIn3>
T computeSmallestEigenValue(const MatrixBase<MatIn> &G,
                            const MatrixBase<VecIn> &Gvpow,
                            const MatrixBase<VecIn2> &vpow,
                            const MatrixBase<VecIn3> &err_vpow,
                            const double epsilon, const int maxIter) {
  T max_eigen_value =
      computeLargestEigenValue(G, Gvpow, vpow, err_vpow, epsilon, maxIter);
  auto &Gvpow_ = Gvpow.const_cast_derived();
  auto &vpow_ = vpow.const_cast_derived();
  auto &err_vpow_ = err_vpow.const_cast_derived();
  return proxsuite::proxqp::dense::min_eigen_value_via_modified_power_iteration<
      T>(G, Gvpow_, vpow_, err_vpow_, max_eigen_value, epsilon, maxIter);
}

//
//
// DualBaseSolver
//
//

template <typename T, template <typename> class C>
const Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
DualBaseSolver<T, C>::getSolution() const {
  return lam_;
}

template <typename T, template <typename> class C>
void DualBaseSolver<T, C>::setJvpProblem(const ContactProblem<T, C> &prob,
                                         const Ref<const MatrixXs> &dg_dtheta) {
  ntheta_ = (int)dg_dtheta.cols();
  dlam_dtheta_.resize(3 * nc_, ntheta_);
  CONTACTBENCH_UNUSED(prob);
}

template <typename T, template <typename> class C>
void DualBaseSolver<T, C>::jvp_fd(ContactProblem<T, C> &prob,
                                  const Ref<const VectorXs> &lam0,
                                  const Ref<const MatrixXs> &dG_dtheta,
                                  const Ref<const MatrixXs> &dg_dtheta,
                                  const Ref<const MatrixXs> &dmu_dtheta,
                                  ContactSolverSettings<T> &settings,
                                  const T delta) {
  CONTACTBENCH_UNUSED(dG_dtheta);
  setJvpProblem(prob, dg_dtheta);
  dlam_dtheta_.setZero();
  // computes jvp via finite differences for right derivatives
  // solve should be called beforehand
  _solve(prob, lam0, settings);
  lam_fd_ = getSolution();
  // compute derivatives wrt Delassus matrix
  for (int i = 0; i < 3 * nc_; i++) {
    for (int j = 0; j < i + 1; j++) {
      prob.Del_->G_(i, j) += delta;
      prob.Del_->G_(j, i) += delta;
      _solve(prob, lam0, settings);
      prob.Del_->G_(i, j) += -delta;
      prob.Del_->G_(j, i) += -delta;
    }
  }
  // compute derivatives wrt g
  for (int i = 0; i < 3 * nc_; i++) {
    prob.g_(i) += delta;
    _solve(prob, lam0, settings);
    dlam_dtheta_ += (getSolution() - lam_fd_) * dg_dtheta.row(i) / delta;
    prob.g_(i) += -delta;
  }
  // compute derivatives wrt mus
  for (int i = 0; i < nc_; i++) {
    T delta_sign = 1.;
    if (prob.contact_constraints_[CAST_UL(i)].mu_ + delta > 1.) {
      // in this case we take the left derivatives
      delta_sign = -1.;
    }
    prob.contact_constraints_[CAST_UL(i)].mu_ += delta_sign * delta;
    _solve(prob, lam0, settings);
    dlam_dtheta_ += (getSolution() - lam_fd_) * dmu_dtheta.row(i) / delta;
    prob.contact_constraints_[CAST_UL(i)].mu_ += -delta_sign * delta;
  }
  lam_ = lam_fd_;
}

#ifdef DIFFCONTACT_WITH_CPPAD
template <typename T, template <typename> class C>
void DualBaseSolver<T, C>::jvp_cppad(ContactProblem<T, C> &prob,
                                     const Ref<const MatrixXs> &dG_dtheta,
                                     const Ref<const MatrixXs> &dg_dtheta,
                                     const Ref<const MatrixXs> &dmu_dtheta,
                                     ContactSolverSettings<T> &settings,
                                     const T eps_reg) {
  // WIP
  assert((std::is_same<T, double>::value &&
          "This function can only be used for scalar type double"));
  // This function is templated by the type T which should be double.
  // Inside of this function, we use cppad to compute the jvp of the
  // solution of the NCP problem with respect to the parameters of the
  // problem.
  // For this, we create another problem with the same parameters but with
  // variables of type ADScalar, and use the solver as well with type
  // ADScalar.

  // The parameters of the problem are called theta and the forwzrd mode
  // derivatives require to have dg_dtheta, dDel_dtheta, dmus_dtheta as inputs
  // vector. The final derivatives dlam_dtheta, dL_dg and dL_dmus are computed
  // by chain rule.

  typedef CppAD::AD<T> ADScalar;
  typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, 1> ADVector;
  typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, Eigen::Dynamic> ADMatrix;
  // typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, Eigen::Dynamic,
  //                       Eigen::RowMajor>
  //     ADRowMatrix;

  // double
  setJvpProblem(prob, dg_dtheta);
  MatrixXs dflatDel_dtheta(9 * nc_ * nc_, ntheta_);
  MatrixXs dx_dtheta(3 * nc_ + 9 * nc_ * nc_ + nc_, ntheta_);
  dx_dtheta << dg_dtheta, dflatDel_dtheta, dmu_dtheta;
  MatrixXs &Del = prob.Del_->G_;
  VectorXs &g = prob.g_;

  // cppad double
  std::vector<ADScalar> ad_mus;
  ad_mus.reserve(nc_);
  for (int i = 0; i < nc_; ++i) {
    ad_mus.push_back(prob.contact_constraints_[i].mu_);
  }
  ADMatrix ad_Del = Del.template cast<ADScalar>();
  ADVector ad_g = g.template cast<ADScalar>();
  ADVector ad_flat_Del = Eigen::Map<ADVector>(ad_Del.data(), ad_Del.size());
  ADVector x(ad_g.size() + ad_flat_Del.size() + ad_mus.size());

  // use cppad to compute the jacobian
  x << ad_g, ad_flat_Del, Eigen::Map<ADVector>(ad_mus.data(), ad_mus.size());
  CppAD::Independent(x);
  // TODO: start from the solution of a previous solve call and then run a few
  // itearations that are actually differentiated.

  ADVector g_ = x.head(ad_g.size());
  ADMatrix Del_ = Eigen::Map<ADMatrix>(x.data() + ad_g.size(), ad_Del.rows(),
                                       ad_Del.cols());
  std::vector<ADScalar> mus_ = std::vector<ADScalar>(
      x.data() + ad_g.size() + ad_flat_Del.size(),
      x.data() + ad_g.size() + ad_flat_Del.size() + ad_mus.size());

  ContactProblem<ADScalar, C> ad_prob(Del_, g_, mus_);
  NCPPGSSolver<ADScalar, C> ad_solver;
  ad_solver.setProblem(ad_prob);
  // ADVector x0 = ADVector::Zero(3 * nc_);
  ADVector x0 = getSolution().template cast<ADScalar>();
  // int maxIter = 100;
  ContactSolverSettings<ADScalar> ADSettings =
      settings.template cast<ADScalar>();
  ad_solver.solve(ad_prob, x0, ADSettings);
  ADVector lam = ad_solver.getSolution();
  CppAD::ADFun<T> ad_fun(x, lam);
  CPPAD_TESTVECTOR(T) dlam_dthetai;
  CPPAD_TESTVECTOR(T) dx_dthetai(static_cast<size_t>(x.rows()));
  for (int i = 0; i < ntheta_; ++i) {
    for (size_t j = 0; j < dx_dthetai.size(); ++j) {
      dx_dthetai[j] = dx_dtheta(j, i);
    }
    dlam_dthetai = ad_fun.Forward(1, dx_dthetai);
    dlam_dtheta_.col(i) = Eigen::Map<VectorXs>(dlam_dthetai.data(), lam.size());
  }
}
#endif

template <typename T, template <typename> class C>
const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
DualBaseSolver<T, C>::getdlamdtheta() const {
  // jvp has to be called beforehand
  return dlam_dtheta_;
}

template <typename T, template <typename> class C>
void DualBaseSolver<T, C>::setVjpProblem(const ContactProblem<T, C> &prob) {
  CONTACTBENCH_UNUSED(prob);
  dL_dDel_.resize(3 * nc_, 3 * nc_);
  dL_dg_.resize(3 * nc_);
  dL_dmus_.resize(nc_);
}

template <typename T, template <typename> class C>
void DualBaseSolver<T, C>::vjp_fd(ContactProblem<T, C> &prob,
                                  const Ref<const VectorXs> &lam0,
                                  const Ref<const VectorXs> &dL_dlam,
                                  ContactSolverSettings<T> &settings,
                                  const T delta) {
  setVjpProblem(prob);
  // computes vjp via finite differences for right derivatives
  // solve should be called beforehand
  _solve(prob, lam0, settings);
  lam_fd_ = getSolution();
  // compute derivatives wrt Delassus matrix
  for (int i = 0; i < 3 * nc_; i++) {
    for (int j = 0; j < i + 1; j++) {
      prob.Del_->G_(i, j) += delta;
      prob.Del_->G_(j, i) += delta;
      _solve(prob, lam0, settings);
      dL_dDel_(i, j) = dL_dlam.dot(getSolution() - lam_fd_) / (2 * delta);
      prob.Del_->G_(i, j) += -delta;
      prob.Del_->G_(j, i) += -delta;
    }
  }
  // compute derivatives wrt g
  for (int i = 0; i < 3 * nc_; i++) {
    prob.g_(i) += delta;
    _solve(prob, lam0, settings);
    dL_dg_(i) = dL_dlam.dot(getSolution() - lam_fd_) / delta;
    prob.g_(i) += -delta;
  }
  // compute derivatives wrt mus
  for (int i = 0; i < nc_; i++) {
    T delta_sign = 1.;
    if (prob.contact_constraints_[CAST_UL(i)].mu_ + delta > 1.) {
      // in this case we take the left derivatives
      delta_sign = -1.;
    }
    prob.contact_constraints_[CAST_UL(i)].mu_ += delta_sign * delta;
    _solve(prob, lam0, settings);
    dL_dmus_(i) = dL_dlam.dot(getSolution() - lam_fd_) / delta;
    prob.contact_constraints_[CAST_UL(i)].mu_ += -delta_sign * delta;
  }
  lam_ = lam_fd_;
}

#ifdef DIFFCONTACT_WITH_CPPAD
template <typename T, template <typename> class C>
void DualBaseSolver<T, C>::vjp_cppad(ContactProblem<T, C> &prob,
                                     ContactSolverSettings<T> &settings,
                                     const Ref<const VectorXs> &dL_dlam) {
  assert((std::is_same<T, double>::value &&
          "This function can only be used for scalar type double"));
  // This function is templated by the type T which should be double.
  // Inside of this function, we use cppad to compute the vjp of the
  // solution of the NCP problem with respect to the parameters of the problem.
  // For this, we create another problem with the same parameters but with
  // variables of type ADScalar, and use the solver as well with type ADScalar.

  // The parameters of the problem are the Del matrix, the g vector and the mu
  // vector. The final derivatives dL_dDel, dL_dg and dL_dmus are computed by
  // chain rule.
  // solve should be called beforehand.

  typedef CppAD::AD<T> ADScalar;
  typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, 1> ADVector;
  typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, Eigen::Dynamic> ADMatrix;
  // typedef Eigen::Matrix<ADScalar, Eigen::Dynamic, Eigen::Dynamic,
  //                       Eigen::RowMajor>
  //     ADRowMatrix;

  // double
  setVjpProblem(prob);
  dL_dDel_.setZero();
  dL_dg_.setZero();
  dL_dmus_.setZero();
  MatrixXs &Del = prob.Del_->G_;
  VectorXs &g = prob.g_;

  // cppad double
  std::vector<ADScalar> ad_mus;
  ad_mus.reserve(nc_);
  for (int i = 0; i < nc_; ++i) {
    ad_mus.push_back(prob.contact_constraints_[i].mu_);
  }
  ADMatrix ad_Del = Del.template cast<ADScalar>();
  ADVector ad_g = g.template cast<ADScalar>();
  ADVector ad_flat_Del = Eigen::Map<ADVector>(ad_Del.data(), ad_Del.size());
  ADVector x(ad_g.size() + ad_flat_Del.size() + ad_mus.size());

  // use cppad to compute the jacobian
  x << ad_g, ad_flat_Del, Eigen::Map<ADVector>(ad_mus.data(), ad_mus.size());
  CppAD::Independent(x);
  // TODO: start from the solution of a previous solve call and then run a few
  // iterations that are actually differentiated.

  ADVector g_ = x.head(ad_g.size());
  ADMatrix Del_ = Eigen::Map<ADMatrix>(x.data() + ad_g.size(), ad_Del.rows(),
                                       ad_Del.cols());
  std::vector<ADScalar> mus_ = std::vector<ADScalar>(
      x.data() + ad_g.size() + ad_flat_Del.size(),
      x.data() + ad_g.size() + ad_flat_Del.size() + ad_mus.size());

  ContactProblem<ADScalar, C> ad_prob(Del_, g_, mus_);
  NCPPGSSolver<ADScalar, C> ad_solver;
  ad_solver.setProblem(ad_prob);
  ADVector x0 = getSolution().template cast<ADScalar>();
  // ADVector x0 = ADVector::Zero(3 * nc_);
  // int maxIter = 100;
  ContactSolverSettings<ADScalar> ADSettings =
      settings.template cast<ADScalar>();
  ad_solver.solve(ad_prob, x0, ADSettings);
  ADVector lam = ad_solver.getSolution(); // TODO : this should be done during
                                          // the forward pass and stored
  CppAD::ADFun<T> ad_fun(x, lam);

  // CPPAD_TESTVECTOR(T) lam_(static_cast<size_t>(x.size()));
  CPPAD_TESTVECTOR(T) ad_dL_dlam(static_cast<size_t>(dL_dlam.rows()));
  for (size_t i = 0; i < dL_dlam.size(); ++i) {
    // lam_[i] = CppAD::Value(x[i]);
    ad_dL_dlam[i] = dL_dlam(i);
  }
  // CPPAD_TESTVECTOR(T) jacobian_result = ad_fun.Jacobian(lam_);
  // VectorXs dL_dx =
  //     dL_dlam.transpose() *
  //     Eigen::Map<RowMatrixXs>(jacobian_result.data(), 3 * nc_, x.size());
  CPPAD_TESTVECTOR(T) ad_dL_dx = ad_fun.Reverse(1, ad_dL_dlam);
  VectorXs dL_dx = Eigen::Map<VectorXs>(ad_dL_dx.data(), x.size());

  int g_size = ad_g.size();
  int Del_size = ad_flat_Del.size();
  int mu_size = ad_mus.size();

  dL_dg_ = dL_dx.segment(0, g_size);
  dL_dDel_ = Eigen::Map<MatrixXs>(dL_dx.data() + g_size, 3 * nc_, 3 * nc_);
  dL_dmus_ = Eigen::Map<VectorXs>(dL_dx.data() + g_size + Del_size, mu_size);
}
#endif

template <typename T, template <typename> class C>
const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
DualBaseSolver<T, C>::getdLdmus() const {
  return dL_dmus_;
}

template <typename T, template <typename> class C>
const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>
DualBaseSolver<T, C>::getdLdDel() const {
  return dL_dDel_;
}

template <typename T, template <typename> class C>
const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
DualBaseSolver<T, C>::getdLdg() const {
  return dL_dg_;
}

//
//
// NCPPGS
//
//

using namespace proxsuite::proxqp;
using std::nullopt;

template <typename T, template <typename> class C>
NCPPGSSolver<T, C>::NCPPGSSolver() : DualBaseSolver<T, C>() {
  timer_.stop();
  step_eps_ = 1e-12;
}

template <typename T, template <typename> class C>
void NCPPGSSolver<T, C>::setProblem(const ContactProblem<T, C> &prob) {
  nc_ = int(prob.g_.size() / 3);
  lam_.resize(3 * nc_);
  lam_pred_.resize(3 * nc_);
  dx_.resize(3 * nc_);
  v_.resize(3 * nc_);
  v_reg_.resize(3 * nc_);
  v_cor_.resize(3 * nc_);
  v_proj_.resize(3 * nc_);
}

template <typename T, template <typename> class C>
T NCPPGSSolver<T, C>::stoppingCriteria(const ContactProblem<T, C> &prob,
                                       const Ref<const VectorXs> &lam,
                                       const Ref<const VectorXs> &v) {
  prob.computeDeSaxceCorrection(v, v_cor_);
  ncp_comp_ = prob.computeConicComplementarity(lam, v_cor_);
  prob.projectDual(v_cor_, v_proj_);
  dual_feas_ = (v_cor_ - v_proj_).template lpNorm<Eigen::Infinity>();
  T stop = std::max(ncp_comp_, dual_feas_);
  return stop;
}

template <typename T, template <typename> class C>
T NCPPGSSolver<T, C>::relativeStoppingCriteria(
    const Ref<const VectorXs> &lam, const Ref<const VectorXs> &lam_pred) {
  T norm_pred = lam_pred.norm();
  dx_ = lam - lam_pred;
  T stop = dx_.norm() / norm_pred;
  return stop;
}

template <typename T, template <typename> class C>
void NCPPGSSolver<T, C>::addStatistics(T stop, T rel_stop, T ncp_comp,
                                       T sig_comp, T dual_feas) {
  stats_.addStop(stop);
  stats_.addRelStop(rel_stop);
  stats_.addComp(ncp_comp);
  stats_.addNcpComp(ncp_comp);
  stats_.addSigComp(sig_comp);
  stats_.addPrimFeas(0.);
  stats_.addDualFeas(dual_feas);
}

template <typename T, template <typename> class C>
bool NCPPGSSolver<T, C>::solve(const ContactProblem<T, C> &prob,
                               const Ref<const VectorXs> &lam0,
                               ContactSolverSettings<T> &settings,
                               const Ref<const VectorXs> &R_reg, T over_relax) {
  CONTACTBENCH_UNUSED(R_reg);
  CONTACTBENCH_NOMALLOC_BEGIN;
  assert(lam0.size() == 3 * nc_);
  if (settings.timings_) {
    timer_.start();
  }
  max_iter_ = settings.max_iter_;
  th_stop_ = settings.th_stop_;
  rel_th_stop_ = settings.rel_th_stop_;
  lam_ = lam0;
  stop_ = std::numeric_limits<double>::max();
  Vector3s lam_tmp_;
  if (settings.statistics_) {
    stats_.reset();
  }
  prob.Del_->computeChol(1e-9);
  prob.Del_->evaluateDel();
  for (int j = 0; j < max_iter_; j++) {
    lam_pred_ = lam_;
    for (int i = 0; i < nc_; i++) {
      prob.Del_->applyPerContactNormalOnTheRight(i, lam_, v_(3 * i + 2));
      v_(3 * i + 2) += prob.g_(3 * i + 2);
      v_reg_(3 * i + 2) = v_(3 * i + 2);
      lam_(3 * i + 2) += -(over_relax / (prob.Del_->G_(3 * i + 2, 3 * i + 2))) *
                         (v_(3 * i + 2));
      if (lam_(3 * i + 2) < 0) {
        lam_(3 * i + 2) = 0;
      }
      T min_dxy = std::max(std::min(prob.Del_->G_(3 * i, 3 * i),
                                    prob.Del_->G_(3 * i + 1, 3 * i + 1)),
                           step_eps_);
      lam_tmp_(2) = lam_(3 * i + 2);
      prob.Del_->applyPerContactTangentOnTheRight(
          i, lam_, v_.template segment<2>(3 * i));
      v_.template segment<2>(3 * i) += prob.g_.template segment<2>(3 * i);
      v_reg_.template segment<2>(3 * i) = v_.template segment<2>(3 * i);
      lam_tmp_.template head<2>() =
          lam_.template segment<2>(3 * i) -
          (over_relax / min_dxy) * (v_.template segment<2>(3 * i));
      prob.contact_constraints_[CAST_UL(i)].projectHorizontal(
          lam_tmp_, lam_.template segment<3>(3 * i));
    }
    stop_ = stoppingCriteria(prob, lam_, v_);
    rel_stop_ = relativeStoppingCriteria(lam_, lam_pred_);
    // const bool convergence_criteria_reached =
    //      check_expression_if_real<Scalar,false>(settings.absolute_residual <=
    //      settings.absolute_accuracy)
    //   || check_expression_if_real<Scalar,false>(settings.relative_residual <=
    //   settings.relative_accuracy); if(convergence_criteria_reached) // In the
    //   case where Scalar is not double, this will iterate for max_it.
    //     break;
    if (settings.statistics_) {
      addStatistics(stop_, rel_stop_, ncp_comp_,
                    prob.computeSignoriniComplementarity(lam_, v_), dual_feas_);
    }
    if (stop_ < th_stop_ || rel_stop_ < rel_th_stop_) {
      n_iter_ = j + 1;
      CONTACTBENCH_NOMALLOC_END;
      if (settings.timings_) {
        timer_.stop();
      }
      return true;
    }
  }
  n_iter_ = max_iter_;
  CONTACTBENCH_NOMALLOC_END;
  if (settings.timings_) {
    timer_.stop();
  }
  return false;
}

template <typename T, template <typename> class C>
bool NCPPGSSolver<T, C>::solve(const ContactProblem<T, C> &prob,
                               const Ref<const VectorXs> &lam0,
                               ContactSolverSettings<T> &settings, T over_relax,
                               T eps_reg) {
  return solve(prob, lam0, settings, VectorXs::Constant(3 * nc_, eps_reg),
               over_relax);
}

template <typename T, template <typename> class C>
bool NCPPGSSolver<T, C>::_solve(ContactProblem<T, C> &prob,
                                const Ref<const VectorXs> &lam0,
                                ContactSolverSettings<T> &settings) {
  return solve(prob, lam0, settings);
}

template <typename T, template <typename> class C>
const Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
NCPPGSSolver<T, C>::getDualSolution() const {
  return v_reg_;
}

template <typename T, template <typename> class C>
void NCPPGSSolver<T, C>::resetStats() {
  stats_.reset();
}

template <typename T, template <typename> class C>
void NCPPGSSolver<T, C>::setLCP(const ContactProblem<T, C> &prob) {
  CONTACTBENCH_UNUSED(prob);
  x_lcp_.resize(6 * nc_);
  Hvpow_.resize(6 * nc_);
  vpow_.resize(6 * nc_);
  err_vpow_.resize(6 * nc_);
  y_lcp_.resize(0);
  z_lcp_.resize(12 * nc_);
}

template <typename T, template <typename> class C>
T NCPPGSSolver<T, C>::computeSmallestEigenValue(const Ref<const MatrixXs> &H,
                                                const double epsilon,
                                                const int max_iter) {
  return contactbench::computeSmallestEigenValue<T>(H, Hvpow_, vpow_, err_vpow_,
                                                    epsilon, max_iter);
}

template <typename T, template <typename> class C>
bool NCPPGSSolver<T, C>::_polish(ContactProblem<T, C> &prob,
                                 const Ref<const VectorXs> &lam,
                                 ContactSolverSettings<T> &settings, T rho,
                                 T eps_reg) {
  // Call proxQP on the locally linearized problem
  prob.setLCP();
  prob.computeInscribedLCP(lam, eps_reg);
  setLCP(prob);

  prob.computeLCPSolution(lam_, x_lcp_);
  y_lcp_.setZero();
  z_lcp_.setZero();

  dim_ = 6 * nc_, neq_ = 0, nin_ = 12 * nc_;
  H_.resize(dim_, dim_);
  H_ = prob.A_ + (prob.A_.transpose());
  C_.resize(nin_, dim_);
  C_.topLeftCorner(dim_, dim_) = prob.A_;
  C_.bottomLeftCorner(dim_, dim_) = MatrixXd::Identity(dim_, dim_);

  l_.resize(nin_);
  l_.setZero();
  l_.head(dim_) = -prob.b_;

  qp_ = std::make_unique<dense::QP<double>>(dim_, neq_, nin_);
  qp_->settings.eps_abs = settings.th_stop_;
  qp_->settings.initial_guess = InitialGuessStatus::WARM_START;
  qp_->settings.verbose = false;
  qp_->settings.max_iter = settings.max_iter_;
  //  handling non convexity
  l_min_ = computeSmallestEigenValue(H_, 1e-10, 100);
  if (l_min_ < 0) {
    qp_->settings.default_rho = -l_min_ + rho;
  } else {
    qp_->settings.default_rho = rho;
  }
  qp_->init(H_, prob.b_, nullopt, nullopt, C_, l_, nullopt);

  qp_->solve(x_lcp_, y_lcp_, z_lcp_);
  bool success = qp_->results.info.status ==
                 proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED;

  x_lcp_ = qp_->results.x;
  lam_ = prob.UD_ * x_lcp_;
  prim_feas_ = qp_->results.info.pri_res;
  dual_feas_ = qp_->results.info.dua_res;
  n_iter_ = (int)qp_->results.info.iter;
  v_reg_ = prob.Del_->G_ * lam_ + prob.g_;
  lam_is_polished_ = true;
  return success;
}

template <typename T, template <typename> class C>
void NCPPGSSolver<T, C>::setApproxVjpProblem(const ContactProblem<T, C> &prob) {
  setVjpProblem(prob);
  dL_dA_.resize(6 * nc_, 6 * nc_);
  dL_db_.resize(6 * nc_);
  dL_dxyz_.resize(18 * nc_);
}

template <typename T, template <typename> class C>
void NCPPGSSolver<T, C>::vjp_approx(ContactProblem<T, C> &prob,
                                    const Ref<const VectorXs> &dL_dlam,
                                    ContactSolverSettings<T> &settings, T rho,
                                    const T eps_reg) {
  // computes approximate vjp via QPLayer on the locally linearized problem
  assert(dL_dlam.size() == 3 * nc_);
  if (!lam_is_polished_) {
    _polish(prob, getSolution(), settings, rho, eps_reg);
  }
  setApproxVjpProblem(prob);
  dL_dDel_.setZero();
  dL_dg_.setZero();
  dL_dmus_.setZero();
  dL_dxyz_.setZero();
  dL_dxyz_.head(dim_).noalias() = prob.UD_.transpose() * dL_dlam;
  dense::compute_backward<T>(*qp_, dL_dxyz_, settings.th_stop_);
  CONTACTBENCH_NOMALLOC_BEGIN;
  dL_dA_ = qp_->model.backward_data.dL_dH;
  dL_dA_ += qp_->model.backward_data.dL_dH.transpose();
  dL_dA_ += qp_->model.backward_data.dL_dC.topLeftCorner(dim_, dim_);
  dL_db_ = qp_->model.backward_data.dL_dg -
           qp_->model.backward_data.dL_dl.head(dim_);
  prob.computeLCPdLdDel(dL_dA_, dL_dDel_);
  prob.computeLCPdLdg(dL_db_, dL_dg_);
  prob.computeInscribedLCPdLdmus(dL_dA_, dL_dmus_);
  CONTACTBENCH_NOMALLOC_END;
}

//
//
// LCPBase
//
//

template <typename T>
T LCPBaseSolver<T>::computeLargestEigenValue(const Ref<const MatrixXs> &H,
                                             const T epsilon,
                                             const int max_iter) {
  return contactbench::computeLargestEigenValue(H, Hvpow_, vpow_, err_vpow_,
                                                epsilon, max_iter);
}

template <typename T>
T LCPBaseSolver<T>::computeSmallestEigenValue(const Ref<const MatrixXs> &H,
                                              const T epsilon,
                                              const int max_iter) {
  return contactbench::computeSmallestEigenValue<T>(H, Hvpow_, vpow_, err_vpow_,
                                                    epsilon, max_iter);
}

template <typename T>
bool LCPBaseSolver<T>::_solve_qp(ContactProblem<T, PyramidCone> &prob,
                                 const Ref<const VectorXs> &lam0,
                                 ContactSolverSettings<T> &settings,
                                 const Ref<const VectorXs> &R_reg, T rho) {
  // solve the LCP via the equivalent QP
  assert(lam0.size() == 3 * nc_);

  if (settings.timings_) {
    timer_.start();
  }
  max_iter_ = settings.max_iter_;
  th_stop_ = settings.th_stop_;
  rel_th_stop_ = settings.rel_th_stop_;
  lam_ = lam0;
  stop_ = std::numeric_limits<T>::max();
  if (settings.statistics_) {
    stats_.reset();
  }
  prob.Del_->computeChol(1e-9);
  prob.Del_->evaluateDel();

  prob.computeLCP(R_reg);

  CONTACTBENCH_NOMALLOC_BEGIN;

  prob.computeLCPSolution(lam_, x_lcp_);
  y_lcp_.setZero();
  z_lcp_.setZero();

  H_ = prob.A_ + (prob.A_.transpose());
  C_.topLeftCorner(dim_, dim_) = prob.A_;
  C_.bottomLeftCorner(dim_, dim_) = MatrixXs::Identity(dim_, dim_);

  l_.setZero();
  l_.head(dim_) = -prob.b_;

  qp_ = std::make_unique<dense::QP<T>>(dim_, neq_, nin_);
  qp_->settings.eps_abs = th_stop_;
  //  handling non convexity
  l_min_ = computeSmallestEigenValue(H_, 1e-10, 100);
  if (l_min_ < 0) {
    qp_->settings.default_rho = -l_min_ + rho;
  } else {
    qp_->settings.default_rho = rho;
  }
  // provide warm start
  qp_->settings.initial_guess = InitialGuessStatus::WARM_START;
  qp_->settings.verbose = false;
  qp_->settings.max_iter = max_iter_;
  qp_->init(H_, prob.b_, nullopt, nullopt, C_, l_, nullopt);

  qp_->solve(x_lcp_, y_lcp_, z_lcp_);
  bool success = qp_->results.info.status ==
                 proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED;
  x_lcp_ = qp_->results.x;
  lam_ = prob.UD_ * x_lcp_;

  prim_feas_ = qp_->results.info.pri_res;
  dual_feas_ = qp_->results.info.dua_res;
  n_iter_ = (int)qp_->results.info.iter;
  lam_is_polished_ = true;
  CONTACTBENCH_NOMALLOC_END;
  if (settings.timings_) {
    timer_.stop();
  }
  return success;
}

template <typename T>
bool LCPBaseSolver<T>::_solve_qp(ContactProblem<T, PyramidCone> &prob,
                                 const Ref<const VectorXs> &lam0,
                                 ContactSolverSettings<T> &settings, T rho,
                                 T eps_reg) {
  assert(eps_reg >= 0);
  R_reg_ = VectorXs::Constant(3 * nc_, eps_reg);
  return _solve_qp(prob, lam0, settings, R_reg_, rho);
}

template <typename T>
const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
LCPBaseSolver<T>::getQPH() const {
  return qp_->model.H;
}
template <typename T>
const Eigen::Matrix<T, Eigen::Dynamic, 1> LCPBaseSolver<T>::getQPg() const {
  return qp_->model.g;
}
template <typename T>
const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
LCPBaseSolver<T>::getQPC() const {
  return qp_->model.C;
}
template <typename T>
const Eigen::Matrix<T, Eigen::Dynamic, 1> LCPBaseSolver<T>::getQPl() const {
  return qp_->model.l;
}

template <typename T>
void LCPBaseSolver<T>::setVjpProblem(
    const ContactProblem<T, PyramidCone> &prob) {
  Base::setVjpProblem(prob);
  dL_dxyz_.resize(18 * nc_);
  dL_dlam_lcp_.resize(6 * nc_);
  dL_dA_.resize(6 * nc_, 6 * nc_);
  dL_db_.resize(6 * nc_);
}

template <typename T>
void LCPBaseSolver<T>::vjp(ContactProblem<T, PyramidCone> &prob,
                           const Ref<const VectorXs> &dL_dlam,
                           ContactSolverSettings<T> &settings) {
  assert(dL_dlam.size() == 3 * nc_);
  if (!lam_is_polished_) {
    _solve_qp(prob, getSolution(), settings);
  }
  setVjpProblem(prob);
  dL_dDel_.setZero();
  dL_dg_.setZero();
  dL_dmus_.setZero();
  dL_dxyz_.setZero();
  dL_dxyz_.head(dim_).noalias() = prob.UD_.transpose() * dL_dlam;
  dense::compute_backward<T>(*qp_, dL_dxyz_, settings.th_stop_);
  CONTACTBENCH_NOMALLOC_BEGIN;

  dL_dA_ = qp_->model.backward_data.dL_dH;
  dL_dA_ += qp_->model.backward_data.dL_dH.transpose();
  dL_dA_ += qp_->model.backward_data.dL_dC.topLeftCorner(dim_, dim_);
  dL_db_ = qp_->model.backward_data.dL_dg -
           qp_->model.backward_data.dL_dl.head(dim_);
  prob.computeLCPdLdDel(dL_dA_, dL_dDel_);
  prob.computeLCPdLdg(dL_db_, dL_dg_);
  prob.computeLCPdLdmus(dL_dA_, dL_dmus_);
  CONTACTBENCH_NOMALLOC_END;
}


//
//
// LCPQP
//
//
template <typename T> LCPQPSolver<T>::LCPQPSolver() : LCPBaseSolver<T>() {
  timer_.stop();
}

template <typename T>
void LCPQPSolver<T>::setProblem(ContactProblem<T, PyramidCone> &prob) {
  nc_ = int(prob.g_.size() / 3);
  lam_.resize(3 * nc_);
  x_lcp_.resize(6 * nc_);
  Hvpow_.resize(6 * nc_);
  vpow_.resize(6 * nc_);
  err_vpow_.resize(6 * nc_);
  y_lcp_.resize(0);
  z_lcp_.resize(12 * nc_);
  v_.resize(3 * nc_);
  dim_ = 6 * nc_, neq_ = 0, nin_ = 12 * nc_;
  H_.resize(dim_, dim_);
  C_.resize(nin_, dim_);
  l_.resize(nin_);
  prob.setLCP();
}

template <typename T>
bool LCPQPSolver<T>::solve(ContactProblem<T, PyramidCone> &prob,
                           const Ref<const VectorXs> &lam0,
                           ContactSolverSettings<T> &settings,
                           const Ref<const VectorXs> &R_reg, T rho) {
  return _solve_qp(prob, lam0, settings, R_reg, rho);
}

template <typename T>
bool LCPQPSolver<T>::solve(ContactProblem<T, PyramidCone> &prob,
                           const Ref<const VectorXs> &lam0,
                           ContactSolverSettings<T> &settings, T rho,
                           T eps_reg) {
  return solve(prob, lam0, settings, VectorXs::Constant(3 * nc_, eps_reg), rho);
}

template <typename T>
bool LCPQPSolver<T>::_solve(ContactProblem<T, PyramidCone> &prob,
                            const Ref<const VectorXs> &lam0,
                            ContactSolverSettings<T> &settings) {
  return solve(prob, lam0, settings);
}

template <typename T>
const Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
LCPQPSolver<T>::getDualSolution() const {
  // TODO: compute v_
  return v_;
}

template <typename T> void LCPQPSolver<T>::resetStats() { stats_.reset(); }

//
//
// NCPStagProj
//
//

template <template <typename> class C>
NCPStagProjSolver<C>::NCPStagProjSolver() : DualBaseSolver<double, C>() {
  timer_.stop();
}

template <template <typename> class C>
void NCPStagProjSolver<C>::setProblem(const ContactProblem<double, C> &prob) {
  nc_ = int(prob.g_.size() / 3);
  lam_.resize(3 * nc_);
  lam_pred_.resize(3 * nc_);
  dlam_.resize(3 * nc_);
  v_.resize(3 * nc_);
  v_n_.resize(nc_);
  v_t_.resize(2 * nc_);
  v_cor_.resize(3 * nc_);
  v_proj_.resize(3 * nc_);
  ind_n_.clear();
  ind_t_.clear();
  for (int i = 0; i < nc_; i++) {
    ind_n_.push_back(i * 3 + 2);
    ind_t_.push_back(i * 3);
    ind_t_.push_back(i * 3 + 1);
  }
  G_n_.resize(nc_, nc_);
  G_n_tild_.resize(nc_, nc_);
  G_n_inv_.resize(nc_, nc_);
  G_n_llt_.resize(nc_, nc_);
  Gnvnpow_.resize(nc_);
  vnpow_.resize(nc_);
  err_vnpow_.resize(nc_);
  G_t_.resize(2 * nc_, 2 * nc_);
  G_t_tild_.resize(2 * nc_, 2 * nc_);
  G_t_inv_.resize(2 * nc_, 2 * nc_);
  G_t_llt_.resize(2 * nc_, 2 * nc_);
  Gtvtpow_.resize(2 * nc_);
  vtpow_.resize(2 * nc_);
  err_vtpow_.resize(2 * nc_);
  G_nt_.resize(nc_, 2 * nc_);
  G_tn_.resize(2 * nc_, nc_);
  g_n_.resize(nc_);
  g_t_.resize(2 * nc_);
  gamma_n_.resize(nc_);
  gamma_t_.resize(2 * nc_);
  lam_n1_.resize(nc_);
  lam_t1_.resize(2 * nc_);
  lam_n1_pred_.resize(nc_);
  lam_n2_pred_.resize(nc_);
  lam_t1_pred_.resize(2 * nc_);
  lam_t2_pred_.resize(2 * nc_);
  dlam_n_.resize(nc_);
  dlam_t_.resize(2 * nc_);
}

template <template <typename> class C>
void NCPStagProjSolver<C>::evaluateNormalDelassus(
    const ContactProblem<double, C> &prob) {
  // prob.evaluateDel and setupProblem should be called beforehand
  G_n_ = prob.Del_->G_(ind_n_, ind_n_);
  G_nt_ = prob.Del_->G_(ind_n_, ind_t_);
}

template <template <typename> class C>
void NCPStagProjSolver<C>::evaluateTangentDelassus(
    const ContactProblem<double, C> &prob) {
  // prob.evaluateDel and setupProblem should be called beforehand
  G_t_ = prob.Del_->G_(ind_t_, ind_t_);
  G_tn_ = prob.Del_->G_(ind_t_, ind_n_);
}

template <template <typename> class C>
double
NCPStagProjSolver<C>::stoppingCriteria(const ContactProblem<double, C> &prob,
                                       const VectorXd &lam, const VectorXd &v) {
  prob.computeDeSaxceCorrection(v, v_cor_);
  comp_ = prob.computeConicComplementarity(lam, v_cor_);
  prob.projectDual(v_cor_, v_proj_);
  dual_feas_ = (v_cor_ - v_proj_).template lpNorm<Infinity>();
  double stop = std::max(comp_, dual_feas_);
  return stop;
}

template <template <typename> class C>
double
NCPStagProjSolver<C>::relativeStoppingCriteria(const VectorXd &lam,
                                               const VectorXd &lam_pred) {
  double norm_pred = lam_pred.norm();
  dlam_ = lam - lam_pred;
  double stop = dlam_.norm() / norm_pred;
  return stop;
}

template <template <typename> class C>
void NCPStagProjSolver<C>::addStatistics(double stop, double rel_stop,
                                         double comp, double sig_comp,
                                         double dual_feas) {
  stats_.addStop(stop);
  stats_.addRelStop(rel_stop);
  stats_.addComp(comp);
  stats_.addNcpComp(comp);
  stats_.addSigComp(sig_comp);
  stats_.addPrimFeas(0.);
  stats_.addDualFeas(dual_feas);
}

template <template <typename> class C>
double NCPStagProjSolver<C>::computeLargestEigenValueNormal(
    const MatrixXd &Gn, const double epsilon,
    const int max_iter) { // computes the biggest eigenvalue of Gn
  return contactbench::computeLargestEigenValue(Gn, Gnvnpow_, vnpow_,
                                                err_vnpow_, epsilon, max_iter);
}

template <template <typename> class C>
double NCPStagProjSolver<C>::computeLargestEigenValueTangent(
    const MatrixXd &Gt, const double epsilon,
    const int max_iter) { // computes the biggest eigenvalue of Gt
  return contactbench::computeLargestEigenValue(Gt, Gtvtpow_, vtpow_,
                                                err_vtpow_, epsilon, max_iter);
}

template <template <typename> class C>
void NCPStagProjSolver<C>::computeGntild(const MatrixXd &Gn, double rho) {
  G_n_tild_ = Gn;
  for (int i = 0; i < nc_; i++) {
    G_n_tild_(i, i) += rho;
  }
}

template <template <typename> class C>
void NCPStagProjSolver<C>::computeGttild(const MatrixXd &Gt, double rho) {
  G_t_tild_ = Gt;
  for (int i = 0; i < 2 * nc_; i++) {
    G_t_tild_(i, i) += rho;
  }
}

template <template <typename> class C>
void NCPStagProjSolver<C>::computeGninv(const MatrixXd &Gn) {
  CONTACTBENCH_UNUSED(Gn);
  G_n_inv_ = G_n_tild_.inverse();
}

template <template <typename> class C>
void NCPStagProjSolver<C>::computeGtinv(const MatrixXd &Gt) {
  CONTACTBENCH_UNUSED(Gt);
  G_t_inv_ = G_t_tild_.inverse();
}

template <template <typename> class C>
template <typename VecIn, typename VecIn2, typename VecIn3, typename VecIn4>
double NCPStagProjSolver<C>::stoppingCriteriaNormal(
    const MatrixBase<VecIn> &lam1, const MatrixBase<VecIn2> &lam2,
    const MatrixBase<VecIn3> &gamma, const MatrixBase<VecIn4> &v) {
  prim_feas_n_ = (lam1 - lam2).template lpNorm<Infinity>();
  dual_feas_n_ = (v + gamma).template lpNorm<Infinity>();
  comp_n_ = v.dot(lam2);
  double stop = std::max(prim_feas_n_, dual_feas_n_);
  return stop;
}

template <template <typename> class C>
template <typename VecIn, typename VecIn2>
double NCPStagProjSolver<C>::relativeStoppingCriteriaNormal(
    const MatrixBase<VecIn> &lam, const MatrixBase<VecIn2> &lam_pred) {
  double norm_pred = lam_pred.norm();
  dlam_n_ = lam - lam_pred;
  double stop = dlam_.norm() / norm_pred;
  return stop;
}

template <template <typename> class C>
template <typename VecIn, typename VecIn2, typename VecIn3, typename VecIn4>
double NCPStagProjSolver<C>::stoppingCriteriaTangent(
    const MatrixBase<VecIn> &lam1, const MatrixBase<VecIn2> &lam2,
    const MatrixBase<VecIn3> &gamma, const MatrixBase<VecIn4> &v) {
  prim_feas_t_ = (lam1 - lam2).template lpNorm<Infinity>();
  dual_feas_t_ = (v + gamma).template lpNorm<Infinity>();
  double stop = std::max(prim_feas_t_, dual_feas_t_);
  return stop;
}

template <template <typename> class C>
template <typename VecIn, typename VecIn2>
double NCPStagProjSolver<C>::relativeStoppingCriteriaTangent(
    const MatrixBase<VecIn> &lam, const MatrixBase<VecIn2> &lam_pred) {
  double norm_pred = lam_pred.norm();
  dlam_t_ = lam - lam_pred;
  double stop = dlam_.norm() / norm_pred;
  return stop;
}

template <template <typename> class C>
bool NCPStagProjSolver<C>::updateRhoNormal(double prim_feas, double dual_feas) {
  if (prim_feas / dual_feas > 10.) {
    rho_n_ *= std::pow(eigval_max_n_ / eigval_min_, 0.1); // increase rho
    return true;
  } else if (prim_feas / dual_feas < 0.1) {
    rho_n_ *= std::pow(eigval_min_ / eigval_max_n_, 0.1); // decresae rho
    return true;
  } else {
    return false;
  }
}

template <template <typename> class C>
bool NCPStagProjSolver<C>::updateRhoTangent(double prim_feas,
                                            double dual_feas) {
  if (prim_feas / dual_feas > 10.) {
    rho_t_ *= std::pow(eigval_max_t_ / eigval_min_, 0.1); // increase rho
    return true;
  } else if (prim_feas / dual_feas < 0.1) {
    rho_t_ *= std::pow(eigval_min_ / eigval_max_t_, 0.1); // decresae rho
    return true;
  } else {
    return false;
  }
}

template <template <typename> class C>
template <typename MatIn, typename VecIn, typename VecIn2, typename VecOut>
void NCPStagProjSolver<C>::solveNormal(
    const MatrixBase<MatIn> &Gn, const MatrixBase<VecIn> &gn,
    LLT<Ref<MatrixXd>> &llt_n, const MatrixBase<VecIn2> &lam0,
    const MatrixBase<VecOut> &lam_out, int maxIter, double th_stop,
    double rel_th_stop, double rho, double over_relax) {
  assert(lam0.size() == nc_);
  auto &lam_n2_ = lam_out.const_cast_derived();
  lam_n1_ = lam0;
  lam_n2_ = lam0;
  gamma_n_.setZero();
  for (int j = 0; j < maxIter; j++) {
    lam_n1_pred_ = lam_n1_;
    lam_n2_pred_ = lam_n2_;
    v_n_.noalias() = Gn * lam_n1_;
    v_n_ += gn;
    lam_n1_ = -(gn + gamma_n_ - rho_n_ * lam_n2_ - rho * lam_n1_pred_);
    llt_n.solveInPlace(lam_n1_);
    lam_n2_ = over_relax * lam_n1_ + (1 - over_relax) * lam_n2_pred_ +
              gamma_n_ / rho_n_;
    lam_n2_ = lam_n2_.cwiseMax(0.);
    gamma_n_ += rho_n_ * (over_relax * lam_n1_ +
                          (1 - over_relax) * lam_n2_pred_ - lam_n2_);
    stop_n_ = stoppingCriteriaNormal(lam_n1_, lam_n2_, gamma_n_, v_n_);
    rel_stop_n_ = relativeStoppingCriteriaNormal(lam_n2_, lam_n2_pred_);
    if (stop_n_ < th_stop || rel_stop_n_ < rel_th_stop) {
      return;
    }
    bool new_rho = updateRhoNormal(prim_feas_n_, dual_feas_n_);
    if (new_rho) {
      computeGntild(G_n_, rho_n_ + rho);
      G_n_llt_ = G_n_tild_;
      llt_n.compute(G_n_llt_);
    }
  }
}

template <template <typename> class C>
template <typename MatIn, typename VecIn, typename VecIn2, typename VecIn3,
          typename VecOut>
void NCPStagProjSolver<C>::solveTangent(
    const MatrixBase<MatIn> &Gt, const MatrixBase<VecIn> &gt,
    LLT<Ref<MatrixXd>> &llt_t, const ContactProblem<double, C> &prob,
    const MatrixBase<VecIn2> &lam_n, const MatrixBase<VecIn3> &lam0,
    const MatrixBase<VecOut> &lam_out, int maxIter, double th_stop,
    double rel_th_stop, double rho, double over_relax) {

  assert(lam0.size() == 2 * nc_);
  auto &lam_t2_ = lam_out.const_cast_derived();
  lam_t2_ = lam0;
  lam_t1_ = lam0;
  Vector2d lam_t2_tmp_;
  gamma_t_.setZero();
  for (int j = 0; j < maxIter; j++) {
    lam_t1_pred_ = lam_t1_;
    lam_t2_pred_ = lam_t2_;
    v_t_.noalias() = Gt * lam_t1_;
    v_t_ += gt;
    lam_t1_ = -(gt + gamma_t_ - rho_t_ * lam_t2_ - rho * lam_t1_pred_);
    llt_t.solveInPlace(lam_t1_);
    lam_t2_ = over_relax * lam_t1_ + (1 - over_relax) * lam_t2_pred_ +
              gamma_t_ / rho_t_;
    for (int i = 0; i < nc_; i++) {
      prob.contact_constraints_[CAST_UL(i)].projectHorizontal(
          lam_t2_.template segment<2>(2 * i), lam_n(i), lam_t2_tmp_);
      lam_t2_.template segment<2>(2 * i) = lam_t2_tmp_;
    }
    gamma_t_ += rho_t_ * (over_relax * lam_t1_ +
                          (1 - over_relax) * lam_t2_pred_ - lam_t2_);
    stop_t_ = stoppingCriteriaTangent(lam_t1_, lam_t2_, gamma_t_, v_t_);
    rel_stop_t_ = relativeStoppingCriteriaTangent(lam_t2_, lam_t2_pred_);
    if (stop_t_ < th_stop || rel_stop_t_ < rel_th_stop) {
      return;
    }
    bool new_rho = updateRhoTangent(prim_feas_t_, dual_feas_t_); // TODO
    if (new_rho) {
      computeGttild(G_t_, rho_t_ + rho);
      G_t_llt_ = G_t_tild_;
      llt_t.compute(G_t_llt_);
    }
  }
}

template <template <typename> class C>
bool NCPStagProjSolver<C>::solve(const ContactProblem<double, C> &prob,
                                 const VectorXd &lam0,
                                 ContactSolverSettings<double> &settings,
                                 int max_inner_iter, double rho,
                                 double over_relax) {
  CONTACTBENCH_NOMALLOC_BEGIN;
  assert(lam0.size() == 3 * nc_);
  if (settings.timings_) {
    timer_.start();
  }
  max_iter_ = settings.max_iter_;
  th_stop_ = settings.th_stop_;
  rel_th_stop_ = settings.rel_th_stop_;
  lam_ = lam0;
  stop_ = std::numeric_limits<double>::max();
  if (settings.statistics_) {
    stats_.reset();
  }
  prob.Del_->computeChol(1e-9);
  prob.Del_->evaluateDel();
  evaluateNormalDelassus(prob);
  evaluateTangentDelassus(prob);
  eigval_max_n_ = computeLargestEigenValueNormal(G_n_);
  eigval_max_t_ = computeLargestEigenValueTangent(G_t_);
  eigval_min_ = rho;
  rho_n_ = std::sqrt(eigval_max_n_ * eigval_min_) *
           (std::pow(eigval_max_n_ / eigval_min_, 0.4));
  rho_t_ = std::sqrt(eigval_max_t_ * eigval_min_) *
           (std::pow(eigval_max_t_ / eigval_min_, 0.4));
  computeGttild(G_t_, rho + rho_t_);
  G_t_llt_ = G_t_tild_;
  using MatRef = Eigen::Ref<Eigen::MatrixXd>;
  Eigen::LLT<MatRef> llt_t(G_t_llt_);
  computeGntild(G_n_, rho + rho_n_);
  G_n_llt_ = G_n_tild_;
  using MatRef = Eigen::Ref<Eigen::MatrixXd>;
  Eigen::LLT<MatRef> llt_n(G_n_llt_);
  // initialize g_n_
  lam_t1_ = lam_(ind_t_);
  lam_n1_ = prob.g_(ind_n_);
  g_n_.noalias() = G_nt_ * lam_t1_;
  g_n_ += lam_n1_;
  for (int j = 0; j < max_iter_; j++) {
    lam_pred_ = lam_;
    solveNormal(G_n_, g_n_, llt_n, lam_(ind_n_), lam_(ind_n_), max_inner_iter,
                th_stop_, rel_th_stop_, rho, over_relax);
    lam_t1_ = prob.g_(ind_t_);
    lam_n1_ = lam_(ind_n_);
    g_t_.noalias() = G_tn_ * lam_n1_;
    g_t_ += lam_t1_;
    solveTangent(G_t_, g_t_, llt_t, prob, lam_(ind_n_), lam_(ind_t_),
                 lam_(ind_t_), max_inner_iter, th_stop_, rel_th_stop_, rho,
                 over_relax);
    v_(ind_t_) =
        v_t_; // TODO: v_(ind_n_) should be updated with the new tangent forces
    lam_t1_ = lam_(ind_t_);
    // updating v_n_ and g_n_
    lam_n1_ = prob.g_(ind_n_);
    g_n_.noalias() = G_nt_ * lam_t1_;
    g_n_ += lam_n1_;
    lam_n1_ = lam_(ind_n_);
    v_n_.noalias() = G_n_ * lam_n1_;
    v_n_ += g_n_;
    v_(ind_n_) = v_n_;
    stop_ = stoppingCriteria(prob, lam_, v_);
    rel_stop_ = relativeStoppingCriteria(lam_, lam_pred_);
    if (settings.statistics_) {
      addStatistics(stop_, rel_stop_, stop_,
                    prob.computeSignoriniComplementarity(lam_, v_), dual_feas_);
    }
    if (stop_ < th_stop_ || rel_stop_ < rel_th_stop_) {
      n_iter_ = j + 1;
      CONTACTBENCH_NOMALLOC_END;
      if (settings.timings_) {
        timer_.stop();
      }
      return true;
    }
  }
  n_iter_ = max_iter_;
  CONTACTBENCH_NOMALLOC_END;
  if (settings.timings_) {
    timer_.stop();
  }
  return false;
}

template <template <typename> class C>
bool NCPStagProjSolver<C>::_solve(ContactProblem<double, C> &prob,
                                  const Ref<const VectorXd> &lam0,
                                  ContactSolverSettings<double> &settings) {
  return solve(prob, lam0, settings);
}

template <template <typename> class C>
const VectorXd &NCPStagProjSolver<C>::getDualNormal() const {
  return gamma_n_;
}

template <template <typename> class C>
const VectorXd &NCPStagProjSolver<C>::getDualTangent() const {
  return gamma_t_;
}

template <template <typename> class C> void NCPStagProjSolver<C>::resetStats() {
  stats_.reset();
}

// 
// 
// CCPBase
// 
// 

template <typename T>
void CCPBaseSolver<T>::setLCCP(const ContactProblem<T, IceCreamCone> &prob) {
  u_eq_.resize(prob.n_eq_tan_);
  y_qp_.resize(prob.n_eq_tan_);
  u_in_.resize(prob.n_in_tan_);
  z_qp_.resize(prob.n_in_tan_);
}

template <typename T>
bool CCPBaseSolver<T>::_polish(ContactProblem<T, IceCreamCone> &prob,
                               const Ref<const VectorXs> &lam,
                               ContactSolverSettings<T> &settings,
                               const Ref<const VectorXs> &R_reg) {
  CONTACTBENCH_UNUSED(R_reg);
  CONTACTBENCH_UNUSED(settings.rel_th_stop_);
  CONTACTBENCH_UNUSED(settings.statistics_);
  prob.setLCCP();
  prob.computeInscribedLCCP(lam);
  setLCCP(prob);
  // computing forward pass of proxqp
  u_in_.setZero();
  u_eq_.setZero();
  y_qp_.setZero();
  z_qp_.setZero();
  dim_ = 3 * nc_, neq_ = prob.n_eq_tan_, nin_ = prob.n_in_tan_;
  qp_ = std::make_unique<dense::QP<T>>(dim_, neq_, nin_);
  qp_->settings.eps_abs = settings.th_stop_;
  qp_->settings.initial_guess = InitialGuessStatus::WARM_START;
  qp_->settings.verbose = false;
  qp_->settings.max_iter = settings.max_iter_;
  // qp_->settings.max_iter = 10;
  // TODO: take R regularization into account !!!
  MatrixXs reg = R_reg_.asDiagonal();
  qp_->init(prob.Del_->G_ + reg, prob.g_, prob.getLinearEqConstraintMatrix(),
            u_eq_, prob.getLinearInConstraintMatrix(), nullopt, u_in_);
  prob.computeDualSolutionOfInscribed(lam, v_reg_, y_qp_, z_qp_,
                                      settings.th_stop_);
  qp_->solve(lam, y_qp_, z_qp_);
  lam_ = qp_->results.x;
  bool success = qp_->results.info.status ==
                 proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED;
  v_reg_ = (prob.Del_->G_ + reg) * lam_ + prob.g_;
  lam_is_polished_ = true;
  return success;
}

template <typename T>
bool CCPBaseSolver<T>::_polish(ContactProblem<T, IceCreamCone> &prob,
                               const Ref<const VectorXs> &lam,
                               ContactSolverSettings<T> &settings, T eps_reg) {
  CONTACTBENCH_UNUSED(eps_reg);
  return _polish(prob, lam, settings, R_reg_);
}

template <typename T>
void CCPBaseSolver<T>::setApproxVjpProblem(
    const ContactProblem<T, IceCreamCone> &prob) {
  // _polish should be called beforehand
  Base::setVjpProblem(prob);
  dL_dC_.resize(4 * nc_, 3 * nc_);
  dL_dxyz_.resize(3 * nc_ + prob.n_eq_tan_ + prob.n_in_tan_);
}

template <typename T>
void CCPBaseSolver<T>::vjp_approx(ContactProblem<T, IceCreamCone> &prob,
                                  const Ref<const VectorXs> &dL_dlam,
                                  ContactSolverSettings<T> &settings,
                                  const T eps_reg) {
  assert(dL_dlam.size() == 3 * nc_);
  if (!lam_is_polished_) {
    _polish(prob, getSolution(), settings, eps_reg);
  }
  setApproxVjpProblem(prob);
  dL_dDel_.setZero();
  dL_dg_.setZero();
  dL_dmus_.setZero();
  dL_dxyz_.setZero();
  dL_dxyz_.head(3 * nc_) = dL_dlam;
  dense::compute_backward<T>(*qp_, dL_dxyz_, settings.th_stop_);
  CONTACTBENCH_NOMALLOC_BEGIN;
  dL_dDel_ = qp_->model.backward_data.dL_dH;
  dL_dDel_ += qp_->model.backward_data.dL_dH.transpose();
  dL_dDel_ /= 2.;
  dL_dg_ = qp_->model.backward_data.dL_dg;
  dL_dC_ = qp_->model.backward_data.dL_dC;
  prob.computeInscribedLCCPdLdmus(dL_dC_, dL_dmus_);
  CONTACTBENCH_NOMALLOC_END;
}


//
//
// CCPPGS
//
//

template <typename T>
void CCPPGSSolver<T>::setProblem(const ContactProblem<T, IceCreamCone> &prob) {
  nc_ = int(prob.g_.size() / 3);
  lam_.resize(3 * nc_);
  lam_pred_.resize(3 * nc_);
  dx_.resize(3 * nc_);
  v_.resize(3 * nc_);
  v_proj_.resize(3 * nc_);
  v_reg_.resize(3 * nc_);
  R_reg_.resize(3 * nc_);
  lam_is_polished_ = false;
}

template <typename T>
T CCPPGSSolver<T>::relativeStoppingCriteria(
    const Ref<const VectorXs> &lam, const Ref<const VectorXs> &lam_pred) {
  T norm_pred = lam_pred.norm();
  dx_ = lam - lam_pred;
  T stop = dx_.norm() / norm_pred;
  return stop;
}

template <typename T>
T CCPPGSSolver<T>::stoppingCriteria(const ContactProblem<T, IceCreamCone> &prob,
                                    const Ref<const VectorXs> &lam,
                                    const Ref<const VectorXs> &v_reg) {
  prim_feas_ = 0.;
  prob.projectDual(v_reg, v_proj_);
  dual_feas_reg_ = (v_reg - v_proj_).template lpNorm<Infinity>();
  T stop = std::max(prim_feas_, dual_feas_reg_);
  comp_reg_ = prob.computeConicComplementarity(lam, v_reg);
  stop = std::max(comp_reg_, stop);
  return stop;
}

template <typename T>
void CCPPGSSolver<T>::computeStatistics(
    const ContactProblem<T, IceCreamCone> &prob, const Ref<const VectorXs> &lam,
    const Ref<const VectorXs> &v, const Ref<const VectorXs> &v_reg) {
  prim_feas_ = 0.;
  prob.projectDual(v, v_proj_);
  dual_feas_ = (v - v_proj_).template lpNorm<Infinity>();
  comp_ = prob.computeConicComplementarity(lam, v);
  sig_comp_ = prob.computeSignoriniComplementarity(lam, v);
  ncp_comp_ = prob.computeContactComplementarity(lam, v);
  prob.projectDual(v_reg, v_proj_);
  dual_feas_reg_ = (v_reg - v_proj_).template lpNorm<Infinity>();
  comp_reg_ = prob.computeConicComplementarity(lam, v_reg);
  sig_comp_reg_ = prob.computeSignoriniComplementarity(lam, v_reg);
  ncp_comp_reg_ = prob.computeContactComplementarity(lam, v_reg);
}

template <typename T>
void CCPPGSSolver<T>::addStatistics(T stop, T rel_stop, T comp, T prim_feas,
                                    T dual_feas, T sig_comp, T ncp_comp) {
  stats_.addStop(stop);
  stats_.addRelStop(rel_stop);
  stats_.addComp(comp);
  stats_.addPrimFeas(prim_feas);
  stats_.addDualFeas(dual_feas);
  stats_.addSigComp(sig_comp);
  stats_.addNcpComp(ncp_comp);
}

template <typename T>
bool CCPPGSSolver<T>::solve(ContactProblem<T, IceCreamCone> &prob,
                            const Ref<const VectorXs> &lam0,
                            ContactSolverSettings<T> &settings,
                            const Ref<const VectorXs> &R_reg, bool polish) {
  CONTACTBENCH_NOMALLOC_BEGIN;
  assert(lam0.size() == 3 * nc_);
  if (settings.timings_) {
    timer_.start();
  }
  max_iter_ = settings.max_iter_;
  th_stop_ = settings.th_stop_;
  rel_th_stop_ = settings.rel_th_stop_;
  lam_ = lam0;
  Vector3s lam_tmp_;
  stop_ = std::numeric_limits<T>::max();
  if (settings.statistics_) {
    stats_.reset();
  }
  prob.Del_->computeChol(1e-9);
  prob.Del_->evaluateDel();
  R_reg_ = R_reg;
  for (int j = 0; j < settings.max_iter_; j++) {
    lam_pred_ = lam_;
    for (int i = 0; i < nc_; i++) {
      prob.Del_->applyPerContactOnTheRight(i, lam_,
                                           v_.template segment<3>(3 * i));
      v_.template segment<3>(3 * i) += prob.g_.template segment<3>(3 * i);
      v_reg_.template segment<3>(3 * i) =
          v_.template segment<3>(3 * i) +
          R_reg_.template segment<3>(3 * i).cwiseProduct(
              lam_.template segment<3>(3 * i));
      v_reg_.template segment<3>(3 * i) +=
          prob.R_comp_.template segment<3>(3 * i).cwiseProduct(
              lam_.template segment<3>(3 * i));
      lam_tmp_ =
          lam_pred_.template segment<3>(3 * i) -
          (3. / std::max(prob.Del_->G_(3 * i, 3 * i) +
                             prob.Del_->G_(3 * i + 1, 3 * i + 1) +
                             prob.Del_->G_(3 * i + 2, 3 * i + 2) +
                             R_reg_(3 * i) + R_reg_(3 * i + 1) +
                             R_reg_(3 * i + 2) + prob.R_comp_(3 * i) +
                             prob.R_comp_(3 * i + 1) + prob.R_comp_(3 * i + 2),
                         step_eps_)) *
              (v_reg_.template segment<3>(3 * i));
      prob.contact_constraints_[CAST_UL(i)].project(
          lam_tmp_, lam_.template segment<3>(3 * i));
    }
    rel_stop_ = relativeStoppingCriteria(lam_, lam_pred_);
    stop_ = stoppingCriteria(
        prob, lam_,
        v_reg_); // TODO adapt computation to the case of regularization eps_reg
    if (settings.statistics_) {
      computeStatistics(prob, lam_, v_, v_reg_);
      addStatistics(stop_, rel_stop_, comp_reg_, prim_feas_, dual_feas_reg_,
                    sig_comp_reg_, ncp_comp_reg_);
    }
    if (stop_ < th_stop_ || rel_stop_ < rel_th_stop_) {
      n_iter_ = j + 1;
      if (polish) {
        bool success = this->_polish(prob, lam_, settings, R_reg);
        CONTACTBENCH_NOMALLOC_END;

        if (settings.timings_) {
          timer_.stop();
        }
        return success;
      } else {
        lam_is_polished_ = false;
        CONTACTBENCH_NOMALLOC_END;
        if (settings.timings_) {
          timer_.stop();
        }
        return true;
      }
    }
  }
  if (settings.timings_) {
    timer_.stop();
  }
  if (polish) {
    bool success = this->_polish(prob, lam_, settings, R_reg);
    CONTACTBENCH_NOMALLOC_END;
    return success;
  } else {
    n_iter_ = max_iter_;
    lam_is_polished_ = false;
    CONTACTBENCH_NOMALLOC_END;
    return false;
  }
}

template <typename T>
bool CCPPGSSolver<T>::solve(ContactProblem<T, IceCreamCone> &prob,
                            const Ref<const VectorXs> &lam0,
                            ContactSolverSettings<T> &settings, T eps_reg,
                            bool polish) {
  R_reg_ = VectorXs::Constant(3 * nc_, eps_reg);
  return solve(prob, lam0, settings, R_reg_, polish);
}

template <typename T>
const Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
CCPPGSSolver<T>::getDualSolution() const {
  return v_reg_;
}

template <typename T>
bool CCPPGSSolver<T>::_solve(ContactProblem<T, IceCreamCone> &prob,
                             const Ref<const VectorXs> &lam0,
                             ContactSolverSettings<T> &settings) {
  return solve(prob, lam0, settings);
}

template <typename T> void CCPPGSSolver<T>::resetStats() { stats_.reset(); }


//
//
// CCPADMM
//
//

template <typename T> CCPADMMSolver<T>::CCPADMMSolver() : CCPBaseSolver<T>(),
power_iteration_algo_(0) {
  timer_.stop();
}

template <typename T>
template <typename MatIn>
T CCPADMMSolver<T>::computeLargestEigenValue(
    const Eigen::MatrixBase<MatIn> &G, const T epsilon,
    const int max_iter) { // computes the largest eigenvalue of G
  return contactbench::computeLargestEigenValue(G, Gvpow_, vpow_, err_vpow_,
                                                epsilon, max_iter);
}

template <typename T>
void CCPADMMSolver<T>::setProblem(const ContactProblem<T, IceCreamCone> &prob) {
  nc_ = int(prob.g_.size() / 3);
  lam2_.resize(3 * nc_);
  lam2_pred_.resize(3 * nc_);
  lam_.resize(3 * nc_);
  lam_pred_.resize(3 * nc_);
  lam_or_.resize(3 * nc_);
  deltalam_.resize(3 * nc_);
  v_.resize(3 * nc_);
  v_reg_.resize(3 * nc_);
  v_proj_.resize(3 * nc_);
  gamma_.resize(3 * nc_);
  Ginv_.resize(3 * nc_, 3 * nc_);
  G_llt_.resize(3 * nc_, 3 * nc_);
  vpow_.resize(3 * nc_);
  err_vpow_.resize(3 * nc_);
  Gvpow_.resize(3 * nc_);
  R_reg_.resize(3 * nc_);
  rhos_.resize(3 * nc_);
  power_iteration_algo_ = PowerIterationAlgo(3 * nc_);
}

template <typename T>
bool CCPADMMSolver<T>::updateRho(T prim_feas, T dual_feas, T &rho_out) {
  if (prim_feas / dual_feas > 10.) {
    if (rho_out > 1e12) {
      return false;
    } else {
      rho_out *= std::pow(eigval_max_ / eigval_min_, 0.1); // increase rho
    }
    return true;
  } else if (prim_feas / dual_feas < 0.1) {
    if (rho_out < 1e-12) {
      return false;
    } else {
      rho_out *= std::pow(eigval_min_ / eigval_max_, 0.1); // decresae rho
      return true;
    }
  } else {
    return false;
  }
}

template <typename T>
void CCPADMMSolver<T>::computeStatistics(
    const ContactProblem<T, IceCreamCone> &prob, const Ref<const VectorXs> &lam,
    const Ref<const VectorXs> &lam2, const Ref<const VectorXs> &gamma,
    const Ref<const VectorXs> &v, const Ref<const VectorXs> &v_reg) {
  CONTACTBENCH_UNUSED(lam);
  dual_feas_ = (v + gamma).template lpNorm<Infinity>();
  comp_ = prob.computeConicComplementarity(lam2, v);
  comp_reg_ = prob.computeConicComplementarity(lam2, v_reg);
  ncp_comp_reg_ = prob.computeContactComplementarity(lam2, v_reg);
  ncp_comp_ = prob.computeContactComplementarity(lam2, v);
  sig_comp_ = prob.computeSignoriniComplementarity(lam2, v);
  sig_comp_reg_ = prob.computeSignoriniComplementarity(lam2, v_reg);
}

template <typename T>
T CCPADMMSolver<T>::stoppingCriteria(
    const ContactProblem<T, IceCreamCone> &prob, const Ref<const VectorXs> &lam,
    const Ref<const VectorXs> &lam_pred, const Ref<const VectorXs> &lam2,
    const Ref<const VectorXs> &lam2_pred, const Ref<const VectorXs> &gamma,
    const T rho_admm, const T rho) {
  prim_feas_ = (lam - lam2).template lpNorm<Infinity>();
  dual_feas_reg_ = (rho_admm * (lam2 - lam2_pred) + rho * (lam - lam_pred))
                       .template lpNorm<Infinity>();
  comp_reg_approx_ = prob.computeConicComplementarity(lam2, gamma);
  T stop = std::max(prim_feas_, dual_feas_reg_);
  stop = std::max(comp_reg_approx_, stop);
  return stop;
}

template <typename T>
T CCPADMMSolver<T>::relativeStoppingCriteria(
    const Ref<const VectorXs> &lam, const Ref<const VectorXs> &lam_pred) {
  T norm_pred = lam_pred.norm();
  deltalam_ = lam - lam_pred;
  T stop = deltalam_.norm() / norm_pred;
  return stop;
}

template <typename T>
void CCPADMMSolver<T>::addStatistics(T stop, T rel_stop, T comp, T prim_feas,
                                     T dual_feas, T sig_comp, T ncp_comp) {
  stats_.addStop(stop);
  stats_.addRelStop(rel_stop);
  stats_.addComp(comp);
  stats_.addPrimFeas(prim_feas);
  stats_.addDualFeas(dual_feas);
  stats_.addSigComp(sig_comp);
  stats_.addNcpComp(ncp_comp);
}

template <typename T>
void CCPADMMSolver<T>::computeChol(const ContactProblem<T, IceCreamCone> &prob,
                                   T rho, bool statistics) {
  prob.Del_->computeChol(rho);
  if (statistics) {
    prob.Del_->evaluateDel();
  }
}

template <typename T>
void CCPADMMSolver<T>::evaluateEigVals(
    const ContactProblem<T, IceCreamCone> &prob, T eps_reg, T rho) {
  evaluateEigValMin(prob, eps_reg, rho);
  this->power_iteration_algo_.run(prob.Del_.get());
  eigval_max_ = this->power_iteration_algo_.largest_eigen_value;
}

template <typename T>
void CCPADMMSolver<T>::evaluateEigVals(
    const ContactProblem<T, IceCreamCone> &prob,
    const Ref<const VectorXs> &R_reg, T rho) {

  evaluateEigValMin(prob, R_reg, rho);
  this->power_iteration_algo_.run(prob.Del_.get());
  eigval_max_ = this->power_iteration_algo_.largest_eigen_value;
}

template <typename T>
void CCPADMMSolver<T>::evaluateEigValMin(
    const ContactProblem<T, IceCreamCone> &prob, T eps_reg, T rho) {
  eigval_min_ = rho;
  prob.Del_->evaluateDiagDel();
  R_reg_ = eps_reg * prob.Del_->G_.diagonal();
  R_reg_ += prob.R_comp_;
  eigval_min_ += R_reg_.minCoeff();
}

template <typename T>
void CCPADMMSolver<T>::evaluateEigValMin(
    const ContactProblem<T, IceCreamCone> &prob,
    const Ref<const VectorXs> &R_reg, T rho) {
  assert((R_reg.array() >= 0.).all());
  eigval_min_ = rho;
  R_reg_ = prob.R_comp_ + R_reg;
  eigval_min_ += R_reg_.minCoeff();
}

template <typename T>
bool CCPADMMSolver<T>::_solve_impl(const ContactProblem<T, IceCreamCone> &prob,
                                   const Ref<const VectorXs> &lam0,
                                   const Ref<const VectorXs> &gamma0,
                                   ContactSolverSettings<T> &settings,
                                   const Ref<const VectorXs> &R_reg, T rho_admm,
                                   T rho, T over_relax) {
  CONTACTBENCH_NOMALLOC_BEGIN;
  assert(lam0.size() == 3 * nc_);
  if (settings.timings_) {
    timer_.start();
  }
  max_iter_ = settings.max_iter_;
  th_stop_ = settings.th_stop_;
  rel_th_stop_ = settings.rel_th_stop_;
  lam2_ = lam0;
  lam_ = lam0;
  gamma_ = gamma0;
  R_reg_ = R_reg;
  if (settings.statistics_) {
    stats_.reset();
  }
  rho_ = rho_admm;
  rhos_ = (prob.R_comp_.array() + R_reg_.array() + rho_ + rho).matrix();
  prob.Del_->updateChol(rhos_);

  stop_ = std::numeric_limits<T>::max();
  for (int j = 0; j < max_iter_; j++) {
    lam2_pred_ = lam2_;
    lam_pred_ = lam_;
    if (settings.statistics_) {
      prob.Del_->applyOnTheRight(lam2_, v_);
      v_ += prob.g_;
      v_reg_ = v_ + R_reg_.cwiseProduct(lam2_);
      v_reg_ += prob.R_comp_.cwiseProduct(lam2_);
      lam2_ = -(v_reg_ + gamma_ + rho_ * (lam2_pred_ - lam_));
      prob.Del_->solveInPlace(lam2_);
      lam2_ += lam2_pred_;
    } else {
      lam2_ = -(prob.g_ + gamma_ - rho_ * lam_ - rho * lam2_pred_);
      prob.Del_->solveInPlace(lam2_);
    }
    lam_or_ = over_relax * lam2_ + (1 - over_relax) * lam_pred_ + gamma_ / rho_;
    prob.project(lam_or_, lam_);
    gamma_ = rho_ * (lam_or_ - lam_);
    stop_ = stoppingCriteria(prob, lam2_, lam2_pred_, lam_, lam_pred_, gamma_,
                             rho_, rho);
    rel_stop_ = relativeStoppingCriteria(lam2_, lam2_pred_);
    if (settings.statistics_) {
      computeStatistics(prob, lam2_, lam_, gamma_, v_, v_reg_);
      addStatistics(stop_, rel_stop_, comp_reg_, prim_feas_, dual_feas_,
                    sig_comp_, ncp_comp_);
    }
    if (stop_ < th_stop_ || rel_stop_ < rel_th_stop_) {
      n_iter_ = j + 1;
      CONTACTBENCH_NOMALLOC_END;
      if (settings.timings_) {
        timer_.stop();
      }
      return true;
    }
    bool new_rho = updateRho(prim_feas_, dual_feas_reg_, rho_); // TODO
    if (new_rho) {
      rhos_ = (prob.R_comp_.array() + R_reg_.array() + rho_ + rho).matrix();
      prob.Del_->updateChol(rhos_);
    }
  }
  n_iter_ = max_iter_;
  CONTACTBENCH_NOMALLOC_END;
  if (settings.timings_) {
    timer_.stop();
  }
  return false;
}

template <typename T>
bool CCPADMMSolver<T>::_solve_impl(const ContactProblem<T, IceCreamCone> &prob,
                                   const Ref<const VectorXs> &lam0,
                                   const Ref<const VectorXs> &gamma0,
                                   ContactSolverSettings<T> &settings,
                                   T rho_admm, T rho, T over_relax, T eps_reg) {
  assert(eps_reg >= 0);
  R_reg_ = VectorXs::Constant(3 * nc_, eps_reg);
  return _solve_impl(prob, lam0, gamma0, settings, R_reg_, rho_admm, rho,
                     over_relax);
}

template <typename T>
bool CCPADMMSolver<T>::solve(const ContactProblem<T, IceCreamCone> &prob,
                             const Ref<const VectorXs> &lam0,
                             ContactSolverSettings<T> &settings, T rho,
                             T over_relax, T eps_reg) {
  computeChol(prob, rho, settings.statistics_);
  evaluateEigVals(prob, eps_reg, rho);
  T rho_admm = std::sqrt(eigval_max_ * eigval_min_) *
               (std::pow(eigval_max_ / eigval_min_, 0.4));
  return _solve_impl(prob, lam0, VectorXs::Zero(3 * nc_), settings, rho_admm,
                     rho, over_relax, eps_reg);
}

template <typename T>
bool CCPADMMSolver<T>::solve(const ContactProblem<T, IceCreamCone> &prob,
                             const Ref<const VectorXs> &lam0,
                             const Ref<const VectorXs> &gamma0,
                             ContactSolverSettings<T> &settings, T rho,
                             T over_relax, T eps_reg) {
  computeChol(prob, rho, settings.statistics_);
  evaluateEigVals(prob, eps_reg, rho);
  T rho_admm = std::sqrt(eigval_max_ * eigval_min_) *
               (std::pow(eigval_max_ / eigval_min_, 0.4));
  return _solve_impl(prob, lam0, gamma0, settings, rho_admm, rho, over_relax,
                     eps_reg);
}

template <typename T>
bool CCPADMMSolver<T>::solve(const ContactProblem<T, IceCreamCone> &prob,
                             const Ref<const VectorXs> &lam0,
                             const Ref<const VectorXs> &gamma0, T rho_admm,
                             ContactSolverSettings<T> &settings, T rho,
                             T over_relax, T eps_reg) {
  computeChol(prob, rho, settings.statistics_);
  evaluateEigVals(prob, eps_reg, rho);
  return _solve_impl(prob, lam0, gamma0, settings, rho_admm, rho, over_relax,
                     eps_reg);
}

template <typename T>
bool CCPADMMSolver<T>::solve(const ContactProblem<T, IceCreamCone> &prob,
                             const Ref<const VectorXs> &lam0,
                             const Ref<const VectorXs> &gamma0, T rho_admm,
                             T max_eigval, ContactSolverSettings<T> &settings,
                             T rho, T over_relax, T eps_reg) {
  computeChol(prob, rho, settings.statistics_);
  eigval_max_ = max_eigval;
  evaluateEigValMin(prob, eps_reg, rho);
  return _solve_impl(prob, lam0, gamma0, settings, rho_admm, rho, over_relax,
                     eps_reg);
}

template <typename T>
bool CCPADMMSolver<T>::solve(const ContactProblem<T, IceCreamCone> &prob,
                             const Ref<const VectorXs> &lam0,
                             ContactSolverSettings<T> &settings,
                             const Ref<const VectorXs> &R_reg, T rho,
                             T over_relax) {
  computeChol(prob, rho, settings.statistics_);
  evaluateEigVals(prob, R_reg, rho);
  T rho_admm = std::sqrt(eigval_max_ * eigval_min_) *
               (std::pow(eigval_max_ / eigval_min_, 0.4));
  return _solve_impl(prob, lam0, VectorXs::Zero(3 * nc_), settings, R_reg,
                     rho_admm, rho, over_relax);
}

template <typename T>
bool CCPADMMSolver<T>::solve(const ContactProblem<T, IceCreamCone> &prob,
                             const Ref<const VectorXs> &lam0,
                             const Ref<const VectorXs> &gamma0,
                             ContactSolverSettings<T> &settings,
                             const Ref<const VectorXs> &R_reg, T rho,
                             T over_relax) {
  computeChol(prob, rho, settings.statistics_);
  evaluateEigVals(prob, R_reg, rho);
  T rho_admm = std::sqrt(eigval_max_ * eigval_min_) *
               (std::pow(eigval_max_ / eigval_min_, 0.4));
  return _solve_impl(prob, lam0, gamma0, settings, R_reg, rho_admm, rho,
                     over_relax);
}

template <typename T>
bool CCPADMMSolver<T>::solve(const ContactProblem<T, IceCreamCone> &prob,
                             const Ref<const VectorXs> &lam0,
                             const Ref<const VectorXs> &gamma0, T rho_admm,
                             ContactSolverSettings<T> &settings,
                             const Ref<const VectorXs> &R_reg, T rho,
                             T over_relax) {
  computeChol(prob, rho, settings.statistics_);
  evaluateEigVals(prob, R_reg, rho);
  return _solve_impl(prob, lam0, gamma0, settings, R_reg, rho_admm, rho,
                     over_relax);
}

template <typename T>
bool CCPADMMSolver<T>::solve(const ContactProblem<T, IceCreamCone> &prob,
                             const Ref<const VectorXs> &lam0,
                             const Ref<const VectorXs> &gamma0, T rho_admm,
                             T max_eigval, ContactSolverSettings<T> &settings,
                             const Ref<const VectorXs> &R_reg, T rho,
                             T over_relax) {
  computeChol(prob, rho, settings.statistics_);
  eigval_max_ = max_eigval;
  evaluateEigValMin(prob, R_reg, rho);
  return _solve_impl(prob, lam0, gamma0, settings, R_reg, rho_admm, rho,
                     over_relax);
}

template <typename T>
bool CCPADMMSolver<T>::_solve(ContactProblem<T, IceCreamCone> &prob,
                              const Ref<const VectorXs> &lam0,
                              ContactSolverSettings<T> &settings) {
  return solve(prob, lam0, settings);
}

template <typename T>
const Eigen::Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
CCPADMMSolver<T>::getDualSolution() const {
  return gamma_;
}

template <typename T> void CCPADMMSolver<T>::resetStats() { stats_.reset(); }

//
//
// CCPADMMPrimal
//
//

template <typename T>
CCPADMMPrimalSolver<T>::CCPADMMPrimalSolver() : CCPADMMSolver<T>() {
  timer_.stop();
}

template <typename T>
void CCPADMMPrimalSolver<T>::setProblem(
    const ContactProblem<T, IceCreamCone> &prob) {
  nc_ = (int)(prob.vstar_.size() / 3);
  nv_ = (int)prob.M_.cols();
  x_.resize(nv_ + 3 * nc_);
  x_v_.resize(nv_ + 3 * nc_);
  x0_.resize(nv_ + 3 * nc_);
  x_pred_.resize(nv_ + 3 * nc_);
  dx_.resize(nv_ + 3 * nc_);
  z_.resize(3 * nc_);
  z_pred_.resize(3 * nc_);
  dz_.resize(3 * nc_);
  z_or_.resize(3 * nc_);
  v_.resize(3 * nc_);
  v_reg_.resize(3 * nc_);
  v_proj_.resize(3 * nc_);
  y_.resize(3 * nc_);
  lam_.resize(3 * nc_);
  P_.resize(nv_ + 3 * nc_, nv_ + 3 * nc_);
  P_.setZero();
  Ptild_.resize(nv_ + 3 * nc_, nv_ + 3 * nc_);
  Pinv_.resize(nv_ + 3 * nc_, nv_ + 3 * nc_);
  P_llt_.resize(nv_ + 3 * nc_, nv_ + 3 * nc_);
  llt_ = Eigen::LLT<MatrixXs>(nv_ + 3 * nc_);
  q_.resize(nv_ + 3 * nc_);
  A_.resize(3 * nc_, nv_ + 3 * nc_);
  ATA_.resize(nv_ + 3 * nc_, nv_ + 3 * nc_);
  vpow_.resize(nv_);
  err_vpow_.resize(nv_);
  Mvpow_.resize(nv_);
  R_reg_.resize(3 * nc_);
  rhos_.resize(3 * nc_);
}

template <typename T>
bool CCPADMMPrimalSolver<T>::updateRho(T prim_feas, T dual_feas, T &rho_out) {
  if (prim_feas / dual_feas > 10.) {
    rho_out *= std::pow(eigval_max_ / eigval_min_, 0.1); // increase rho
    return true;
  } else if (prim_feas / dual_feas < 0.1) {
    rho_out *= std::pow(eigval_min_ / eigval_max_, 0.1); // decresae rho
    return true;
  } else {
    return false;
  }
}

template <typename T>
template <typename MatIn>
void CCPADMMPrimalSolver<T>::evaluateEigVals(const Eigen::MatrixBase<MatIn> &P,
                                             T rho, T eps, int max_iter) {
  eigval_min_ = rho;
  eigval_max_ = 1. / R_reg_.minCoeff();
  eigval_max_ = std::max(
      contactbench::computeLargestEigenValue(P.topLeftCorner(nv_, nv_), Mvpow_,
                                             vpow_, err_vpow_, eps, max_iter),
      eigval_max_);
}

template <typename T>
void CCPADMMPrimalSolver<T>::computeStatistics(
    const ContactProblem<T, IceCreamCone> &prob, const Ref<const VectorXs> &x,
    const Ref<const VectorXs> &z, const Ref<const VectorXs> &y,
    const Ref<const VectorXs> &v, const Ref<const VectorXs> &v_reg) {
  CONTACTBENCH_UNUSED(x);
  CONTACTBENCH_UNUSED(z);
  dual_feas_ = (v + A_.transpose() * y).template lpNorm<Infinity>();
  dual_feas_reg_ = (v_reg + A_.transpose() * y).template lpNorm<Infinity>();
  comp_ = prob.computeConicComplementarity(-y, v);
  comp_reg_ = prob.computeConicComplementarity(-y, v_reg);
  sig_comp_ = prob.computeSignoriniComplementarity(-y, v);
  ncp_comp_ = prob.computeContactComplementarity(-y, v);
  sig_comp_reg_ = prob.computeSignoriniComplementarity(-y, v_reg);
  ncp_comp_reg_ = prob.computeContactComplementarity(-y, v_reg);
}

template <typename T>
void CCPADMMPrimalSolver<T>::addStatistics(T stop, T rel_stop, T comp,
                                           T prim_feas, T dual_feas, T sig_comp,
                                           T ncp_comp) {
  stats_.addStop(stop);
  stats_.addRelStop(rel_stop);
  stats_.addComp(comp);
  stats_.addPrimFeas(prim_feas);
  stats_.addDualFeas(dual_feas);
  stats_.addSigComp(sig_comp);
  stats_.addNcpComp(ncp_comp);
}

template <typename T>
T CCPADMMPrimalSolver<T>::stoppingCriteria(
    const ContactProblem<T, IceCreamCone> &prob, const Ref<const VectorXs> &x,
    const Ref<const VectorXs> &x_pred, const Ref<const VectorXs> &z,
    const Ref<const VectorXs> &z_pred, const Ref<const VectorXs> &y,
    const T rho_, const T rho) {
  prim_feas_ = (A_ * x - z).template lpNorm<Infinity>();
  dz_ = z - z_pred;
  dx_ = x - x_pred;
  dual_feas_reg_ =
      (rho_ * A_.transpose() * dz_ + rho * dx_).template lpNorm<Infinity>();
  comp_reg_approx_ = prob.computeConicComplementarity(A_ * x, -y);
  T stop = std::max(prim_feas_, dual_feas_reg_);
  stop = std::max(comp_reg_approx_, stop);
  return stop;
}

template <typename T>
void CCPADMMPrimalSolver<T>::setCompliance(
    const ContactProblem<T, IceCreamCone> &prob, T eps_reg) {
  prob.Del_->computeChol(1e-6);
  prob.Del_->evaluateDiagDel();
  R_reg_ = eps_reg * prob.Del_->G_.diagonal();
}

template <typename T>
void CCPADMMPrimalSolver<T>::setCompliance(const Ref<const VectorXs> &R_reg) {
  R_reg_ = R_reg;
}

template <typename T>
bool CCPADMMPrimalSolver<T>::_solve_impl(
    const ContactProblem<T, IceCreamCone> &prob, const Ref<const VectorXs> &x0,
    const Ref<const VectorXs> &y0, ContactSolverSettings<T> &settings,
    const Ref<const VectorXs> &R_reg, T rho_admm, T rho, T over_relax) {
  CONTACTBENCH_UNUSED(R_reg);
  CONTACTBENCH_NOMALLOC_BEGIN;
  assert(x0.size() == nv_ + 3 * nc_);
  assert(y0.size() == 3 * nc_);
  assert((R_reg.array() > 0).all());
  if (settings.timings_) {
    timer_.start();
  }
  max_iter_ = settings.max_iter_;
  th_stop_ = settings.th_stop_;
  rel_th_stop_ = settings.rel_th_stop_;
  y_ = y0;
  x_ = x0;
  z_.setZero();
  if (settings.statistics_) {
    stats_.reset();
  }
  rho_ = rho_admm;
  rhos_ = (R_reg.array() + rho_ + rho).matrix();
  prob.Del_->updateChol(rhos_);
  A_.topLeftCorner(3 * nc_, nv_) = prob.J_;
  A_.topRightCorner(3 * nc_, 3 * nc_) = -MatrixXs::Identity(3 * nc_, 3 * nc_);
  ATA_.noalias() = A_.transpose() * A_;
  Ptild_ = P_ + rho_ * ATA_;
  Ptild_.diagonal().array() += rho;
  P_llt_ = Ptild_;
  llt_.compute(P_llt_);
  q_.head(nv_).noalias() = -prob.M_ * prob.dqf_;
  q_.tail(3 * nc_) = -(prob.vstar_.array() / R_reg.array()).matrix();
  stop_ = std::numeric_limits<T>::max();
  for (int j = 0; j < max_iter_; j++) {
    x_pred_ = x_;
    z_pred_ = z_;
    if (settings.statistics_) {
      x_v_.noalias() = P_ * x_;
      x_v_ += q_;
      v_ = A_.topLeftCorner(3 * nc_, nv_) * x_.head(nv_);
      v_reg_ = v_;
      v_reg_ += -(y_.array() * R_reg.array()).matrix();
      x_ = x_v_;
      x_.noalias() += A_.transpose() * y_;
      x_.noalias() += rho_ * (ATA_ * x_pred_);
      x_.noalias() += -rho_ * (A_.transpose() * z_);
      x_ *= -1.;
      llt_.solveInPlace(x_);
      x_ += x_pred_;
    } else {
      //  TODO
      x_ = q_ - rho * x_pred_;
      y_ -= rho_ * z_;
      x_.noalias() += A_.transpose() * y_;
      y_ += rho_ * z_;
      x_ *= -1.;
      llt_.solveInPlace(x_);
    }
    z_or_.noalias() = A_ * x_;
    z_or_ = over_relax * z_or_ + (1 - over_relax) * z_pred_ + y_ / rho_;
    prob.projectDual(z_or_, z_);
    y_ = rho_ * (z_or_ - z_);
    stop_ = stoppingCriteria(prob, x_, x_pred_, z_, z_pred_, y_, rho_, rho);
    rel_stop_ = CCPADMMSolver<T>::relativeStoppingCriteria(x_, x_pred_);
    if (settings.statistics_) {
      computeStatistics(prob, x_, z_, y_, v_, v_reg_);
      addStatistics(stop_, rel_stop_, comp_reg_, prim_feas_, dual_feas_reg_,
                    sig_comp_reg_, ncp_comp_reg_);
    }
    if (stop_ < th_stop_ || rel_stop_ < rel_th_stop_) {
      n_iter_ = j + 1;
      CONTACTBENCH_NOMALLOC_END;
      if (settings.timings_) {
        timer_.stop();
      }
      return true;
    }
    bool new_rho = updateRho(prim_feas_, dual_feas_reg_, rho_); // TODO
    if (new_rho) {
      Ptild_ = P_ + rho_ * ATA_;
      Ptild_.diagonal().array() += rho;
      P_llt_ = Ptild_;
      llt_.compute(P_llt_);
    }
  }
  n_iter_ = max_iter_;
  CONTACTBENCH_NOMALLOC_END;
  if (settings.timings_) {
    timer_.stop();
  }
  return false;
}

template <typename T>
bool CCPADMMPrimalSolver<T>::_solve_impl(
    const ContactProblem<T, IceCreamCone> &prob, const Ref<const VectorXs> &x0,
    const Ref<const VectorXs> &y0, ContactSolverSettings<T> &settings,
    T rho_admm, T rho, T over_relax, T eps_reg) {
  assert(x0.size() == nv_ + 3 * nc_);
  assert(y0.size() == 3 * nc_);
  assert(eps_reg > 0);
  return _solve_impl(prob, x0, y0, settings,
                     VectorXs::Constant(3 * nc_, eps_reg), rho_admm, rho,
                     over_relax);
}

template <typename T>
bool CCPADMMPrimalSolver<T>::solve(const ContactProblem<T, IceCreamCone> &prob,
                                   const Ref<const VectorXs> &lam0,
                                   ContactSolverSettings<T> &settings,
                                   const Ref<const VectorXs> &R_reg, T rho,
                                   T over_relax) {
  setCompliance(R_reg);
  P_.topLeftCorner(nv_, nv_) = prob.M_;
  P_.bottomRightCorner(3 * nc_, 3 * nc_).diagonal() =
      (1. / R_reg_.array()).matrix();
  evaluateEigVals(P_, rho);
  T rho_admm = std::sqrt(eigval_max_ * eigval_min_) *
               (std::pow(eigval_max_ / eigval_min_, 0.4));
  // TODO: fix initial guess by avoiding malloc when prob.M_.inverse() and
  // rather call llt
  x0_.head(nv_) = prob.dqf_ + prob.M_.inverse() * prob.J_.transpose() * lam0;
  // TODO fix initial guess: it should not be contact point velocity
  x0_.tail(3 * nc_) = prob.J_ * x0_.head(nv_);
  return _solve_impl(prob, x0_, -lam0, settings, R_reg, rho_admm, rho,
                     over_relax);
}

template <typename T>
bool CCPADMMPrimalSolver<T>::solve(const ContactProblem<T, IceCreamCone> &prob,
                                   const Ref<const VectorXs> &lam0,
                                   ContactSolverSettings<T> &settings, T rho,
                                   T over_relax, T eps_reg) {
  return solve(prob, lam0, settings, VectorXs::Constant(3 * nc_, eps_reg), rho,
               over_relax);
}

template <typename T>
bool CCPADMMPrimalSolver<T>::_solve(ContactProblem<T, IceCreamCone> &prob,
                                    const Ref<const VectorXs> &lam0,
                                    ContactSolverSettings<T> &settings) {
  return solve(prob, lam0, settings);
}

template <typename T>
const Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
CCPADMMPrimalSolver<T>::getSolution() const {
  return x_.head(nv_);
}

template <typename T>
const Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
CCPADMMPrimalSolver<T>::getDualSolution() {
  //  TODO : fix this
  lam_ = -y_;
  return lam_;
}

//
//
// CCPNewtonPrimalSolver
//
//

template <typename T>
CCPNewtonPrimalSolver<T>::CCPNewtonPrimalSolver() : CCPBaseSolver<T>() {
  timer_.stop();
}

template <typename T>
void CCPNewtonPrimalSolver<T>::setProblem(
    const ContactProblem<T, IceCreamCone> &prob) {
  nc_ = (int)(prob.vstar_.size() / 3);
  nv_ = (int)prob.M_.cols();
  dq_.resize(nv_);
  dq_try_.resize(nv_);
  dq_pred_.resize(nv_);
  ddq_.resize(nv_);
  ddq2_.resize(nv_);
  ddq3_.resize(nv_);
  y_.resize(3 * nc_);
  y_tilde_.resize(3 * nc_);
  dvstar_.resize(3 * nc_);
  v_.resize(3 * nc_);
  v_proj_.resize(3 * nc_);
  grad_.resize(nv_);
  H_.resize(nv_, nv_);
  H_yy_.resize(3 * nc_, 3 * nc_);
  H_llt_.resize(3 * nc_, 3 * nc_);
  llt_ = Eigen::LLT<MatrixXs>(3 * nc_);
  lam_.resize(3 * nc_);
  R_reg_.resize(3 * nc_);
  R_sqrt_.resize(3 * nc_);
  M_diag_sqrt_.resize(nv_);
  M_diag_sqrt_ = prob.M_.diagonal().cwiseSqrt();
}

template <typename T>
void CCPNewtonPrimalSolver<T>::setCompliance(
    const ContactProblem<T, IceCreamCone> &prob, T eps_reg) {
  CONTACTBENCH_UNUSED(eps_reg);
  prob.Del_->computeChol(1e-6);
  prob.Del_->evaluateDel();
  for (int i = 0; i < nc_; i++) {
    R_reg_(3 * i + 2) =
        std::max(prob.Del_->G_.template block<3, 3>(3 * i, 3 * i).norm() /
                     (4. * M_PI * M_PI),
                 0.);
    R_reg_(3 * i) = 1e-3 * R_reg_(3 * i + 2);
    R_reg_(3 * i + 1) = R_reg_(3 * i);
  }
  R_sqrt_ = R_reg_.cwiseSqrt();
  mus_tilde_.clear();
  mus_hat_.clear();
  for (int i = 0; i < nc_; i++) {
    mus_tilde_.push_back(R_sqrt_(3 * i) / R_sqrt_(3 * i + 2) *
                         prob.contact_constraints_[CAST_UL(i)].mu_);
    mus_hat_.push_back(R_reg_(3 * i) / R_reg_(3 * i + 2) *
                       prob.contact_constraints_[CAST_UL(i)].mu_);
  }
}

template <typename T>
void CCPNewtonPrimalSolver<T>::setCompliance(const Ref<const VectorXs> &R_reg) {
  R_reg_ = R_reg;
}

template <typename T>
T CCPNewtonPrimalSolver<T>::relativeStoppingCriteria(
    const Ref<const VectorXs> &dq, const Ref<const VectorXs> &dq_pred) {
  T norm_pred = dq_pred.norm();
  ddq2_ = dq - dq_pred;
  T stop = ddq2_.norm() / norm_pred;
  return stop;
}

template <typename T>
void CCPNewtonPrimalSolver<T>::computeStatistics(
    const ContactProblem<T, IceCreamCone> &prob,
    const Ref<const VectorXs> &R_reg, const Ref<const VectorXs> &dq,
    const Ref<const VectorXs> &lam) {
  CONTACTBENCH_UNUSED(dq);
  v_ = prob.Del_->G_ * lam + prob.g_;
  comp_ = prob.computeConicComplementarity(lam, v_);
  sig_comp_ = prob.computeSignoriniComplementarity(lam, v_);
  ncp_comp_ = prob.computeContactComplementarity(lam, v_);
  v_ += lam.cwiseProduct(R_reg);
  prob.projectDual(v_, v_proj_);
  dual_feas_reg_ = (v_ - v_proj_).template lpNorm<Infinity>();
  prim_feas_ = 0.;
  comp_reg_ = prob.computeConicComplementarity(lam, v_);
  ncp_comp_reg_ = prob.computeContactComplementarity(lam, v_);
  sig_comp_reg_ = prob.computeSignoriniComplementarity(lam, v_);
}

template <typename T>
void CCPNewtonPrimalSolver<T>::addStatistics(T stop, T rel_stop, T comp,
                                             T sig_comp, T ncp_comp,
                                             T prim_feas, T dual_feas) {
  stats_.addStop(stop);
  stats_.addRelStop(rel_stop);
  stats_.addComp(comp);
  stats_.addDualFeas(dual_feas);
  stats_.addPrimFeas(prim_feas);
  stats_.addSigComp(sig_comp);
  stats_.addNcpComp(ncp_comp);
}

template <typename T>
template <typename VecIn, typename VecIn2, typename VecOut>
void CCPNewtonPrimalSolver<T>::complianceMap(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &y,
    const Eigen::MatrixBase<VecOut> &y_out) {
  auto &y_out_ = y_out.const_cast_derived();
  dvstar_ = y - prob.vstar_;
  y_out_ = -(dvstar_.array() / R.array()).matrix();
}

template <typename T>
template <typename VecIn, typename VecIn2, typename VecOut>
void CCPNewtonPrimalSolver<T>::projKR(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &y,
    const Eigen::MatrixBase<VecOut> &projy) {
  CONTACTBENCH_UNUSED(prob);
  auto &projy_ = projy.const_cast_derived();
  R_sqrt_ = R.cwiseSqrt();
  y_tilde_ = (R_sqrt_.array() * y.array()).matrix();
  ContactProblem<T, IceCreamCone>::project(mus_tilde_, y_tilde_, projy_);
  projy_ = (projy_.array() / R_sqrt_.array()).matrix();
}

template <typename T>
template <typename VecIn, typename VecIn2, typename VecIn3, typename VecOut>
void CCPNewtonPrimalSolver<T>::projKR(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &R_sqrt,
    const Eigen::MatrixBase<VecIn3> &y,
    const Eigen::MatrixBase<VecOut> &projy) {
  CONTACTBENCH_UNUSED(prob);
  CONTACTBENCH_UNUSED(R);
  auto &projy_ = projy.const_cast_derived();
  y_tilde_ = (R_sqrt.array() * y.array()).matrix();
  ContactProblem<T, IceCreamCone>::project(mus_tilde_, y_tilde_, projy_);
  projy_ = (projy_.array() / R_sqrt.array()).matrix();
}

template <typename T>
template <typename VecIn, typename VecIn2>
T CCPNewtonPrimalSolver<T>::regularizationCost(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &dq) {
  v_.noalias() = prob.J_ * dq;
  complianceMap(prob, R, v_, y_);
  projKR(prob, R, y_, lam_);
  return regularizationCost(prob, R, dq, y_, lam_);
}

template <typename T>
template <typename VecIn, typename VecIn2, typename VecIn3>
T CCPNewtonPrimalSolver<T>::regularizationCost(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &dq,
    const Eigen::MatrixBase<VecIn3> &y) {
  projKR(prob, R, y, lam_);
  return regularizationCost(prob, R, dq, y, lam_);
}

template <typename T>
template <typename VecIn, typename VecIn2, typename VecIn3, typename VecIn4>
T CCPNewtonPrimalSolver<T>::regularizationCost(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &dq,
    const Eigen::MatrixBase<VecIn3> &y, const Eigen::MatrixBase<VecIn4> &lam) {
  CONTACTBENCH_UNUSED(prob);
  CONTACTBENCH_UNUSED(dq);
  CONTACTBENCH_UNUSED(y);
  T cost = 0.;
  cost += 0.5 * lam.dot((R.array() * lam.array()).matrix());
  return cost;
}

template <typename T>
template <typename VecIn, typename VecIn2>
T CCPNewtonPrimalSolver<T>::unconstrainedCost(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &dq) {
  T cost = 0.;
  ddq2_ = dq - prob.dqf_;
  ddq3_.noalias() = prob.M_ * ddq2_;
  cost += 0.5 * ddq2_.dot(ddq3_);
  cost += regularizationCost(prob, R, dq);
  return cost;
}

template <typename T>
template <typename VecIn, typename VecIn2, typename VecIn3>
T CCPNewtonPrimalSolver<T>::unconstrainedCost(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &dq,
    const Eigen::MatrixBase<VecIn3> &y) {
  T cost = 0.;
  ddq2_ = dq - prob.dqf_;
  ddq3_.noalias() = prob.M_ * ddq2_;
  cost += 0.5 * ddq2_.dot(ddq3_);
  cost += regularizationCost(prob, R, dq, y);
  return cost;
}

template <typename T>
template <typename VecIn, typename VecIn2, typename VecIn3, typename VecIn4>
T CCPNewtonPrimalSolver<T>::unconstrainedCost(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &dq,
    const Eigen::MatrixBase<VecIn3> &y, const Eigen::MatrixBase<VecIn4> &lam) {
  T cost = 0.;
  ddq2_ = dq - prob.dqf_;
  ddq3_.noalias() = prob.M_ * ddq2_;
  cost += 0.5 * ddq2_.dot(ddq3_);
  cost += regularizationCost(prob, R, dq, y, lam);
  return cost;
}

template <typename T>
template <typename VecIn, typename VecIn2, typename VecOut>
void CCPNewtonPrimalSolver<T>::computeRegularizationGrad(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &dq,
    const Eigen::MatrixBase<VecOut> &grad_out) {
  complianceMap(prob, R, prob.J_ * dq, y_);
  projKR(prob, R, y_, lam_);
  computeRegularizationGrad(prob, R, dq, y_, lam_, grad_out);
}

template <typename T>
template <typename VecIn, typename VecIn2, typename VecIn3, typename VecOut>
void CCPNewtonPrimalSolver<T>::computeRegularizationGrad(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &dq,
    const Eigen::MatrixBase<VecIn3> &y,
    const Eigen::MatrixBase<VecOut> &grad_out) {
  projKR(prob, R, y, lam_);
  computeRegularizationGrad(prob, R, dq, y, lam_, grad_out);
}

template <typename T>
template <typename VecIn, typename VecIn2, typename VecIn3, typename VecIn4,
          typename VecOut>
void CCPNewtonPrimalSolver<T>::computeRegularizationGrad(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &dq,
    const Eigen::MatrixBase<VecIn3> &y, const Eigen::MatrixBase<VecIn4> &lam,
    const Eigen::MatrixBase<VecOut> &grad_out) {
  CONTACTBENCH_UNUSED(R);
  CONTACTBENCH_UNUSED(dq);
  CONTACTBENCH_UNUSED(y);
  auto &grad_out_ = grad_out.const_cast_derived();
  grad_out_.noalias() = -prob.J_.transpose() * lam;
}

template <typename T>
template <typename VecIn, typename VecIn2, typename VecOut>
void CCPNewtonPrimalSolver<T>::computeGrad(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &dq,
    const Eigen::MatrixBase<VecOut> &grad_out) {
  auto &grad_out_ = grad_out.const_cast_derived();
  computeRegularizationGrad(prob, R, dq, grad_out_);
  grad_out_.noalias() += prob.M_ * (dq - prob.dqf_);
}

template <typename T>
template <typename VecIn, typename VecIn2, typename VecIn3, typename VecOut>
void CCPNewtonPrimalSolver<T>::computeGrad(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &dq,
    const Eigen::MatrixBase<VecIn3> &y,
    const Eigen::MatrixBase<VecOut> &grad_out) {
  auto &grad_out_ = grad_out.const_cast_derived();
  computeRegularizationGrad(prob, R, dq, y, grad_out_);
  grad_out_.noalias() += prob.M_ * (dq - prob.dqf_);
}

template <typename T>
template <typename VecIn, typename VecIn2, typename VecIn3, typename VecIn4,
          typename VecOut>
void CCPNewtonPrimalSolver<T>::computeGrad(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &dq,
    const Eigen::MatrixBase<VecIn3> &y, const Eigen::MatrixBase<VecIn4> &lam,
    const Eigen::MatrixBase<VecOut> &grad_out) {
  auto &grad_out_ = grad_out.const_cast_derived();
  computeRegularizationGrad(prob, R, dq, y, lam, grad_out_);
  grad_out_.noalias() += prob.M_ * (dq - prob.dqf_);
}

template <typename T>
template <typename VecIn, typename VecIn2, typename VecIn3, typename MatOut>
void CCPNewtonPrimalSolver<T>::computeHessReg(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &y,
    const Eigen::MatrixBase<VecIn3> &y_tilde,
    const Eigen::MatrixBase<MatOut> &H_out) {
  CONTACTBENCH_UNUSED(prob);
  auto &H_out_ = H_out.const_cast_derived();
  H_out_.setZero();
  for (int i = 0; i < nc_; i++) {
    bool stiction = IceCreamCone<T>::isInside(
        mus_tilde_[CAST_UL(i)], y_tilde.template segment<3>(3 * i), 0.);
    if (stiction) { // sticking
      H_out_.template block<3, 3>(3 * i, 3 * i).diagonal() =
          R.template segment<3>(3 * i);
    } else {
      double y_r = y.template segment<2>(3 * i).norm();
      double y_tilde_r = y_tilde.template segment<2>(3 * i).norm();
      bool breaking =
          (y_tilde(3 * i + 2) < -mus_tilde_[CAST_UL(i)] * y_tilde_r);
      if (breaking) { // breaking
        continue;
      } else { // sliding
        t_dir_ = y.template segment<2>(3 * i) / y_r;
        P_t_ = t_dir_ * t_dir_.transpose();
        P_t_perp_ = Matrix2s::Identity() - P_t_;
        double sy = mus_hat_[CAST_UL(i)] * y_r + y(3 * i + 2);
        H_out_.template block<2, 2>(3 * i, 3 * i) = mus_hat_[CAST_UL(i)] * P_t_;
        H_out_.template block<2, 2>(3 * i, 3 * i) += sy * P_t_perp_ / y_r;
        H_out_.template block<2, 2>(3 * i, 3 * i) *= mus_hat_[CAST_UL(i)];
        H_out_.template block<2, 1>(3 * i, 3 * i + 2) =
            mus_hat_[CAST_UL(i)] * t_dir_;
        H_out_.template block<1, 2>(3 * i + 2, 3 * i) =
            mus_hat_[CAST_UL(i)] * t_dir_;
        H_out_(3 * i + 2, 3 * i + 2) = 1.;
        H_out_.template block<3, 3>(3 * i, 3 * i) *=
            R(3 * i + 2) /
            (1. + mus_tilde_[CAST_UL(i)] * mus_tilde_[CAST_UL(i)]);
      }
    }
  }
}

template <typename T>
template <typename VecIn, typename VecIn2, typename MatOut>
void CCPNewtonPrimalSolver<T>::computeHess(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &dq,
    const Eigen::MatrixBase<MatOut> &H_out) {
  auto &H_out_ = H_out.const_cast_derived();
  H_out_ = prob.M_;
  complianceMap(prob, R, prob.J_ * dq, y_);
  R_sqrt_ = R.cwiseSqrt();
  y_tilde_ = (R_sqrt_.array() * y_.array()).matrix();
  computeHessReg(prob, R, y_, y_tilde_, H_yy_);
  H_out_.noalias() += prob.J_.transpose() *
                      (1. / R_reg_.array()).matrix().asDiagonal() * H_yy_ *
                      (1. / R_reg_.array()).matrix().asDiagonal() * prob.J_;
}

template <typename T>
template <typename VecIn, typename VecIn2, typename VecIn3, typename VecIn4,
          typename MatOut>
void CCPNewtonPrimalSolver<T>::computeHess(
    const ContactProblem<T, IceCreamCone> &prob,
    const Eigen::MatrixBase<VecIn> &R, const Eigen::MatrixBase<VecIn2> &dq,
    const Eigen::MatrixBase<VecIn3> &y,
    const Eigen::MatrixBase<VecIn4> &y_tilde,
    const Eigen::MatrixBase<MatOut> &H_out) {
  CONTACTBENCH_UNUSED(dq);
  auto &H_out_ = H_out.const_cast_derived();
  H_out_ = prob.M_;
  computeHessReg(prob, R, y, y_tilde, H_yy_);
  H_out_.noalias() += prob.J_.transpose() *
                      (1. / R_reg_.array()).matrix().asDiagonal() * H_yy_ *
                      (1. / R_reg_.array()).matrix().asDiagonal() * prob.J_;
}

template <typename T>
template <typename VecIn, typename MatIn, typename VecOut>
void CCPNewtonPrimalSolver<T>::computeDescentDirection(
    const Eigen::MatrixBase<VecIn> &grad, const Eigen::MatrixBase<MatIn> &H,
    const Eigen::MatrixBase<VecOut> &ddq) {
  CONTACTBENCH_UNUSED(H);
  auto &ddq_cc_ = ddq.const_cast_derived();
  H_llt_ = H_;
  llt_.compute(H_llt_);
  ddq_cc_ = -llt_.solve(grad);
}

template <typename T>
bool CCPNewtonPrimalSolver<T>::_solve_impl(
    const ContactProblem<T, IceCreamCone> &prob, const Ref<const VectorXs> &dq0,
    ContactSolverSettings<T> &settings, const Ref<const VectorXs> &R_reg) {
  CONTACTBENCH_UNUSED(R_reg);
  CONTACTBENCH_NOMALLOC_BEGIN;
  assert(dq0.size() == nv_);
  assert((R_reg.array() > 0).all());
  if (settings.timings_) {
    timer_.start();
  }
  max_iter_ = settings.max_iter_;
  th_stop_ = settings.th_stop_;
  rel_th_stop_ = settings.rel_th_stop_;
  if (settings.statistics_) {
    stats_.reset();
  }
  dq_ = dq0;
  cost_ = unconstrainedCost(prob, R_reg_, dq_);
  for (int j = 0; j < max_iter_; j++) {
    dq_pred_ = dq_;
    computeGrad(prob, R_reg_, dq_, y_, lam_, grad_);
    computeHess(prob, R_reg_, dq_, y_, y_tilde_, H_);
    computeDescentDirection(grad_, H_, ddq_);
    // Linesearch
    alpha_ = 1.25;
    double exp_dec = 1e-4 * alpha_ * grad_.dot(ddq_);
    for (int k = 0; k < max_iter_; k++) {
      dq_try_ = dq_ + alpha_ * ddq_;
      cost_try_ = unconstrainedCost(prob, R_reg_, dq_try_);
      if (cost_try_ < cost_ + exp_dec) {
        break;
      } else {
        alpha_ *= 0.8;
        exp_dec *= 0.8;
      }
    }
    dq_ = dq_try_;
    cost_ = cost_try_;
    stop_ = (grad_.array() / M_diag_sqrt_.array())
                .matrix()
                .template lpNorm<Infinity>();
    rel_stop_ = relativeStoppingCriteria(dq_, dq_pred_);
    n_iter_ = j + 1;
    if (settings.statistics_) {
      computeStatistics(prob, R_reg_, dq_, lam_);
      addStatistics(stop_, rel_stop_, comp_reg_, sig_comp_reg_, ncp_comp_reg_,
                    prim_feas_, dual_feas_reg_);
    }
    if (stop_ < th_stop_ || rel_stop_ < rel_th_stop_) {
      CONTACTBENCH_NOMALLOC_END;
      if (settings.timings_) {
        timer_.stop();
      }
      return true;
    }
  }
  CONTACTBENCH_NOMALLOC_END;
  if (settings.timings_) {
    timer_.stop();
  }
  return false;
}

template <typename T>
bool CCPNewtonPrimalSolver<T>::_solve_impl(
    const ContactProblem<T, IceCreamCone> &prob, const Ref<const VectorXs> &dq0,
    ContactSolverSettings<T> &settings, T eps_reg) {
  return _solve_impl(prob, dq0, settings, VectorXs::Constant(3 * nc_, eps_reg));
}

template <typename T>
bool CCPNewtonPrimalSolver<T>::solve(
    const ContactProblem<T, IceCreamCone> &prob,
    const Ref<const VectorXs> &lam0, ContactSolverSettings<T> &settings,
    const Ref<const VectorXs> &R_reg) {
  setCompliance(prob, 0.);
  dq_ = prob.dqf_ + prob.M_.inverse() * prob.J_.transpose() * lam0;
  return _solve_impl(prob, dq_, settings, R_reg);
}

template <typename T>
bool CCPNewtonPrimalSolver<T>::solve(
    const ContactProblem<T, IceCreamCone> &prob,
    const Ref<const VectorXs> &lam0, ContactSolverSettings<T> &settings,
    T eps_reg) {
  return solve(prob, lam0, settings, VectorXs::Constant(3 * nc_, eps_reg));
}

template <typename T>
bool CCPNewtonPrimalSolver<T>::_solve(ContactProblem<T, IceCreamCone> &prob,
                                      const Ref<const VectorXs> &lam0,
                                      ContactSolverSettings<T> &settings) {
  return solve(prob, lam0, settings);
}

template <typename T>
const Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
CCPNewtonPrimalSolver<T>::getCompliance() const {
  return R_reg_;
}

template <typename T>
const Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
CCPNewtonPrimalSolver<T>::getSolution() const {
  return dq_;
}

template <typename T>
const Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
CCPNewtonPrimalSolver<T>::getDualSolution() const {
  return lam_;
}

//
//
// RaisimSolver
//
//

template <typename T>
void RaisimSolver<T>::setProblem(const ContactProblem<T, IceCreamCone> &prob) {
  nc_ = int(prob.g_.size() / 3);
  Glam_.resize(3, nc_ * nc_);
  Ginv_.resize(3, 3 * nc_);
  c_.resize(3, nc_);
  lam_.resize(3 * nc_);
  lam_proj_.resize(3 * nc_);
  lam_pred_.resize(3 * nc_);
  dlam_.resize(3 * nc_);
  v_.resize(3 * nc_);
  v_cor_.resize(3 * nc_);
  v_proj_.resize(3 * nc_);
}

template <typename T>
void RaisimSolver<T>::computeGinv(const Ref<const MatrixXs> &G) {
  for (int i = 0; i < nc_; i++) {
    Ginv_.template middleCols<3>(3 * i) =
        G.template block<3, 3>(3 * i, 3 * i).inverse();
  }
}

template <typename T>
const Ref<const Eigen::Matrix<T, 3, 3>> RaisimSolver<T>::getGinv(int i) const {
  return Ginv_.template middleCols<3>(3 * i);
}

template <typename T>
void RaisimSolver<T>::computeGlam(const Ref<const MatrixXs> &G,
                                  const Ref<const VectorXs> &lam) {
  Vector3s Glamij;
  for (int i = 0; i < nc_; i++) {
    for (int j = 0; j < nc_; j++) {
      Glamij =
          G.template block<3, 3>(3 * i, 3 * j) * lam.template segment<3>(3 * j);
      setGlam(i, j, Glamij);
    }
  }
}

template <typename T>
const Eigen::Ref<const Eigen::Matrix<T, 3, 1>>
RaisimSolver<T>::getGlam(int i, int j) const {
  assert(i < nc_);
  assert(j < nc_);
  return Glam_.col(nc_ * i + j);
}

template <typename T>
void RaisimSolver<T>::setGlam(int i, int j, Vector3s Glamij) {
  assert(i < nc_);
  assert(j < nc_);
  Glam_.col(i * nc_ + j) = Glamij;
}

template <typename T>
template <typename VecIn>
void RaisimSolver<T>::updateGlam(int j, const Ref<const MatrixXs> &G,
                                 const Eigen::MatrixBase<VecIn> &lamj) {
  assert(j < nc_);
  Vector3s Glamij;
  for (int i = 0; i < nc_; i++) {
    Glamij = G.template block<3, 3>(3 * i, 3 * j) * lamj;
    setGlam(i, j, Glamij);
  }
}

template <typename T>
void RaisimSolver<T>::computeC(const Ref<const MatrixXs> &G,
                               const Ref<const VectorXs> &b,
                               const Ref<const VectorXs> &lam) {
  c_ = MatrixXs::Zero(3, nc_);
  computeGlam(G, lam);
  for (int i = 0; i < nc_; i++) {
    for (int j = 0; j < nc_; j++) {
      if (j != i) {
        c_.col(i) += getGlam(i, j);
      }
    }
    c_.col(i) += b.template segment<3>(3 * i);
    v_.template segment<3>(3 * i) = c_.col(i) + getGlam(i, i);
  }
}

template <typename T>
template <typename VecIn>
void RaisimSolver<T>::updateC(int j, const Ref<const MatrixXs> &G,
                              const Ref<const VectorXs> &b,
                              const Eigen::MatrixBase<VecIn> &lamj) {
  CONTACTBENCH_UNUSED(b);
  for (int i = 0; i < nc_; i++) {
    if (j != i) {
      c_.col(i) += -getGlam(i, j);
    }
  }
  updateGlam(j, G, lamj);
  for (int i = 0; i < nc_; i++) {
    if (j != i) {
      c_.col(i) += getGlam(i, j);
    }
    v_.template segment<3>(3 * i) = c_.col(i) + getGlam(i, i);
  }
}

template <typename T>
const Ref<const Eigen::Matrix<T, 3, 1>> RaisimSolver<T>::getC(int i) {
  return c_.col(i);
}

template <typename T>
template <typename MatIn, typename VecIn, typename VecOut>
void RaisimSolver<T>::computeLamV0(const MatrixBase<MatIn> &Ginvi,
                                   const MatrixBase<VecIn> &ci,
                                   const MatrixBase<VecOut> &lam_out) {
  const_cast<MatrixBase<VecOut> &>(lam_out).noalias() = -Ginvi * ci;
}

template <typename T>
template <typename MatIn>
void RaisimSolver<T>::computeH1Grad(const MatrixBase<MatIn> &G,
                                    Ref<Vector3s> grad_out) {
  grad_out = G.row(2).transpose();
}

template <typename T>
template <typename VecIn>
void RaisimSolver<T>::computeH2Grad(const T mu, const MatrixBase<VecIn> &lam,
                                    Ref<Vector3s> grad_out) {
  grad_out(0) = 2 * lam(0);
  grad_out(1) = 2 * lam(1);
  grad_out(2) = -2 * (mu * mu) * lam(2);
}

template <typename T>
template <typename MatIn, typename VecIn>
void RaisimSolver<T>::computeEta(const MatrixBase<MatIn> &G, const T mu,
                                 const MatrixBase<VecIn> &lam,
                                 Ref<Vector3s> eta_out) {
  Vector3s gradH1;
  Vector3s gradH2;
  computeH1Grad(G, gradH1);
  computeH2Grad(mu, lam, gradH2);
  eta_out = gradH1.cross(gradH2);
}

template <typename T>
template <typename MatIn>
void RaisimSolver<T>::computeEta(const MatrixBase<MatIn> &G,
                                 const Ref<const Vector3s> &gradH2,
                                 Ref<Vector3s> eta_out) {
  Vector3s gradH1;
  computeH1Grad(G, gradH1);
  eta_out = gradH1.cross(gradH2);
}

template <typename T>
template <typename MatIn>
void RaisimSolver<T>::computeGbar(const MatrixBase<MatIn> &G,
                                  Matrix<T, 3, 2> &Gbar) {
  Matrix<T, 1, 2> Gtild;
  Gtild(0, 0) = G(2, 0) / G(2, 2);
  Gtild(0, 1) = G(2, 1) / G(2, 2);
  Gbar = G.template leftCols<2>() - G.col(2) * Gtild;
}

template <typename T>
template <typename MatIn, typename VecIn>
void RaisimSolver<T>::computeCbar(const MatrixBase<MatIn> &G,
                                  const MatrixBase<VecIn> &c,
                                  Ref<Vector3s> cbar) {
  cbar = c - (c(2) / G(2, 2)) * G.col(2);
}

template <typename T>
template <typename MatIn, typename VecIn, typename VecIn2>
T RaisimSolver<T>::computeGradTheta(const MatrixBase<MatIn> &G,
                                    const MatrixBase<VecIn> &c, const T mu,
                                    const MatrixBase<VecIn2> &lam) {
  Vector3s eta;
  computeEta(G, mu, lam, eta);
  Matrix<T, 3, 2> Gbar;
  computeGbar(G, Gbar);
  Vector3s cbar;
  computeCbar(G, c, cbar);
  T grad_theta = eta.dot(Gbar * lam.template head<2>() + cbar);
  return grad_theta;
}

template <typename T>
template <typename MatIn, typename VecIn, typename VecIn2>
T RaisimSolver<T>::computeGradTheta(const MatrixBase<MatIn> &G,
                                    const MatrixBase<VecIn> &c,
                                    const MatrixBase<VecIn2> &lam,
                                    const Ref<const Vector3s> &gradH2) {
  Vector3s eta;
  computeEta(G, gradH2, eta);
  Matrix<T, 3, 2> Gbar;
  computeGbar(G, Gbar);
  Vector3s cbar;
  computeCbar(G, c, cbar);
  T grad_theta = eta.dot(Gbar * lam.template head<2>() + cbar);
  return grad_theta;
}

template <typename T> T RaisimSolver<T>::computeLamZ(const T mu, const T r) {
  return r / mu;
}

template <typename T>
template <typename MatIn, typename VecIn>
T RaisimSolver<T>::computeR(const MatrixBase<MatIn> &G,
                            const MatrixBase<VecIn> &c, const T mu,
                            const T theta) {
  T r = -c(2) /
        (G(2, 2) / mu + G(2, 0) * std::cos(theta) + G(2, 1) * std::sin(theta));
  return r;
}

template <typename T>
template <typename VecOut>
void RaisimSolver<T>::computeLam(const T r, const T theta, const T lamZ,
                                 MatrixBase<VecOut> &lam_out) {
  lam_out(0) = r * std::cos(theta);
  lam_out(1) = r * std::sin(theta);
  lam_out(2) = lamZ;
}

template <typename T>
template <typename VecIn>
T RaisimSolver<T>::computeTheta(const MatrixBase<VecIn> &lam) {
  T theta = std::atan2(lam(1), lam(0));
  return theta;
}

template <typename T>
template <typename MatIn, typename MatIn2, typename VecIn>
void RaisimSolver<T>::bisectionStep(const MatrixBase<MatIn> &G,
                                    const MatrixBase<MatIn2> &Ginv,
                                    const MatrixBase<VecIn> &c, const T mu,
                                    const Ref<const Vector3s> &lam_v0,
                                    Ref<Vector3s> lam_out, int max_iter, T th,
                                    T beta1, T beta2, T beta3) {
  CONTACTBENCH_UNUSED(Ginv);
  T theta = computeTheta(lam_v0);
  T r = computeR(G, c, mu, theta);
  T lamZ = computeLamZ(mu, r);
  computeLam(r, theta, lamZ, lam_out);
  T d0 = computeGradTheta(G, c, mu, lam_out);
  T dtheta;
  if (d0 >= 0) {
    dtheta = -beta1;
  } else {
    dtheta = beta1;
  }
  T theta_pred;
  Vector3s lam_pred, gradH2;
  n_iter_mod_ = max_iter;
  for (int i = 0; i < max_iter; i++) { // initial stepping
    theta_pred = theta;
    lam_pred = lam_out;
    theta += dtheta;
    r = computeR(G, c, mu, theta);
    lamZ = computeLamZ(mu, r);
    computeLam(r, theta, lamZ, lam_out);
    computeH2Grad(mu, lam_out, gradH2);
    if (gradH2.dot(lam_v0 - lam_out) < 0 || r < 0) {
      dtheta *= beta2;
      theta = theta_pred;
      lam_out = lam_pred; // TODO: should check this
    } else {
      T grad = computeGradTheta(G, c, lam_out, gradH2);
      if (grad * d0 > 0) {
        dtheta *= beta3;
      } else {
        n_iter_mod_ = i + 1;
        break;
      }
    }
  } 
  T theta_bis, r_bis, lamZ_bis, grad_bis;
  Vector3s lam_bis;
  n_iter_bis_ = max_iter;
  for (int i = 0; i < max_iter; i++) { // bisection
    theta_bis = .5 * (theta + theta_pred);
    r_bis = computeR(G, c, mu, theta_bis);
    lamZ_bis = computeLamZ(mu, r_bis);
    computeLam(r_bis, theta_bis, lamZ_bis, lam_bis);
    grad_bis = computeGradTheta(G, c, mu, lam_bis);
    if (grad_bis * d0 > 0) {
      theta_pred = theta_bis;
      lam_pred = lam_bis;
    } else {
      theta = theta_bis;
      lam_out = lam_bis;
    }
    if ((lam_out - lam_pred).template lpNorm<Infinity>() < th) {
      n_iter_bis_ = i + 1;
      break;
    }
  }
}
template <typename T>
T RaisimSolver<T>::stoppingCriteria(const ContactProblem<T, IceCreamCone> &prob,
                                    const Ref<const VectorXs> &lam,
                                    const Ref<const VectorXs> &v) {
  prob.computeDeSaxceCorrection(v, v_cor_);
  comp_ = prob.computeConicComplementarity(lam, v_cor_);
  prob.project(lam, lam_proj_);
  prim_feas_ = (lam - lam_proj_).template lpNorm<Infinity>();
  prob.projectDual(v_cor_, v_proj_);
  dual_feas_ = (v_cor_ - v_proj_).template lpNorm<Infinity>();
  T stop = std::max(comp_, prim_feas_);
  stop = std::max(stop, dual_feas_);
  return stop;
}

template <typename T>
T RaisimSolver<T>::relativeStoppingCriteria(
    const Ref<const VectorXs> &lam, const Ref<const VectorXs> &lam_pred) {
  T norm_pred = lam_pred.norm();
  dlam_ = lam - lam_pred;
  T stop = dlam_.norm() / norm_pred;
  return stop;
}

template <typename T>
void RaisimSolver<T>::computeStatistics(
    const ContactProblem<T, IceCreamCone> &prob, const Ref<const VectorXs> &lam,
    const Ref<const VectorXs> &v) {
  sig_comp_ = prob.computeSignoriniComplementarity(lam, v);
  ncp_comp_ = prob.computeContactComplementarity(lam, v);
}

template <typename T>
void RaisimSolver<T>::addStatistics(T stop, T rel_stop, T comp, T sig_comp,
                                    T ncp_comp, T prim_feas, T dual_feas) {
  stats_.addStop(stop);
  stats_.addRelStop(rel_stop);
  stats_.addComp(comp);
  stats_.addNcpComp(ncp_comp);
  stats_.addSigComp(sig_comp);
  stats_.addPrimFeas(prim_feas);
  stats_.addDualFeas(dual_feas);
}

template <typename T>
bool RaisimSolver<T>::solve(const ContactProblem<T, IceCreamCone> &prob,
                            const Ref<const VectorXs> &lam0,
                            ContactSolverSettings<T> &settings, T eps_reg,
                            T alpha, T alpha_min, T beta1, T beta2, T beta3,
                            T gamma, T th) {

  return solve(prob, lam0, settings, VectorXs::Constant(3 * nc_, eps_reg),
               alpha, alpha_min, beta1, beta2, beta3, gamma, th);
}

template <typename T>
bool RaisimSolver<T>::solve(const ContactProblem<T, IceCreamCone> &prob,
                            const Ref<const VectorXs> &lam0,
                            ContactSolverSettings<T> &settings,
                            const Ref<const VectorXs> &R_reg, T alpha,
                            T alpha_min, T beta1, T beta2, T beta3, T gamma,
                            T th) {
  CONTACTBENCH_UNUSED(R_reg);
  CONTACTBENCH_NOMALLOC_BEGIN;
  assert(lam0.size() == 3 * nc_);
  if (settings.timings_) {
    timer_.start();
  }
  alpha_ = alpha;
  alpha_min_ = alpha_min;
  beta1_ = beta1;
  beta2_ = beta2;
  beta3_ = beta3;
  gamma_ = gamma;
  th_ = th;
  th_stop_ = settings.th_stop_;
  rel_th_stop_ = settings.rel_th_stop_;
  max_iter_ = settings.max_iter_;
  stop_ = std::numeric_limits<T>::max();
  lam_ = lam0;
  if (settings.statistics_) {
    stats_.reset();
  }
  prob.Del_->computeChol(1e-9);
  prob.Del_->evaluateDel();
  computeGinv(prob.Del_->G_);
  computeC(prob.Del_->G_, prob.g_, lam_);
  Vector3s lam_star, lam_v0;
  for (int i = 0; i < max_iter_; i++) {
    lam_pred_ = lam_;
    for (int j = 0; j < nc_; j++) {
      computeLamV0(Ginv_.template middleCols<3>(3 * j), c_.col(j), lam_v0);
      T muj = prob.contact_constraints_[CAST_UL(j)].mu_;
      if (c_.col(j)(2) > 0) { // opening contact
        lam_.template segment<3>(3 * j) *= 1 - alpha_;
      } else if (lam_v0(2) * muj >=
                 lam_v0.template head<2>().norm()) { // sticking contact
        lam_.template segment<3>(3 * j) *= 1 - alpha_;
        lam_.template segment<3>(3 * j) += alpha_ * lam_v0;
      } else { // sliding contact
        lam_.template segment<3>(3 * j) *= 1 - alpha_;
        bisectionStep(prob.Del_->G_.template block<3, 3>(3 * j, 3 * j),
                      Ginv_.template middleCols<3>(3 * j), c_.col(j), muj,
                      lam_v0, lam_star, max_iter_, th_, beta1_, beta2_, beta3_);
        lam_.template segment<3>(3 * j) += alpha_ * lam_star;
      }
      updateC(j, prob.Del_->G_, prob.g_, lam_.template segment<3>(3 * j));
    }
    stop_ = stoppingCriteria(prob, lam_, v_);
    rel_stop_ = relativeStoppingCriteria(lam_, lam_pred_);
    if (settings.statistics_) {
      computeStatistics(prob, lam_, v_);
      addStatistics(stop_, rel_stop_, comp_, sig_comp_, ncp_comp_, prim_feas_,
                    dual_feas_);
    }
    if (stop_ < th_stop_ || rel_stop_ < rel_th_stop_) {
      n_iter_ = i + 1;
      CONTACTBENCH_NOMALLOC_END;
      if (settings.timings_) {
        timer_.stop();
      }
      return true;
    }
    alpha_ = (1. - gamma_) * alpha_min_ + gamma_ * alpha_;
  }
  n_iter_ = max_iter_;
  CONTACTBENCH_NOMALLOC_END;
  if (settings.timings_) {
    timer_.stop();
  }
  return false;
}

template <typename T>
bool RaisimSolver<T>::_solve(ContactProblem<T, IceCreamCone> &prob,
                             const Ref<const VectorXs> &lam0,
                             ContactSolverSettings<T> &settings) {
  return solve(prob, lam0, settings);
}

template <typename T>
const Ref<const Eigen::Matrix<T, Eigen::Dynamic, 1>>
RaisimSolver<T>::getDualSolution() const {
  return v_;
}

template <typename T> void RaisimSolver<T>::resetStats() { stats_.reset(); }

//
//
// RaisimCorrectedSolver
//
//


template <typename T>
template <typename VecIn, typename VecIn2>
Eigen::Matrix<T, 3, 1> RaisimCorrectedSolver<T>::computeCorrectedC(
    const MatrixBase<VecIn> &ci, const MatrixBase<VecIn2> &vi, const T mu) {
  Vector3s ci_cor = ci;
  double norm_vt = vi.template head<2>().norm();
  ci_cor(2) += mu * norm_vt;
  return ci_cor;
}

template <typename T>
template <typename MatIn, typename VecIn, typename VecIn2, typename VecOut>
void RaisimCorrectedSolver<T>::computeCorrectedLamV0(
    const MatrixBase<MatIn> &Ginvi, const MatrixBase<VecIn> &ci,
    const MatrixBase<VecIn2> &vi, const T mu,
    const MatrixBase<VecOut> &lam_out) {
  Vector3s ci_cor = computeCorrectedC(ci, vi, mu);
  const_cast<MatrixBase<VecOut> &>(lam_out).noalias() = -Ginvi * ci_cor;
}

template <typename T>
template <typename MatIn, typename MatIn2, typename VecIn, typename VecIn2>
void RaisimCorrectedSolver<T>::bisectionStepCorrected(
    const MatrixBase<MatIn> &G, const MatrixBase<MatIn2> &Ginv,
    const MatrixBase<VecIn> &c, const MatrixBase<VecIn2> &v, const T mu,
    const Ref<const Vector3s> &lam_v0_cor, Ref<Vector3s> lam_out, int max_iter,
    T th, T beta1, T beta2, T beta3) {
  CONTACTBENCH_UNUSED(Ginv);
  Vector3s c_cor = computeCorrectedC(c, v, mu);
  T theta = computeTheta(lam_v0_cor);
  T r = computeR(G, c, mu, theta);
  T lamZ = computeLamZ(mu, r);
  computeLam(r, theta, lamZ, lam_out);
  T d0 = computeGradTheta(G, c_cor, mu, lam_out);
  T dtheta;
  if (d0 >= 0) {
    dtheta = -beta1;
  } else {
    dtheta = beta1;
  }
  T theta_pred;
  Vector3s lam_pred, gradH2;
  n_iter_mod_ = max_iter;

  for (int i = 0; i < max_iter; i++) { // initial stepping
    theta_pred = theta;
    lam_pred = lam_out;
    theta += dtheta;
    r = computeR(G, c, mu, theta);
    lamZ = computeLamZ(mu, r);
    computeLam(r, theta, lamZ, lam_out);
    computeH2Grad(mu, lam_out, gradH2);
    if (gradH2.dot(lam_v0_cor - lam_out) < 0 || r < 0) {
      // if (gradH2.dot(lam_out - lam_v0) < 0 || r < 0) {
      dtheta *= beta2;
      theta = theta_pred;
      lam_out = lam_pred; // TODO: should check this
    } else {
      T grad = computeGradTheta(G, c_cor, lam_out, gradH2);
      if (grad * d0 > 0) {
        dtheta *= beta3;
      } else {
        n_iter_mod_ = i + 1;
        break;
      }
    }
  } 
  T theta_bis, r_bis, lamZ_bis, grad_bis;
  Vector3s lam_bis;
  n_iter_bis_ = max_iter;
  for (int i = 0; i < max_iter; i++) { // bisection
    theta_bis = .5 * (theta + theta_pred);
    r_bis = computeR(G, c, mu, theta_bis);
    lamZ_bis = computeLamZ(mu, r_bis);
    computeLam(r_bis, theta_bis, lamZ_bis, lam_bis);
    grad_bis = computeGradTheta(G, c_cor, mu, lam_bis);
    if (grad_bis * d0 > 0) {
      theta_pred = theta_bis;
      lam_pred = lam_bis;
    } else {
      theta = theta_bis;
      lam_out = lam_bis;
    }
    if ((lam_out - lam_pred).template lpNorm<Infinity>() < th) {
      n_iter_bis_ = i + 1;
      break;
    }
  }
}

template <typename T>
bool RaisimCorrectedSolver<T>::solve(
    const ContactProblem<T, IceCreamCone> &prob,
    const Ref<const VectorXs> &lam0, ContactSolverSettings<T> &settings,
    T eps_reg, T alpha, T alpha_min, T beta1, T beta2, T beta3, T gamma, T th) {
  CONTACTBENCH_UNUSED(eps_reg);

  CONTACTBENCH_NOMALLOC_BEGIN;
  assert(lam0.size() == 3 * nc_);
  if (settings.timings_) {
    timer_.start();
  }
  alpha_ = alpha;
  alpha_min_ = alpha_min;
  beta1_ = beta1;
  beta2_ = beta2;
  beta3_ = beta3;
  gamma_ = gamma;
  th_ = th;
  th_stop_ = settings.th_stop_;
  rel_th_stop_ = settings.rel_th_stop_;
  max_iter_ = settings.max_iter_;
  stop_ = std::numeric_limits<double>::max();
  lam_ = lam0;
  // lam_pred_ = lam0;
  if (settings.statistics_) {
    stats_.reset();
  }
  prob.Del_->computeChol(1e-9);
  prob.Del_->evaluateDel();
  computeGinv(prob.Del_->G_);
  computeC(prob.Del_->G_, prob.g_, lam_);
  Vector3s lam_star, lam_v0, lam_v0_cor;
  for (int i = 0; i < max_iter_; i++) {
    lam_pred_ = lam_;
    for (int j = 0; j < nc_; j++) {
      computeLamV0(Ginv_.template middleCols<3>(3 * j), c_.col(j), lam_v0);
      double muj = prob.contact_constraints_[CAST_UL(j)].mu_;
      if (c_.col(j)(2) > 0) { // opening contact
        lam_.template segment<3>(3 * j) *= 1 - alpha_;
      } else if (lam_v0(2) * muj >=
                 lam_v0.template head<2>().norm()) { // sticking contact
        lam_.template segment<3>(3 * j) *= 1 - alpha_;
        lam_.template segment<3>(3 * j) += alpha_ * lam_v0;
      } else { // sliding contact
        computeCorrectedLamV0(Ginv_.template middleCols<3>(3 * j), c_.col(j),
                              v_.template segment<3>(3 * j), muj, lam_v0_cor);
        bisectionStepCorrected(prob.Del_->G_.template block<3, 3>(3 * j, 3 * j),
                               Ginv_.template middleCols<3>(3 * j), c_.col(j),
                               v_.template segment<3>(3 * j), muj, lam_v0_cor,
                               lam_star, max_iter_, th_, beta1_, beta2_,
                               beta3_);
        lam_.template segment<3>(3 * j) *= 1 - alpha_;
        lam_.template segment<3>(3 * j) += alpha_ * lam_star;
      }
      updateC(j, prob.Del_->G_, prob.g_, lam_.template segment<3>(3 * j));
    }
    stop_ = stoppingCriteria(prob, lam_, v_);
    rel_stop_ = relativeStoppingCriteria(lam_, lam_pred_);
    if (settings.statistics_) {
      computeStatistics(prob, lam_, v_);
      addStatistics(stop_, rel_stop_, comp_, sig_comp_, ncp_comp_, prim_feas_,
                    dual_feas_);
    }
    if (stop_ < th_stop_ || rel_stop_ < rel_th_stop_) {
      n_iter_ = i + 1;
      CONTACTBENCH_NOMALLOC_END;
      if (settings.timings_) {
        timer_.stop();
      }
      return true;
    }
    alpha_ = (1. - gamma_) * alpha_min_ + gamma_ * alpha_;
  }
  n_iter_ = max_iter_;
  CONTACTBENCH_NOMALLOC_END;
  if (settings.timings_) {
    timer_.stop();
  }
  return false;
}

template <typename T>
bool RaisimCorrectedSolver<T>::_solve(ContactProblem<T, IceCreamCone> &prob,
                                      const Ref<const VectorXs> &lam0,
                                      ContactSolverSettings<T> &settings) {
  return solve(prob, lam0, settings);
}

//
// Statistics
//

template <typename T> void Statistics<T>::addStop(T stop) {
  stop_.push_back(stop);
}

template <typename T> void Statistics<T>::addRelStop(T rel_stop) {
  rel_stop_.push_back(rel_stop);
}

template <typename T> void Statistics<T>::addComp(T comp) {
  comp_.push_back(comp);
}

template <typename T> void Statistics<T>::addPrimFeas(T prim_feas) {
  prim_feas_.push_back(prim_feas);
}

template <typename T> void Statistics<T>::addDualFeas(T dual_feas) {
  dual_feas_.push_back(dual_feas);
}

template <typename T> void Statistics<T>::addSigComp(T sig_comp) {
  sig_comp_.push_back(sig_comp);
}

template <typename T> void Statistics<T>::addNcpComp(T ncp_comp) {
  ncp_comp_.push_back(ncp_comp);
}

template <typename T> void Statistics<T>::reset() {
  stop_.clear();
  rel_stop_.clear();
  comp_.clear();
  prim_feas_.clear();
  dual_feas_.clear();
  sig_comp_.clear();
  ncp_comp_.clear();
}

} // namespace contactbench
