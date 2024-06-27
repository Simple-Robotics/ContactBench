#pragma once
#include "contactbench/contact-problem.hpp"
#include <cmath>

namespace contactbench {
using Eigen::Matrix;
using Eigen::MatrixBase;
using Eigen::MatrixXd;
using Eigen::VectorXd;

template <typename T, template <typename> class ConstraintTpl>
ContactProblem<T, ConstraintTpl>::ContactProblem(const Ref<const MatrixXs> &G,
                                                 const Ref<const VectorXs> &g,
                                                 const std::vector<T> &mus,
                                                 const T comp)
    : Del_(std::make_shared<DelassusDense<T>>(G)), g_(g), nc_(g_.size() / 3),
      v_(3 * nc_), v_comp_(3 * nc_) {
  assert(nc_ == G.cols() / 3);
  assert(nc_ == (isize)mus.size());
  R_comp_ = comp * VectorXs::Zero(3 * nc_);
  for (auto &mu : mus) {
    contact_constraints_.emplace_back(mu);
  }
  v_.setZero();
}

template <typename T, template <typename> class ConstraintTpl>
ContactProblem<T, ConstraintTpl>::ContactProblem(
    const Ref<const MatrixXs> &G, const Ref<const VectorXs> &g,
    const std::vector<T> &mus, const Ref<const VectorXs> &comp)
    : Del_(std::make_shared<DelassusDense<T>>(G)), g_(g), nc_(g_.size() / 3),
      v_(3 * nc_), v_comp_(3 * nc_) {
  assert(nc_ == G.cols() / 3);
  assert(nc_ == (isize)mus.size());
  assert(3 * nc_ == (isize)comp.size());
  R_comp_ = comp;
  for (auto &mu : mus) {
    contact_constraints_.emplace_back(mu);
  }
  v_.setZero();
}

template <typename T, template <typename> class ConstraintTpl>
ContactProblem<T, ConstraintTpl>::ContactProblem(
    std::shared_ptr<DelassusBase<T>> Del, const Ref<const VectorXs> &g,
    const std::vector<T> &mus, const T comp)
    : Del_(Del), g_(g), nc_(g_.size() / 3), v_(3 * nc_), v_comp_(3 * nc_) {
  assert(nc_ == Del_->nc_);
  assert(nc_ == (isize)mus.size());
  R_comp_ = comp * VectorXs::Zero(3 * nc_);
  for (auto &mu : mus) {
    contact_constraints_.emplace_back(mu);
  }
  v_.setZero();
}

template <typename T, template <typename> class ConstraintTpl>
ContactProblem<T, ConstraintTpl>::ContactProblem(
    std::shared_ptr<DelassusBase<T>> Del, const Ref<const VectorXs> &g,
    const std::vector<T> &mus, const Ref<const VectorXs> &comp)
    : Del_(Del), g_(g), nc_(g_.size() / 3), v_(3 * nc_), v_comp_(3 * nc_) {
  assert(nc_ == Del_->nc_);
  assert(nc_ == (isize)mus.size());
  assert(3 * nc_ == (isize)comp.size());
  R_comp_ = comp;
  for (auto &mu : mus) {
    contact_constraints_.emplace_back(mu);
  }
  v_.setZero();
}

template <typename T, template <typename> class ConstraintTpl>
ContactProblem<T, ConstraintTpl>::ContactProblem(
    const Ref<const MatrixXs> &M, const Ref<const MatrixXs> &J,
    const Ref<const VectorXs> &dqf, const Ref<const VectorXs> &vstar,
    const std::vector<T> &mus, const T comp)
    : M_(M), J_(J), dqf_(dqf), vstar_(vstar), nc_(vstar_.size() / 3),
      v_(3 * nc_), v_comp_(3 * nc_) {
  assert(nc_ == J.rows() / 3);
  assert(nc_ == (isize)mus.size());
  R_comp_ = comp * VectorXs::Zero(3 * nc_);
  for (auto &mu : mus) {
    contact_constraints_.emplace_back(mu);
  }
  v_.setZero();
}

template <typename T, template <typename> class ConstraintTpl>
ContactProblem<T, ConstraintTpl>::ContactProblem(
    const Ref<const MatrixXs> &M, const Ref<const MatrixXs> &J,
    const Ref<const VectorXs> &dqf, const Ref<const VectorXs> &vstar,
    const std::vector<T> &mus, const Ref<const VectorXs> &comp)
    : M_(M), J_(J), dqf_(dqf), vstar_(vstar), nc_(vstar_.size() / 3),
      v_(3 * nc_), v_comp_(3 * nc_) {
  assert(nc_ == J.rows() / 3);
  assert(nc_ == (isize)mus.size());
  assert(3 * nc_ == (isize)comp.size());
  R_comp_ = comp;
  for (auto &mu : mus) {
    contact_constraints_.emplace_back(mu);
  }
  v_.setZero();
}

template <typename T, template <typename> class ConstraintTpl>
ContactProblem<T, ConstraintTpl>::ContactProblem(
    const Ref<const MatrixXs> &G, const Ref<const VectorXs> &g,
    const Ref<const MatrixXs> &M, const Ref<const MatrixXs> &J,
    const Ref<const VectorXs> &dqf, const Ref<const VectorXs> &vstar,
    const std::vector<T> &mus, const T comp)
    : Del_(std::make_shared<DelassusDense<T>>(G)), g_(g), M_(M), J_(J),
      dqf_(dqf), vstar_(vstar), nc_(g_.size() / 3), v_(3 * nc_),
      v_comp_(3 * nc_) {
  assert(nc_ == G.cols() / 3);
  assert(nc_ == (isize)mus.size());
  R_comp_ = comp * VectorXs::Zero(3 * nc_);
  for (auto &mu : mus) {
    contact_constraints_.emplace_back(mu);
  }
  v_.setZero();
}

template <typename T, template <typename> class ConstraintTpl>
ContactProblem<T, ConstraintTpl>::ContactProblem(
    const Ref<const MatrixXs> &G, const Ref<const VectorXs> &g,
    const Ref<const MatrixXs> &M, const Ref<const MatrixXs> &J,
    const Ref<const VectorXs> &dqf, const Ref<const VectorXs> &vstar,
    const std::vector<T> &mus, const Ref<const VectorXs> &comp)
    : Del_(std::make_shared<DelassusDense<T>>(G)), g_(g), M_(M), J_(J),
      dqf_(dqf), vstar_(vstar), nc_(g_.size() / 3), v_(3 * nc_),
      v_comp_(3 * nc_) {
  assert(nc_ == G.cols() / 3);
  assert(nc_ == (isize)mus.size());
  assert(3 * nc_ == (isize)comp.size());
  R_comp_ = comp;
  for (auto &mu : mus) {
    contact_constraints_.emplace_back(mu);
  }
  v_.setZero();
}

template <typename T, template <typename> class ConstraintTpl>
ContactProblem<T, ConstraintTpl>::ContactProblem(
    std::shared_ptr<DelassusBase<T>> Del, const Ref<const VectorXs> &g,
    const Ref<const MatrixXs> &M, const Ref<const MatrixXs> &J,
    const Ref<const VectorXs> &dqf, const Ref<const VectorXs> &vstar,
    const std::vector<T> &mus, const T comp)
    : Del_(Del), g_(g), M_(M), J_(J), dqf_(dqf), vstar_(vstar),
      nc_(g_.size() / 3), v_(3 * nc_), v_comp_(3 * nc_) {
  assert(nc_ == Del_->nc_);
  assert(nc_ == (isize)mus.size());
  R_comp_ = comp * VectorXs::Zero(3 * nc_);
  for (auto &mu : mus) {
    contact_constraints_.emplace_back(mu);
  }
  v_.setZero();
}

template <typename T, template <typename> class ConstraintTpl>
ContactProblem<T, ConstraintTpl>::ContactProblem(
    std::shared_ptr<DelassusBase<T>> Del, const Ref<const VectorXs> &g,
    const Ref<const MatrixXs> &M, const Ref<const MatrixXs> &J,
    const Ref<const VectorXs> &dqf, const Ref<const VectorXs> &vstar,
    const std::vector<T> &mus, const Ref<const VectorXs> &comp)
    : Del_(Del), g_(g), M_(M), J_(J), dqf_(dqf), vstar_(vstar),
      nc_(g_.size() / 3), v_(3 * nc_), v_comp_(3 * nc_) {
  assert(nc_ == Del_->nc_);
  assert(nc_ == (isize)mus.size());
  assert(3 * nc_ == (isize)comp.size());
  R_comp_ = comp;
  for (auto &mu : mus) {
    contact_constraints_.emplace_back(mu);
  }
  v_.setZero();
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::setCompliance(
    const Ref<const VectorXs> &comp) {
  assert(3 * nc_ == (isize)comp.size());
  R_comp_ = comp;
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::setCompliance(const T comp) {
  setCompliance(comp * VectorXs::Zero(3 * nc_));
}

template <typename T, template <typename> class ConstraintTpl>
template <typename VecIn, typename VecOut>
void ContactProblem<T, ConstraintTpl>::project(
    const std::vector<T> &mus, const MatrixBase<VecIn> &lam,
    const MatrixBase<VecOut> &lam_out_) {
  CONTACTBENCH_NOMALLOC_BEGIN;
  int nc = (int)mus.size();
  auto &lam_out = lam_out_.const_cast_derived();
  for (isize i = 0; i < nc; i++) {
    ConstraintType::project(mus[(usize)i], lam.template segment<3>(3 * i),
                            lam_out.template segment<3>(3 * i));
  }
  CONTACTBENCH_NOMALLOC_END;
}

template <typename T, template <typename> class ConstraintTpl>
template <typename VecIn, typename VecOut>
void ContactProblem<T, ConstraintTpl>::project(
    const MatrixBase<VecIn> &lam, const MatrixBase<VecOut> &lam_out_) const {
  CONTACTBENCH_NOMALLOC_BEGIN;
  auto &lam_out = lam_out_.const_cast_derived();
  for (isize i = 0; i < nc_; i++) {
    contact_constraints_[(usize)i].project(lam.template segment<3>(3 * i),
                                           lam_out.template segment<3>(3 * i));
  }
  CONTACTBENCH_NOMALLOC_END;
}

template <typename T, template <typename> class ConstraintTpl>
template <typename VecIn, typename VecOut>
void ContactProblem<T, ConstraintTpl>::projectDual(
    const MatrixBase<VecIn> &v, const MatrixBase<VecOut> &v_out_) const {
  CONTACTBENCH_NOMALLOC_BEGIN;
  auto &v_out = v_out_.const_cast_derived();
  for (isize i = 0; i < nc_; i++) {
    contact_constraints_[(usize)i].projectDual(
        v.template segment<3>(3 * i), v_out.template segment<3>(3 * i));
  }
  CONTACTBENCH_NOMALLOC_END;
}

template <typename T, template <typename> class ConstraintTpl>
bool ContactProblem<T, ConstraintTpl>::isInside(const Ref<const VectorXs> &lam,
                                                T thresh) const {
  for (isize i = 0; i < nc_; i++) {
    bool isIni = contact_constraints_[(usize)i].isInside(
        lam.template segment<3>(3 * i), thresh);
    if (!isIni) {
      return false;
    }
  }
  return true;
}

template <typename T, template <typename> class ConstraintTpl>
bool ContactProblem<T, ConstraintTpl>::isInsideDual(
    const Ref<const VectorXs> &v, T thresh) const {
  for (isize i = 0; i < nc_; i++) {
    bool isIni = contact_constraints_[(usize)i].isInsideDual(
        v.template segment<3>(3 * i), thresh);
    if (!isIni) {
      return false;
    }
  }
  return true;
}

template <typename T, template <typename> class ConstraintTpl>
template <typename VecIn, typename VecOut>
void ContactProblem<T, ConstraintTpl>::computeVelocity(
    const MatrixBase<VecIn> &lam, const MatrixBase<VecOut> &v_out_) const {
  CONTACTBENCH_NOMALLOC_BEGIN;
  auto &v_out = v_out_.const_cast_derived();
  Del_->applyOnTheRight(lam, v_out);
  v_out += g_;
  v_out += (R_comp_.array() * lam.array()).matrix();
  CONTACTBENCH_NOMALLOC_END;
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeDeSaxceCorrection(
    const Ref<const VectorXs> &v, VectorRef v_out) const {
  for (isize i = 0; i < nc_; i++) {
    contact_constraints_[(usize)i].computeDeSaxceCorrection(
        v.template segment<3>(3 * i), v_out.template segment<3>(3 * i));
  }
}

template <typename T, template <typename> class ConstraintTpl>
T ContactProblem<T, ConstraintTpl>::computeComplementarity(
    const Ref<const VectorXs> &lam, const Ref<const VectorXs> &v) {
  isize dim = (isize)lam.size();
  assert(dim == (isize)v.size());
  T max_comp = 0.;
  for (isize i = 0; i < dim; i++) {
    T comp_i = std::abs(lam(i) * v(i));
    if (comp_i > max_comp) {
      max_comp = comp_i;
    }
  }
  return max_comp;
}

template <typename T, template <typename> class ConstraintTpl>
T ContactProblem<T, ConstraintTpl>::computeContactComplementarity(
    const Ref<const VectorXs> &lam, const Ref<const VectorXs> &v) const {
  T max_comp = 0.;
  for (isize i = 0; i < nc_; i++) {
    T comp_i = computePerContactContactComplementarity(
        (usize)i, lam.template segment<3>(3 * i), v.template segment<3>(3 * i));
    if (comp_i > max_comp) {
      max_comp = comp_i;
    }
  }
  return max_comp;
}

template <typename T, template <typename> class ConstraintTpl>
T ContactProblem<T, ConstraintTpl>::computeContactComplementarity(
    const Ref<const VectorXs> &lam) {
  // Del_->applyOnTheRight(lam, v_);
  // v_ += g_;
  // v_ += R_comp_.array() * lam.array();
  computeVelocity(lam, v_comp_);
  T max_comp = computeContactComplementarity(lam, v_comp_);
  return max_comp;
}

template <typename T, template <typename> class ConstraintTpl>
T ContactProblem<T, ConstraintTpl>::computePerContactContactComplementarity(
    usize i, const Vector3s &lam, const Vector3s &v) const {
  return contact_constraints_[i].computeContactComplementarity(lam, v);
}

template <typename T, template <typename> class ConstraintTpl>
T ContactProblem<T, ConstraintTpl>::computeConicComplementarity(
    const Ref<const VectorXs> &lam, const Ref<const VectorXs> &v) const {
  T max_comp = 0.;
  for (isize i = 0; i < nc_; i++) {
    T comp_i = computePerContactConicComplementarity(
        (usize)i, lam.template segment<3>(3 * i), v.template segment<3>(3 * i));
    if (comp_i > max_comp) {
      max_comp = comp_i;
    }
  }
  return max_comp;
}

template <typename T, template <typename> class ConstraintTpl>
T ContactProblem<T, ConstraintTpl>::computeConicComplementarity(
    const Ref<const VectorXs> &lam) {
  // Del_->applyOnTheRight(lam, v_);
  // v_ += g_;
  // v_ += (R_comp_.array() * lam.array()).matrix();
  computeVelocity(lam, v_comp_);
  T max_comp = computeConicComplementarity(lam, v_comp_);
  return max_comp;
}

template <typename T, template <typename> class ConstraintTpl>
T ContactProblem<T, ConstraintTpl>::computePerContactConicComplementarity(
    usize i, const Vector3s &lam, const Vector3s &v) const {
  return contact_constraints_[(usize)i].computeConicComplementarity(lam, v);
}

template <typename T, template <typename> class ConstraintTpl>
T ContactProblem<T, ConstraintTpl>::computeSignoriniComplementarity(
    const Ref<const VectorXs> &lam, const Ref<const VectorXs> &v) const {
  T max_comp = 0.;
  for (isize i = 0; i < nc_; i++) {
    T comp_i = computePerContactSignoriniComplementarity(
        (usize)i, lam.template segment<3>(3 * i), v.template segment<3>(3 * i));
    if (comp_i > max_comp) {
      max_comp = comp_i;
    }
  }
  return max_comp;
}

template <typename T, template <typename> class ConstraintTpl>
T ContactProblem<T, ConstraintTpl>::computeSignoriniComplementarity(
    const Ref<const VectorXs> &lam) {
  // Del_->applyOnTheRight(lam, v_);
  // v_ += g_;
  // v_ += R_comp_.array() * lam.array();
  computeVelocity(lam, v_comp_);
  T max_comp = computeSignoriniComplementarity(lam, v_comp_);
  return max_comp;
}

template <typename T, template <typename> class ConstraintTpl>
T ContactProblem<T, ConstraintTpl>::computePerContactSignoriniComplementarity(
    usize i, const Vector3s &lam, const Vector3s &v) const {
  return contact_constraints_[i].computeSignoriniComplementarity(lam, v);
}

template <typename T, template <typename> class ConstraintTpl>
T ContactProblem<T, ConstraintTpl>::computeLinearComplementarity(
    const Ref<const VectorXs> &lam, const Ref<const VectorXs> &v) {
  // computeLCP, computeTangentLCP, computeInscribedLCP should be called
  // beforehand
  computeLCPSolution(lam, v, lam_lcp_);
  v_lcp_ = A_ * lam_lcp_ + b_;
  T max_comp = ContactProblem<T, ConstraintTpl>::computeComplementarity(
      lam_lcp_, v_lcp_);
  return max_comp;
}

template <typename T, template <typename> class ConstraintTpl>
T ContactProblem<T, ConstraintTpl>::computeLinearComplementarity(
    const Ref<const VectorXs> &lam) {
  // computeLCP, computeTangentLCP, computeInscribedLCP should be called
  // beforehand
  computeLCPSolution(lam, lam_lcp_);
  v_lcp_ = A_ * lam_lcp_ + b_;
  T max_comp = ContactProblem<T, ConstraintTpl>::computeComplementarity(
      lam_lcp_, v_lcp_);
  return max_comp;
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::setLCP() {
  A_.resize(6 * nc_, 6 * nc_);
  b_.resize(6 * nc_);
  lam_lcp_.resize(6 * nc_);
  v_lcp_.resize(6 * nc_);
  UD_.resize(3 * nc_, 6 * nc_);
  UD_.setZero();
  GUD_.resize(3 * nc_, 6 * nc_);
  G_reg_.resize(3 * nc_, 3 * nc_);
  E_.resize(nc_, 6 * nc_);
  E_.setZero();
  F_.resize(4 * nc_, 6 * nc_);
  F_.setZero();
  H_.resize(nc_, 6 * nc_);
  H_.setZero();
  for (isize i = 0; i < nc_; i++) {
    E_(i, 6 * i + 5) = 1.;
    F_.template block<4, 4>(4 * i, 6 * i) = Matrix4s::Identity();
    H_(i, 6 * i + 4) = 1.;
  }
  dL_dDel_t_.resize(3 * nc_, 3 * nc_);
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeAb_(
    const Ref<const VectorXs> &R_reg) {
  A_.setZero();
  b_.setZero();
  G_reg_ = Del_->G_;
  G_reg_.diagonal() += R_comp_;
  G_reg_.diagonal() += R_reg;
  GUD_.noalias() = G_reg_ * UD_;
  for (isize i = 0; i < nc_; i++) {
    b_.template segment<4>(6 * i) =
        linear_contact_constraints_[CAST_UL(i)].UD_.transpose() *
        g_.template segment<2>(3 * i);
    b_(6 * i + 4) = g_(3 * i + 2);
    A_.template middleRows<4>(6 * i).noalias() =
        linear_contact_constraints_[CAST_UL(i)].UD_.transpose() *
        GUD_.template middleRows<2>(3 * i);
    A_.template middleRows<4>(6 * i).noalias() +=
        linear_contact_constraints_[CAST_UL(i)].e_ * E_.row(i);
    A_.row(6 * i + 4) = GUD_.row(3 * i + 2);
    A_.row(6 * i + 5) =
        std::sqrt(2) * linear_contact_constraints_[CAST_UL(i)].mu_ * H_.row(i);
    A_.row(6 * i + 5).noalias() +=
        -linear_contact_constraints_[CAST_UL(i)].e_.transpose() *
        F_.template middleRows<4>(4 * i);
  }
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeAb_(const T eps_reg) {
  computeAb_(VectorXs::Constant(3 * nc_, eps_reg));
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeLCP(
    const Ref<const VectorXs> &R_reg) {
  for (isize i = 0; i < nc_; i++) {
    linear_contact_constraints_.emplace_back(
        contact_constraints_[CAST_UL(i)].mu_);
    UD_.template block<3, 6>(3 * i, 6 * i) =
        linear_contact_constraints_[CAST_UL(i)].UD_lcp_;
  }
  computeAb_(R_reg);
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeLCP(const T eps_reg) {
  computeLCP(VectorXs::Constant(3 * nc_, eps_reg));
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeTangentLCP(
    const Ref<const VectorXs> &lam, const Ref<const VectorXs> &R_reg) {
  linear_contact_constraints_.clear();
  for (isize i = 0; i < nc_; i++) {
    linear_contact_constraints_.emplace_back(
        contact_constraints_[CAST_UL(i)].mu_, lam.template segment<2>(3 * i));
    UD_.template block<3, 6>(3 * i, 6 * i) =
        linear_contact_constraints_[CAST_UL(i)].UD_lcp_;
  }
  computeAb_(R_reg);
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeTangentLCP(
    const Ref<const VectorXs> &lam, const T eps_reg) {
  computeTangentLCP(lam, VectorXs::Constant(3 * nc_, eps_reg));
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeInscribedLCP(
    const Ref<const VectorXs> &lam, const Ref<const VectorXs> &R_reg) {
  linear_contact_constraints_.clear();
  for (isize i = 0; i < nc_; i++) {
    linear_contact_constraints_.emplace_back(
        contact_constraints_[CAST_UL(i)].mu_ / std::sqrt(2.),
        PI4ROT.transpose() * lam.template segment<2>(3 * i));
    UD_.template block<3, 6>(3 * i, 6 * i) =
        linear_contact_constraints_[CAST_UL(i)].UD_lcp_;
  }
  computeAb_(R_reg);
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeInscribedLCP(
    const Ref<const VectorXs> &lam, const T eps_reg) {
  computeInscribedLCP(lam, VectorXs::Constant(3 * nc_, eps_reg));
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::setLCCP() {
  C_in_.resize(4 * nc_, 3 * nc_);
  v_tmp_.resize(3 * nc_);
  n_in_tan_ = 4 * nc_;
  n_eq_tan_ = 0;
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeC_() {
  CONTACTBENCH_NOMALLOC_BEGIN;
  C_in_.setZero();
  for (isize i = 0; i < nc_; i++) {
    C_in_.row(4 * i).template segment<2>(3 * i) =
        linear_contact_constraints_[CAST_UL(i)].Uxy_.col(0);
    C_in_(4 * i, 3 * i + 2) = -linear_contact_constraints_[CAST_UL(i)].mu_;
    C_in_.row(4 * i + 1).template segment<2>(3 * i) =
        -linear_contact_constraints_[CAST_UL(i)].Uxy_.col(0);
    C_in_(4 * i + 1, 3 * i + 2) = -linear_contact_constraints_[CAST_UL(i)].mu_;
    C_in_.row(4 * i + 2).template segment<2>(3 * i) =
        linear_contact_constraints_[CAST_UL(i)].Uxy_.col(1);
    C_in_(4 * i + 2, 3 * i + 2) = -linear_contact_constraints_[CAST_UL(i)].mu_;
    C_in_.row(4 * i + 3).template segment<2>(3 * i) =
        -linear_contact_constraints_[CAST_UL(i)].Uxy_.col(1);
    C_in_(4 * i + 3, 3 * i + 2) = -linear_contact_constraints_[CAST_UL(i)].mu_;
  }
  CONTACTBENCH_NOMALLOC_END;
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeLCCP() {
  CONTACTBENCH_NOMALLOC_BEGIN;
  computeC_();
  CONTACTBENCH_NOMALLOC_END;
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeTangentLCCP(
    const Ref<const VectorXs> &lam) {
  CONTACTBENCH_NOMALLOC_BEGIN;
  C_in_.setZero();
  linear_contact_constraints_.clear();
  for (isize i = 0; i < nc_; i++) {
    linear_contact_constraints_.emplace_back(
        contact_constraints_[CAST_UL(i)].mu_, lam.template segment<2>(3 * i));
  }
  computeC_();
  CONTACTBENCH_NOMALLOC_END;
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeInscribedLCCP(
    const Ref<const VectorXs> &lam) {
  CONTACTBENCH_NOMALLOC_BEGIN;
  C_in_.setZero();
  linear_contact_constraints_.clear();
  for (isize i = 0; i < nc_; i++) {
    linear_contact_constraints_.emplace_back(
        contact_constraints_[CAST_UL(i)].mu_ / std::sqrt(2.),
        PI4ROT.transpose() * lam.template segment<2>(3 * i));
  }
  computeC_();
  CONTACTBENCH_NOMALLOC_END;
}

template <typename T, template <typename> class ConstraintTpl>
template <typename VecIn, typename VecOut>
void ContactProblem<T, ConstraintTpl>::computeVelInTangentBasis(
    const MatrixBase<VecIn> &v, const MatrixBase<VecOut> &v_out_) const {
  CONTACTBENCH_NOMALLOC_BEGIN;
  auto &v_out = v_out_.const_cast_derived();
  for (isize i = 0; i < nc_; i++) {
    v_out(3 * i + 2) = v(3 * i + 2);
    linear_contact_constraints_[(usize)i].computeCoord(
        v.template segment<2>(3 * i), v_out.template segment<2>(3 * i));
  }
  CONTACTBENCH_NOMALLOC_END;
}

template <typename T, template <typename> class ConstraintTpl>
template <typename VecIn, typename VecOut>
void ContactProblem<T, ConstraintTpl>::computeVelInInscribedBasis(
    const MatrixBase<VecIn> &v, const MatrixBase<VecOut> &v_out_) const {
  CONTACTBENCH_NOMALLOC_BEGIN;
  auto &v_out = v_out_.const_cast_derived();
  for (isize i = 0; i < nc_; i++) {
    v_out(3 * i + 2) = v(3 * i + 2);
    linear_contact_constraints_[(usize)i].computeCoord(
        v.template segment<2>(3 * i), v_out.template segment<2>(3 * i));
  }
  CONTACTBENCH_NOMALLOC_END;
}

template <typename T, template <typename> class ConstraintTpl>
template <typename VecIn, typename VecIn2, typename VecOut, typename VecOut2>
void ContactProblem<T, ConstraintTpl>::computeDualSolutionOfTangent(
    const MatrixBase<VecIn> &lam, const MatrixBase<VecIn2> &v,
    const MatrixBase<VecOut> &y_out_, const MatrixBase<VecOut2> &z_out_,
    const double eps) const {
  CONTACTBENCH_UNUSED(y_out_);
  CONTACTBENCH_NOMALLOC_BEGIN;
  auto &z_out = z_out_.const_cast_derived();
  computeVelInTangentBasis(v, v_tmp_);
  for (isize i = 0; i < nc_; i++) {
    if (lam(3 * i + 2) < eps) {
      // TODO: how to init dual variable when contact is breaking
    } else {
      z_out(4 * i) = -v_tmp_(3 * i);
      z_out(4 * i + 1) = 0.;
      z_out(4 * i + 2) = 0.;
      z_out(4 * i + 3) = 0.;
    }
  }
  CONTACTBENCH_NOMALLOC_END;
}

template <typename T, template <typename> class ConstraintTpl>
template <typename VecIn, typename VecIn2, typename VecOut, typename VecOut2>
void ContactProblem<T, ConstraintTpl>::computeDualSolutionOfInscribed(
    const MatrixBase<VecIn> &lam, const MatrixBase<VecIn2> &v,
    const MatrixBase<VecOut> &y_out_, const MatrixBase<VecOut2> &z_out_,
    const double eps) const {
  CONTACTBENCH_UNUSED(y_out_);
  CONTACTBENCH_NOMALLOC_BEGIN;
  auto &z_out = z_out_.const_cast_derived();
  computeVelInInscribedBasis(v, v_tmp_);
  // std::cout << "v:    " << v << std::endl;
  // std::cout << "eps:    " << eps << std::endl;
  for (isize i = 0; i < nc_; i++) {
    if (lam(3 * i + 2) < eps) {
      // TODO: how to init dual variable when contact is breaking
    } else {
      z_out(4 * i) = -v_tmp_(3 * i);
      z_out(4 * i + 1) = 0.;
      z_out(4 * i + 2) = -v_tmp_(3 * i + 1);
      z_out(4 * i + 3) = 0.;
    }
  }
  CONTACTBENCH_NOMALLOC_END;
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeLCPSolution(
    const Ref<const VectorXs> &lam, const Ref<const VectorXs> &c,
    VectorRef lam_lcp) const {
  CONTACTBENCH_NOMALLOC_BEGIN;
  for (isize i = 0; i < nc_; i++) {
    // contact_constraints_[i].computeCoordinatesInD(
    linear_contact_constraints_[CAST_UL(i)].computeCoordinatesInD(
        lam.template segment<3>(3 * i), lam_lcp.template segment<4>(6 * i));
    lam_lcp(6 * i + 4) = lam(3 * i + 2);
    lam_lcp(6 * i + 5) = c.template segment<2>(3 * i).norm();
  }
  CONTACTBENCH_NOMALLOC_END;
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeLCPSolution(
    const Ref<const VectorXs> &lam, VectorRef lam_lcp) {
  CONTACTBENCH_NOMALLOC_BEGIN;
  // Del_->applyOnTheRight(lam, v_);
  // v_ += g_;
  // v_ += R_comp_.array() * lam.array();
  computeVelocity(lam, v_comp_);
  computeLCPSolution(lam, v_comp_, lam_lcp);
  CONTACTBENCH_NOMALLOC_END;
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeLCPdLdg(
    const Ref<const VectorXs> &dL_db, VectorRef dL_dg) {
  for (isize i = 0; i < nc_; i++) {
    dL_dg.template segment<2>(3 * i).noalias() =
        linear_contact_constraints_[CAST_UL(i)].UD_ *
        dL_db.template segment<4>(6 * i);
    dL_dg(3 * i + 2) = dL_db(6 * i + 4);
  }
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeLCPdLdDel(
    const Ref<const MatrixXs> &dL_dA, MatrixRef dL_dDel) {
  for (isize i = 0; i < nc_; i++) {
    for (isize k = 0; k < 4; k++) {
      for (isize l = 0; l < 6 * nc_; l++) {
        dL_dDel.template middleRows<2>(3 * i).noalias() +=
            dL_dA.template middleRows<4>(6 * i)(k, l) *
            linear_contact_constraints_[CAST_UL(i)].UD_.col(k) *
            (UD_.col(l).transpose());
      }
    }
    dL_dDel.row(3 * i + 2).noalias() = dL_dA.row(6 * i + 4) * UD_.transpose();
  }
  dL_dDel_t_ = dL_dDel.transpose();
  dL_dDel += dL_dDel_t_;
  dL_dDel /= 2;
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeLCPdLdmus(
    const Ref<const MatrixXs> &dL_dA, VectorRef dL_dmus) {
  for (isize i = 0; i < nc_; i++) {
    dL_dmus(i) = std::sqrt(2) * H_.row(i) * dL_dA.row(6 * i + 5).transpose();
  }
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeInscribedLCPdLdmus(
    const Ref<const MatrixXs> &dL_dA, VectorRef dL_dmus) {
  computeLCPdLdmus(dL_dA, dL_dmus);
  dL_dmus /= std::sqrt(2.);
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeTangentLCPdLdmus(
    const Ref<const MatrixXs> &dL_dA, VectorRef dL_dmus) {
  computeLCPdLdmus(dL_dA, dL_dmus);
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeLCCPdLdmus(
    const Ref<const MatrixXs> &dL_dC, VectorRef dL_dmus) {
  for (isize i = 0; i < nc_; i++) {
    dL_dmus(i) = 0;
    for (isize j = 0; j < 4; j++) {
      dL_dmus(i) += -dL_dC(4 * i + j, 3 * i + 2);
    }
  }
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeInscribedLCCPdLdmus(
    const Ref<const MatrixXs> &dL_dC, VectorRef dL_dmus) {
  computeLCCPdLdmus(dL_dC, dL_dmus);
  dL_dmus /= std::sqrt(2.);
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeTangentLCCPdLdmus(
    const Ref<const MatrixXs> &dL_dC, VectorRef dL_dmus) {
  computeLCCPdLdmus(dL_dC, dL_dmus);
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeLCPdbdtheta(
    const Ref<const MatrixXs> &dg_dtheta, MatrixRef db_dtheta) {
  for (isize i = 0; i < nc_; i++) {
    db_dtheta.template middleRows<4>(6 * i) =
        linear_contact_constraints_[CAST_UL(i)].UD_.transpose() *
        dg_dtheta.template middleRows<2>(3 * i);
    db_dtheta.row(6 * i + 4) = dg_dtheta.row(3 * i + 2);
  }
}

template <typename T, template <typename> class ConstraintTpl>
void ContactProblem<T, ConstraintTpl>::computeLCPdAdtheta(
    const Ref<const MatrixXs> &dmus_dtheta,
    const Ref<const MatrixXs> &dG_dtheta, const Ref<const VectorXs> &lam,
    MatrixRef mat_out) {
  CONTACTBENCH_UNUSED(dmus_dtheta);
  CONTACTBENCH_UNUSED(dG_dtheta);
  CONTACTBENCH_UNUSED(lam);
  CONTACTBENCH_UNUSED(mat_out);
  for (isize i = 0; i < nc_; i++) {
    // mat_out.template middleRows<4>(6 * i).noalias() =
    // linear_contact_constraints_[i].UD_.transpose() * dG_dtheta.template
    // middleRows<2>(3 * i) * (UD_ * lam); mat_out.row(6*i+4)= dG_dtheta(3 * i +
    // 2) * (UD_ * lam); mat_out.row(6*i+5) = std::sqrt(2) * dmus_dtheta(i) *
    // (H_.row(i)*lam);
  }
}

template <typename T, template <typename> class ConstraintTpl>
const Eigen::Ref<const Matrix<T, -1, -1>>
ContactProblem<T, ConstraintTpl>::getLinearInConstraintMatrix() const {
  return C_in_;
}

template <typename T, template <typename> class ConstraintTpl>
const Eigen::Ref<const Matrix<T, -1, -1>>
ContactProblem<T, ConstraintTpl>::getLinearEqConstraintMatrix() const {
  return C_eq_;
}

} // namespace contactbench
