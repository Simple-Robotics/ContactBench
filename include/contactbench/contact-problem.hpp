#ifndef CONTACT_BENCH_PROBLEM_H
#define CONTACT_BENCH_PROBLEM_H

#include <vector>
#include <Eigen/Core>
#include <contactbench/friction-constraint.hpp>
#include <contactbench/delassus-wrapper.hpp>
#include <contactbench/helpers.hpp>
namespace contactbench {

namespace {
using Eigen::Matrix;
using Eigen::MatrixBase;

} // namespace

using isize = Eigen::Index;
using usize = std::make_unsigned<isize>::type;

template <typename T, template <typename> class ConstraintTpl>
struct ContactProblem {
  using ConstraintType = ConstraintTpl<T>;
  static_assert(
      std::is_base_of<FrictionConstraint<T>, ConstraintType>::value,
      "ConstraintType should be inheritating from FrictionConstraint.");

public:
  CONTACTBENCH_EIGEN_TYPEDEFS(T);
  // physical quantities involved in the dual contact problem
  std::shared_ptr<DelassusBase<T>> Del_;
  VectorXs g_;
  // physical quantities involved in the primal contact problem
  MatrixXs M_, J_;
  VectorXs dqf_, vstar_, R_comp_;
  // friction constraints
  std::vector<ConstraintType> contact_constraints_;
  std::vector<PyramidCone<T>> linear_contact_constraints_;

  isize nc_, n_breaking_;

  // quantities necesary for the equivalent LCP
  MatrixXs A_, UD_;
  VectorXs b_;
  isize n_in_tan_, n_eq_tan_;

  ContactProblem(){};
  ContactProblem(const Ref<const MatrixXs> &G, const Ref<const VectorXs> &g,
                 const std::vector<T> &mus, const T comp = 0.);
  ContactProblem(const Ref<const MatrixXs> &G, const Ref<const VectorXs> &g,
                 const std::vector<T> &mus, const Ref<const VectorXs> &comp);
  ContactProblem(std::shared_ptr<DelassusBase<T>> Del,
                 const Ref<const VectorXs> &g, const std::vector<T> &mus,
                 const T comp = 0.);
  ContactProblem(std::shared_ptr<DelassusBase<T>> Del,
                 const Ref<const VectorXs> &g, const std::vector<T> &mus,
                 const Ref<const VectorXs> &comp);
  ContactProblem(const Ref<const MatrixXs> &M, const Ref<const MatrixXs> &J,
                 const Ref<const VectorXs> &dqf,
                 const Ref<const VectorXs> &vstar, const std::vector<T> &mus,
                 const T comp = 0.);
  ContactProblem(const Ref<const MatrixXs> &M, const Ref<const MatrixXs> &J,
                 const Ref<const VectorXs> &dqf,
                 const Ref<const VectorXs> &vstar, const std::vector<T> &mus,
                 const Ref<const VectorXs> &comp);
  ContactProblem(const Ref<const MatrixXs> &G, const Ref<const VectorXs> &g,
                 const Ref<const MatrixXs> &M, const Ref<const MatrixXs> &J,
                 const Ref<const VectorXs> &dqf,
                 const Ref<const VectorXs> &vstar, const std::vector<T> &mus,
                 const T comp = 0.);
  ContactProblem(const Ref<const MatrixXs> &G, const Ref<const VectorXs> &g,
                 const Ref<const MatrixXs> &M, const Ref<const MatrixXs> &J,
                 const Ref<const VectorXs> &dqf,
                 const Ref<const VectorXs> &vstar, const std::vector<T> &mus,
                 const Ref<const VectorXs> &comp);
  ContactProblem(std::shared_ptr<DelassusBase<T>> Del,
                 const Ref<const VectorXs> &g, const Ref<const MatrixXs> &M,
                 const Ref<const MatrixXs> &J, const Ref<const VectorXs> &dqf,
                 const Ref<const VectorXs> &vstar, const std::vector<T> &mus,
                 const T comp = 0.);
  ContactProblem(std::shared_ptr<DelassusBase<T>> Del,
                 const Ref<const VectorXs> &g, const Ref<const MatrixXs> &M,
                 const Ref<const MatrixXs> &J, const Ref<const VectorXs> &dqf,
                 const Ref<const VectorXs> &vstar, const std::vector<T> &mus,
                 const Ref<const VectorXs> &comp);

  void setCompliance(const T comp);

  void setCompliance(const Ref<const VectorXs> &comp);

  template <typename VecIn, typename VecOut>
  static void project(const std::vector<T> &mus,
                      const Eigen::MatrixBase<VecIn> &lam,
                      const Eigen::MatrixBase<VecOut> &lam_out);

  template <typename VecIn, typename VecOut>
  void project(const Eigen::MatrixBase<VecIn> &lam,
               const Eigen::MatrixBase<VecOut> &lam_out) const;

  template <typename VecIn, typename VecOut>
  void projectDual(const Eigen::MatrixBase<VecIn> &v,
                   const Eigen::MatrixBase<VecOut> &v_out) const;

  bool isInside(const Ref<const VectorXs> &lam, T thresh) const;

  bool isInsideDual(const Ref<const VectorXs> &v, T thresh) const;

  template <typename VecIn, typename VecOut>
  void computeVelocity(const Eigen::MatrixBase<VecIn> &lam,
                       const Eigen::MatrixBase<VecOut> &v_out_) const;

  void computeDeSaxceCorrection(const Ref<const VectorXs> &v,
                                VectorRef v_out) const;

  static T computeComplementarity(const Ref<const VectorXs> &lam,
                                  const Ref<const VectorXs> &v);

  T computeContactComplementarity(const Ref<const VectorXs> &lam);

  T computeContactComplementarity(const Ref<const VectorXs> &lam,
                                  const Ref<const VectorXs> &v) const;

  T computePerContactContactComplementarity(usize i, const Vector3s &lam,
                                            const Vector3s &v) const;

  T computeSignoriniComplementarity(const Ref<const VectorXs> &lam);

  T computeSignoriniComplementarity(const Ref<const VectorXs> &lam,
                                    const Ref<const VectorXs> &v) const;

  T computePerContactSignoriniComplementarity(usize i, const Vector3s &lam,
                                              const Vector3s &v) const;

  T computeConicComplementarity(const Ref<const VectorXs> &lam);

  T computeConicComplementarity(const Ref<const VectorXs> &lam,
                                const Ref<const VectorXs> &v) const;

  T computePerContactConicComplementarity(usize i, const Vector3s &lam,
                                          const Vector3s &v) const;

  T computeLinearComplementarity(const Ref<const VectorXs> &lam);

  T computeLinearComplementarity(const Ref<const VectorXs> &lam,
                                 const Ref<const VectorXs> &v);

  void setLCP();

  void computeLCP(const T eps_reg = 0.);

  void computeLCP(const Ref<const VectorXs> &R_reg);

  void computeTangentLCP(const Ref<const VectorXs> &lam,
                         const Ref<const VectorXs> &R_reg);

  void computeTangentLCP(const Ref<const VectorXs> &lam, const T esp_reg = 0.);

  void computeInscribedLCP(const Ref<const VectorXs> &lam,
                           const Ref<const VectorXs> &R_reg);

  void computeInscribedLCP(const Ref<const VectorXs> &lam,
                           const T eps_reg = 0.);

  void setLCCP();

  void computeLCCP();

  void computeTangentLCCP(const Ref<const VectorXs> &lam);

  void computeInscribedLCCP(const Ref<const VectorXs> &lam);

  void computeLCPSolution(const Ref<const VectorXs> &lam,
                          const Ref<const VectorXs> &c,
                          VectorRef lam_lcp) const;

  void computeLCPSolution(const Ref<const VectorXs> &lam, VectorRef lam_lcp);

  void computeLCPdLdg(const Ref<const VectorXs> &dL_db, VectorRef dL_dg);
  void computeLCPdLdDel(const Ref<const MatrixXs> &dL_dA, MatrixRef dL_dDel);
  void computeLCPdLdmus(const Ref<const MatrixXs> &dL_dA, VectorRef dL_dmus);

  void computeLCPdbdtheta(const Ref<const MatrixXs> &dg_dtheta,
                          MatrixRef db_dtheta);

  void computeLCPdAdtheta(const Ref<const MatrixXs> &dmus_dtheta,
                          const Ref<const MatrixXs> &dG_dtheta,
                          const Ref<const VectorXs> &lam, MatrixRef mat_out);

  void computeInscribedLCPdLdmus(const Ref<const MatrixXs> &dL_dA,
                                 VectorRef dL_dmus);
  void computeTangentLCPdLdmus(const Ref<const MatrixXs> &dL_dA,
                               VectorRef dL_dmus);

  void computeLCCPdLdmus(const Ref<const MatrixXs> &dL_dC, VectorRef dL_dmus);

  void computeInscribedLCCPdLdmus(const Ref<const MatrixXs> &dL_dC,
                                  VectorRef dL_dmus);

  void computeTangentLCCPdLdmus(const Ref<const MatrixXs> &dL_dC,
                                VectorRef dL_dmus);

  const Ref<const MatrixXs> getLinearInConstraintMatrix() const;

  const Ref<const MatrixXs> getLinearEqConstraintMatrix() const;

  template <typename VecIn, typename VecOut>
  void computeVelInTangentBasis(const MatrixBase<VecIn> &v,
                                const MatrixBase<VecOut> &v_out) const;

  template <typename VecIn, typename VecOut>
  void computeVelInInscribedBasis(const MatrixBase<VecIn> &v,
                                  const MatrixBase<VecOut> &v_out) const;

  template <typename VecIn, typename VecIn2, typename VecOut, typename VecOut2>
  void computeDualSolutionOfTangent(const MatrixBase<VecIn> &lam,
                                    const MatrixBase<VecIn2> &v,
                                    const MatrixBase<VecOut> &y_out,
                                    const MatrixBase<VecOut2> &z_out,
                                    const double eps) const;

  template <typename VecIn, typename VecIn2, typename VecOut, typename VecOut2>
  void computeDualSolutionOfInscribed(const MatrixBase<VecIn> &lam,
                                      const MatrixBase<VecIn2> &v,
                                      const MatrixBase<VecOut> &y_out,
                                      const MatrixBase<VecOut2> &z_out,
                                      const double eps) const;

  template <typename NewScalar>
  ContactProblem<NewScalar, ConstraintTpl> cast() const {
    using NewContactProblem = ContactProblem<NewScalar, ConstraintTpl>;
    NewContactProblem newProblem;

    newProblem.nc_ = nc_;
    newProblem.n_breaking_ = n_breaking_;
    newProblem.n_in_tan_ = n_in_tan_;
    newProblem.n_eq_tan_ = n_eq_tan_;

    newProblem.contact_constraints_.reserve(contact_constraints_.size());
    for (const auto &constraint : contact_constraints_) {
      newProblem.contact_constraints_.push_back(
          constraint.template cast<NewScalar>());
    }

    newProblem.g_ = g_.template cast<NewScalar>();
    newProblem.dqf_ = dqf_.template cast<NewScalar>();
    newProblem.vstar_ = vstar_.template cast<NewScalar>();
    newProblem.R_comp_ = R_comp_.template cast<NewScalar>();
    newProblem.M_ = M_.template cast<NewScalar>();
    newProblem.J_ = J_.template cast<NewScalar>();
    newProblem.UD_ = UD_.template cast<NewScalar>();
    newProblem.A_ = A_.template cast<NewScalar>();
    newProblem.b_ = b_.template cast<NewScalar>();

    return newProblem;
  }

protected:
  VectorXs v_, v_comp_, v_tmp_;
  MatrixXs GUD_, G_reg_, E_, F_, H_, C_in_, C_eq_;
  MatrixXs dL_dDel_t_;

  void computeAb_(const T eps_reg = 0.);
  void computeAb_(const Ref<const VectorXs> &R_reg);
  void computeC_();
  VectorXs lam_lcp_, v_lcp_;
};

} // end namespace contactbench

#endif

#include "contactbench/contact-problem.hxx"

#if CONTACTBENCH_ENABLE_TEMPLATE_INSTANTIATION
#include "contactbench/contact-problem.txx"
#endif // CONTACTBENCH_ENABLE_TEMPLATE_INSTANTIATION
