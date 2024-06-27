#pragma once
#include "contactbench/delassus-wrapper.hpp"
#include <Eigen/src/Core/Ref.h>

namespace contactbench {
namespace {

using Eigen::MatrixBase;
using Eigen::Ref;
} // namespace

template <typename T>
DelassusDense<T>::DelassusDense(const Ref<const MatrixXs> &G) : Base(G) {
  G_llt_.resize(3 * nc_, 3 * nc_);
  llt_ = Eigen::LLT<MatrixXs>(3 * nc_);
  R_reg_.resize(3 * nc_);
  Gvpow_.resize(3 * nc_);
  vpow_.resize(3 * nc_);
  err_vpow_.resize(3 * nc_);
}

template <typename T> void DelassusDense<T>::evaluateDel() {}

template <typename T> void DelassusDense<T>::evaluateDiagDel() {}

template <typename T>
void DelassusDense<T>::applyOnTheRight(const Ref<const VectorXs> &x,
                                       VectorRef x_out_) const {
  x_out_.noalias() = G_ * x;
}

template <typename T>
void DelassusDense<T>::applyPerContactOnTheRight(int i,
                                                 const Ref<const VectorXs> &x,
                                                 VectorRef x_out_) const {
  // evaluateDel should be called beforehand
  assert(i < nc_);
  x_out_.noalias() = G_.template middleRows<3>(3 * i) * x;
}

template <typename T>
void DelassusDense<T>::applyPerContactNormalOnTheRight(
    int i, const Ref<const VectorXs> &x, T &x_out_) const {
  // evaluateDel should be called beforehand
  assert(i < nc_);
  x_out_ = (G_.row(3 * i + 2) * x).value();
}

template <typename T>
void DelassusDense<T>::applyPerContactTangentOnTheRight(
    int i, const Ref<const VectorXs> &x, VectorRef x_out_) const {
  // evaluateDel should be called beforehand
  assert(i < nc_);
  x_out_.noalias() = G_.template middleRows<2>(3 * i) * x;
}

// template <typename T>
// T DelassusDense<T>::computeLargestEigenValue(const int max_it, const T tol) {
//   // vpow_.setOnes();
//   // vpow_ /= std::sqrt(vpow_.size());
//   // for (int i = 0; i < max_it; i++) {
//   //   applyOnTheRight(vpow_, Gvpow_);
//   //   // Gvpow_.noalias() = G * vpow_;
//   //   vpow_ = Gvpow_;
//   //   vpow_.normalize();
//   // };
//   // applyOnTheRight(vpow_, Gvpow_);
//   // // Gvpow_.noalias() = G_ * vpow_;
//   // T l_max = vpow_.dot(Gvpow_);
//   // return l_max;
//   T lam = proxsuite::proxqp::dense::power_iteration<T>(G_, Gvpow_, vpow_,
//                                                        err_vpow_, tol, max_it);
//   return lam;
// }

template <typename T> void DelassusDense<T>::computeChol(T mu) {
  G_llt_ = G_;
  G_llt_.diagonal().array() += mu;
  llt_.compute(G_llt_);
  R_reg_ = mu * VectorXs::Ones(3 * nc_);
}

template <typename T>
void DelassusDense<T>::computeChol(const Ref<const VectorXs> &mus) {
  G_llt_ = G_;
  G_llt_.diagonal() += mus;
  llt_.compute(G_llt_);
  R_reg_ = mus;
}

template <typename T> void DelassusDense<T>::updateChol(T mu) {
  G_llt_ = G_;
  G_llt_.diagonal().array() += mu;
  llt_.compute(G_llt_);
  R_reg_ = mu * VectorXs::Ones(3 * nc_);
}

template <typename T>
void DelassusDense<T>::updateChol(const Ref<const VectorXs> &mus) {
  G_llt_ = G_;
  G_llt_.diagonal() += mus;
  llt_.compute(G_llt_);
  R_reg_ = mus;
}

template <typename T>
void DelassusDense<T>::solve(const Ref<const VectorXs> &x,
                             VectorRef x_out_) const {
  x_out_ = llt_.solve(x);
}

template <typename T> void DelassusDense<T>::solveInPlace(VectorRef x) const {
  llt_.solveInPlace(x);
}

template <typename T>
DelassusPinocchio<T>::DelassusPinocchio(
    const Model &model, const Data &data,
    const RigidConstraintModelVector &contact_models,
    const RigidConstraintDataVector &contact_datas)
    : Base(), model_(model), data_(data), contact_models_(contact_models),
      contact_datas_(contact_datas) {
  contact_chol_ = ContactCholeskyDecomposition(model_, contact_models_);
  nc_ = (int)contact_chol_.numContacts();
  G_.resize(3 * nc_, 3 * nc_);
  R_reg_.resize(3 * nc_);
}

template <typename T> void DelassusPinocchio<T>::evaluateDel() {
  // computeChol should be called beforehand
  G_ = contact_chol_.getInverseOperationalSpaceInertiaMatrix();
  G_.diagonal() -= R_reg_;
}

template <typename T> void DelassusPinocchio<T>::evaluateDiagDel() {
  // TODO
  G_ = contact_chol_.getInverseOperationalSpaceInertiaMatrix();
  G_.diagonal() -= R_reg_;
}

template <typename T>
void DelassusPinocchio<T>::applyOnTheRight(const Ref<const VectorXs> &x,
                                           VectorRef x_out_) const {
  // evaluateDel should be called beforehand
  x_out_.noalias() = G_ * x;
}

template <typename T>
void DelassusPinocchio<T>::applyPerContactOnTheRight(
    int i, const Ref<const VectorXs> &x, VectorRef x_out_) const {
  // evaluateDel should be called beforehand
  assert(i < nc_);
  x_out_.noalias() = G_.template middleRows<3>(3 * i) * x;
}

template <typename T>
void DelassusPinocchio<T>::applyPerContactNormalOnTheRight(
    int i, const Ref<const VectorXs> &x, T &x_out_) const {
  // evaluateDel should be called beforehand
  assert(i < nc_);
  x_out_ = (G_.row(3 * i + 2) * x).value();
}

template <typename T>
void DelassusPinocchio<T>::applyPerContactTangentOnTheRight(
    int i, const Ref<const VectorXs> &x, VectorRef x_out_) const {
  // evaluateDel should be called beforehand
  assert(i < nc_);
  x_out_.noalias() = G_.template middleRows<2>(3 * i) * x;
}

// template <typename T>
// T DelassusPinocchio<T>::computeLargestEigenValue(const int max_it,
//                                                  const T rel_tol) {
//   auto chol_expr = contact_chol_.getDelassusCholeskyExpression();
//   return chol_expr.computeLargestEigenValue(true, max_it, rel_tol);
// }

template <typename T> void DelassusPinocchio<T>::computeChol(T mu) {
  contact_chol_.compute(model_, data_, contact_models_, contact_datas_, mu);
  R_reg_ = mu * VectorXs::Ones(3 * nc_);
}

template <typename T>
void DelassusPinocchio<T>::computeChol(const Ref<const VectorXs> &mus) {
  contact_chol_.compute(model_, data_, contact_models_, contact_datas_, mus);
  R_reg_ = mus;
}

template <typename T> void DelassusPinocchio<T>::updateChol(T mu) {
  contact_chol_.updateDamping(mu);
  R_reg_ = mu * VectorXs::Ones(3 * nc_);
}

template <typename T>
void DelassusPinocchio<T>::updateChol(const Ref<const VectorXs> &mus) {
  contact_chol_.updateDamping(mus);
  R_reg_ = mus;
}

template <typename T>
void DelassusPinocchio<T>::solve(const Ref<const VectorXs> &x,
                                 VectorRef x_out_) const {
  auto chol_expr = contact_chol_.getDelassusCholeskyExpression();
  x_out_ = chol_expr.solve(x);
}

template <typename T>
void DelassusPinocchio<T>::solveInPlace(VectorRef x) const {
  auto chol_expr = contact_chol_.getDelassusCholeskyExpression();
  chol_expr.solveInPlace(x);
}

} // namespace contactbench
