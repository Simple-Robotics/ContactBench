#pragma once
#include "contactbench/friction-constraint.hpp"
#include <iostream>
// #include <pinocchio/utils/static-if.hpp>
#include <cmath>

namespace contactbench {
namespace {

using Eigen::Infinity;
using Eigen::MatrixBase;
} // namespace

template <typename T, typename VecIn, typename VecIn2>
T computeContactComplementarity(const MatrixBase<VecIn> &lam,
                                const MatrixBase<VecIn2> &v, const T mu) {
  using Vector3s = Eigen::Matrix<T, 3, 1>;
  T norm_vt = v.template head<2>().norm();
  Vector3s dual = v;
  dual(2) += mu * norm_vt;
  return abs(dual.dot(lam));
}

template <typename T, typename VecIn, typename VecIn2>
T computeSignoriniComplementarity(const MatrixBase<VecIn> &lam,
                                  const MatrixBase<VecIn2> &v) {
  return abs(v(2) * lam(2));
}

template <typename T, typename VecIn, typename VecIn2>
T computeConicComplementarity(const MatrixBase<VecIn> &lam,
                              const MatrixBase<VecIn2> &v) {
  return abs(v.dot(lam));
}

template <typename T, typename MatOut>
void computeRotMat2D(const T theta, const MatrixBase<MatOut> &R_out_) {
  auto &R_out = R_out_.const_cast_derived();
  R_out(0, 0) = cos(theta);
  R_out(1, 1) = cos(theta);
  R_out(0, 1) = -sin(theta);
  R_out(1, 0) = sin(theta);
}

//
//
//  IceCreamCone
//
//
//

static Eigen::Matrix2d PI4ROT = [] {
  Eigen::Matrix2d matrix;
  matrix << std::cos(M_PI / 4), -std::sin(M_PI / 4), std::sin(M_PI / 4),
      std::cos(M_PI / 4);
  return matrix;
}();

template <typename T>
template <typename VecIn, typename VecOut>
void IceCreamCone<T>::project(const T &mu, const MatrixBase<VecIn> &x,
                              const MatrixBase<VecOut> &x_out_) {
  T norm_x = x.template head<2>().norm();
  auto &x_out = x_out_.const_cast_derived();
  if (norm_x <= -(1 / mu) * x(2)) {
    x_out = Vector3s::Zero();
  } else {
    if (norm_x <= mu * x(2)) {
      x_out = x;
    } else {
      x_out(2) = (mu * norm_x + x(2)) / (mu * mu + 1);
      x_out.template head<2>() = x.template head<2>() * mu * x_out(2) / norm_x;
    }
  }
}

template <typename T>
template <typename VecIn, typename VecOut>
void IceCreamCone<T>::project(const MatrixBase<VecIn> &x,
                              const MatrixBase<VecOut> &x_out_) const {
  IceCreamCone<T>::project(mu_, x, x_out_);
}

template <typename T>
template <typename VecIn, typename VecOut>
void IceCreamCone<T>::projectDual(const MatrixBase<VecIn> &x,
                                  const MatrixBase<VecOut> &x_out_) const {
  T mu_dual = 1 / mu_;
  IceCreamCone<T>::project(mu_dual, x, x_out_);
}

template <typename T>
template <typename VecIn>
bool IceCreamCone<T>::isInside(const T &mu, const Eigen::MatrixBase<VecIn> &x,
                               T thresh) {
  T norm_x = x.template head<2>().norm();
  if (norm_x - mu * x(2) <= -thresh) {
    return true;
  } else {
    return false;
  }
}

template <typename T>
template <typename VecIn>
bool IceCreamCone<T>::isInside(const Eigen::MatrixBase<VecIn> &x,
                               T thresh) const {
  return IceCreamCone<T>::isInside(mu_, x, thresh);
}

template <typename T>
template <typename VecIn>
bool IceCreamCone<T>::isInsideDual(const Eigen::MatrixBase<VecIn> &x,
                                   T thresh) const {
  double mu_dual = 1 / mu_;
  return IceCreamCone<T>::isInside(mu_dual, x, thresh);
}

template <typename T>
template <typename VecIn>
bool IceCreamCone<T>::isOnBorder(const T &mu, const Eigen::MatrixBase<VecIn> &x,
                                 T thresh) {
  T norm_x = x.template head<2>().norm();
  if (abs(norm_x - mu * x(2)) <= thresh) {
    return true;
  } else {
    return false;
  }
}

template <typename T>
template <typename VecIn>
bool IceCreamCone<T>::isOnBorder(const Eigen::MatrixBase<VecIn> &x,
                                 T thresh) const {
  return IceCreamCone<T>::isOnBorder(mu_, x, thresh);
}

template <typename T>
template <typename VecIn>
bool IceCreamCone<T>::isOnBorderDual(const Eigen::MatrixBase<VecIn> &x,
                                     T thresh) const {
  double mu_dual = 1 / mu_;
  return IceCreamCone<T>::isOnBorder(mu_dual, x, thresh);
}

template <typename T>
template <typename VecIn, typename VecOut>
void IceCreamCone<T>::projectHorizontal(const T &mu, const MatrixBase<VecIn> &x,
                                        const MatrixBase<VecOut> &x_out_) {
  // static_assert(std::is_same<Vector3d,MatrixBase<Vec>>::value,
  // "projectHorizontal requires vectors of dimension 3 as inputs.");
  T norm_x = x.template head<2>().norm();
  T radius = mu * x(2);
  auto &x_out = x_out_.const_cast_derived();
  if (norm_x <= radius) {
    x_out = x;
  } else {
    x_out(2) = x(2);
    x_out.template head<2>() = radius * x.template head<2>() / norm_x;
  }
  // This here fixing the if-else branching issue for CppAd
  // x_out(0) = pinocchio::internal::if_then_else(
  //     pinocchio::internal::LT, norm_x, radius, x(0),
  //     radius * x(0) / norm_x);

  // x_out(1) = pinocchio::internal::if_then_else(
  //     pinocchio::internal::LT, norm_x, radius, x(1),
  //     radius * x(1) / norm_x);
  // x_out(2) = x(2);
}

template <typename T>
template <typename VecIn, typename VecOut>
void IceCreamCone<T>::projectHorizontal(
    const MatrixBase<VecIn> &x, const MatrixBase<VecOut> &x_out_) const {
  IceCreamCone<T>::projectHorizontal(mu_, x, x_out_);
}

template <typename T>
template <typename VecIn, typename VecOut>
void IceCreamCone<T>::projectHorizontal(
    const MatrixBase<VecIn> &x, double x_n,
    const MatrixBase<VecOut> &x_out_) const {
  T norm_x = x.norm();
  T radius = mu_ * x_n;
  auto &x_out = x_out_.const_cast_derived();
  if (norm_x <= radius) {
    x_out = x;
  } else {
    x_out = radius * x / norm_x;
  }
}

template <typename T>
template <typename VecIn, typename VecOut>
void IceCreamCone<T>::computeDeSaxceCorrection(
    const MatrixBase<VecIn> &v, const MatrixBase<VecOut> &v_out_) const {
  T norm_vt = v.template head<2>().norm();
  auto &v_out = v_out_.const_cast_derived();
  v_out = v;
  v_out(2) += mu_ * norm_vt;
}

template <typename T>
template <typename VecIn, typename VecIn2>
T IceCreamCone<T>::computeContactComplementarity(
    const MatrixBase<VecIn> &lam, const MatrixBase<VecIn2> &v) const {
  return contactbench::computeContactComplementarity<T, VecIn, VecIn2>(lam, v,
                                                                       mu_);
}

template <typename T>
template <typename VecIn, typename VecIn2>
T IceCreamCone<T>::computeSignoriniComplementarity(
    const MatrixBase<VecIn> &lam, const MatrixBase<VecIn2> &v) const {
  return contactbench::computeSignoriniComplementarity<T, VecIn, VecIn2>(lam,
                                                                         v);
}

template <typename T>
template <typename VecIn, typename VecIn2>
T IceCreamCone<T>::computeConicComplementarity(
    const MatrixBase<VecIn> &lam, const MatrixBase<VecIn2> &v) const {
  return contactbench::computeConicComplementarity<T, VecIn, VecIn2>(lam, v);
}

// template <typename T>
// template <typename VecIn>
// void IceCreamCone<T>::computeTangentPyramid(const MatrixBase<VecIn> &lam) {
//   tangentPyramid_.setOrientation(lam.template head<2>());
// }

// template <typename T>
// template <typename VecIn>
// void IceCreamCone<T>::computeInscribedPyramid(const MatrixBase<VecIn> &lam) {
//   inscribedPyramid_.setOrientation(PI4ROT.transpose() * lam.template
//   head<2>());
// }

// template <typename T>
// const Eigen::Matrix<T, 2, 2> &IceCreamCone<T>::getTangentBasisMatrix() const
// {
//   return tangentPyramid_.Uxy_;
// }

// template <typename T>
// const Eigen::Matrix<T, 2, 2> &IceCreamCone<T>::getInscribedBasisMatrix()
// const {
//   return inscribedPyramid_.Uxy_;
// }

// template <typename T>
// const Eigen::Matrix<T, 3, 6> &
// IceCreamCone<T>::getLCPTangentBasisMatrix() const {
//   return tangentPyramid_.UD_lcp_;
// }

// template <typename T>
// const Eigen::Matrix<T, 3, 6> &
// IceCreamCone<T>::getLCPInscribedBasisMatrix() const {
//   return inscribedPyramid_.UD_lcp_;
// }

// template <typename T>
// template <typename VecIn, typename VecOut>
// void IceCreamCone<T>::computeVelInTangentBasis(
//     const MatrixBase<VecIn> &v, const MatrixBase<VecOut> &v_out_) const {
//   auto &v_out = v_out_.const_cast_derived();
//   tangentPyramid_.computeCoord(v, v_out);
// }

// template <typename T>
// template <typename VecIn, typename VecOut>
// void IceCreamCone<T>::computeVelInInscribedBasis(
//     const MatrixBase<VecIn> &v, const MatrixBase<VecOut> &v_out_) const {
//   auto &v_out = v_out_.const_cast_derived();
//   inscribedPyramid_.computeCoord(v, v_out);
// }

//
//
//  PyramidCone
//
//
//

template <typename T>
template <typename VecIn>
void PyramidCone<T>::setOrientation(const MatrixBase<VecIn> &ux) {
  if (ux.norm() <= 0) {
    Uxy_.setIdentity();
  } else {
    Uxy_.col(0) = ux;
    Uxy_.col(0).normalize();
    Uxy_(0, 1) = -Uxy_(1, 0);
    Uxy_(1, 1) = Uxy_(0, 0);
  }
  UD_ = Uxy_ * D_;
  UD_lcp_.setZero();
  UD_lcp_(2, 4) = 1.;
  UD_lcp_.template block<2, 4>(0, 0) = UD_;
}

template <typename T>
template <typename VecIn, typename VecOut>
void PyramidCone<T>::computeCoord(const MatrixBase<VecIn> &x,
                                  const MatrixBase<VecOut> &x_out_) const {
  auto &x_out = x_out_.const_cast_derived();
  x_out = Uxy_.transpose() * x;
}

template <typename T>
template <typename VecIn, typename VecOut>
void PyramidCone<T>::project(const MatrixBase<VecIn> &x,
                             const MatrixBase<VecOut> &x_out_) const {
  // TODO
  CONTACTBENCH_UNUSED(x);
  CONTACTBENCH_UNUSED(x_out_);
}

template <typename T>
template <typename VecIn, typename VecOut>
void PyramidCone<T>::projectDual(const MatrixBase<VecIn> &x,
                                 const MatrixBase<VecOut> &x_out_) const {
  // TODO
  CONTACTBENCH_UNUSED(x);
  CONTACTBENCH_UNUSED(x_out_);
}

template <typename T>
template <typename VecIn>
bool PyramidCone<T>::isInside(const Eigen::MatrixBase<VecIn> &x,
                              T thresh) const {
  T norm_x =
      (Uxy_.transpose() * x.template head<2>()).template lpNorm<Infinity>();
  if (norm_x - mu_ * x(2) <= thresh) {
    return true;
  } else {
    return false;
  }
}

template <typename T>
template <typename VecIn>
bool PyramidCone<T>::isInsideDual(const Eigen::MatrixBase<VecIn> &x,
                                  T thresh) const {
  // TODO
  CONTACTBENCH_UNUSED(x);
  CONTACTBENCH_UNUSED(thresh);
  return false;
}

template <typename T>
template <typename VecIn, typename VecOut>
void PyramidCone<T>::projectHorizontal(const MatrixBase<VecIn> &x,
                                       const MatrixBase<VecOut> &x_out_) const {
  T radius = mu_ * x(2);
  auto &x_out = x_out_.const_cast_derived();
  x_out(2) = x(2);
  x_out.template head<2>() =
      (Uxy_.transpose() * x.template head<2>()).cwiseMin(radius);
  x_out.template head<2>() = x_out.template head<2>().cwiseMax(-radius);
  x_out.template head<2>() = Uxy_ * x_out.template head<2>();
}

template <typename T>
template <typename VecIn, typename VecOut>
void PyramidCone<T>::projectHorizontal(const MatrixBase<VecIn> &x, double x_n,
                                       const MatrixBase<VecOut> &x_out_) const {
  T radius = mu_ * x_n;
  auto &x_out = x_out_.const_cast_derived();
  x_out = (Uxy_.transpose() * x).cwiseMin(radius);
  x_out = x_out.cwiseMax(-radius);
  x_out = Uxy_ * x_out;
}

template <typename T>
template <typename VecIn, typename VecOut>
void PyramidCone<T>::computeDeSaxceCorrection(
    const MatrixBase<VecIn> &v, const MatrixBase<VecOut> &v_out_) const {
  T norm_vt = v.template head<2>().norm();
  auto &v_out = v_out_.const_cast_derived();
  v_out = v;
  v_out(2) += mu_ * norm_vt;
}

template <typename T>
template <typename VecIn, typename VecIn2>
T PyramidCone<T>::computeContactComplementarity(
    const MatrixBase<VecIn> &lam, const MatrixBase<VecIn2> &v) const {
  return contactbench::computeContactComplementarity<T, VecIn, VecIn2>(lam, v,
                                                                       mu_);
}

template <typename T>
template <typename VecIn, typename VecIn2>
T PyramidCone<T>::computeSignoriniComplementarity(
    const MatrixBase<VecIn> &lam, const MatrixBase<VecIn2> &v) const {
  return contactbench::computeSignoriniComplementarity<T, VecIn, VecIn2>(lam,
                                                                         v);
}

template <typename T>
template <typename VecIn, typename VecIn2>
T PyramidCone<T>::computeConicComplementarity(
    const MatrixBase<VecIn> &lam, const MatrixBase<VecIn2> &v) const {
  return contactbench::computeConicComplementarity<T, VecIn, VecIn2>(lam, v);
}

// TODO add linear complementarity

template <typename T>
template <typename VecIn, typename VecOut>
void PyramidCone<T>::computeCoordinatesInD(
    const MatrixBase<VecIn> &x, const MatrixBase<VecOut> &x_out_) const {

  auto &x_out = x_out_.const_cast_derived();
  Vector2s x_u = Uxy_.transpose() * x.template head<2>();
  if (x_u(1) > x_u(0)) {
    if (x_u(1) > -x_u(0)) {
      x_out(2) = 0.;
      x_out(3) = 0.;
      computeRotMat2D(-M_PI / 4, D_red_inv_);
      x_out(0) = (D_red_inv_.row(0) * x_u.template head<2>()).value();
      x_out(1) = (D_red_inv_.row(1) * x_u.template head<2>()).value();
    } else {
      x_out(0) = 0.;
      x_out(3) = 0.;
      computeRotMat2D(-3. * M_PI / 4, D_red_inv_);
      x_out(1) = (D_red_inv_.row(0) * x_u.template head<2>()).value();
      x_out(2) = (D_red_inv_.row(1) * x_u.template head<2>()).value();
    }
  } else {
    if (x_u(1) > -x_u(0)) {
      x_out(1) = 0.;
      x_out(2) = 0.;
      computeRotMat2D(-7. * M_PI / 4, D_red_inv_);
      x_out(0) = (D_red_inv_.row(1) * x_u.template head<2>()).value();
      x_out(3) = (D_red_inv_.row(0) * x_u.template head<2>()).value();

    } else {
      x_out(0) = 0.;
      x_out(1) = 0.;
      computeRotMat2D(-5. * M_PI / 4, D_red_inv_);
      x_out(2) = (D_red_inv_.row(0) * x_u.template head<2>()).value();
      x_out(3) = (D_red_inv_.row(1) * x_u.template head<2>()).value();
    }
  }
}

template <typename T>
template <typename MatOut>
void PyramidCone<T>::computeRotMat2D(const T theta,
                                     const MatrixBase<MatOut> &R_out_) const {
  contactbench::computeRotMat2D<T, MatOut>(theta, R_out_);
}

//
//
//  Box
//
//

template <typename T>
template <typename VecIn, typename VecOut>
void Box<T>::project(const MatrixBase<VecIn> &x,
                     const MatrixBase<VecOut> &x_out_) const {
  auto &x_out = x_out_.const_cast_derived();
  x_out = x.min(mus_);
  x_out = x_out.max(-mus_);
}

template <typename T>
template <typename VecIn>
bool Box<T>::isInside(const Eigen::MatrixBase<VecIn> &x, T thresh) const {
  CONTACTBENCH_UNUSED(x);
  CONTACTBENCH_UNUSED(thresh);
  return false;
}

} // namespace contactbench
