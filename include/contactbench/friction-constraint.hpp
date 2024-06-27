#ifndef CONTACT_BENCH_FRICTION_H
#define CONTACT_BENCH_FRICTION_H

#include <Eigen/Core>
#include <cmath>
#include "contactbench/fwd.hpp"
namespace contactbench {

template <typename T> struct FrictionConstraint {

  int n_cons_;

  FrictionConstraint(){};

  template <typename VecIn, typename VecOut>
  static void
  project(CONTACTBENCH_MAYBE_UNUSED const T &mu,
          CONTACTBENCH_MAYBE_UNUSED const Eigen::MatrixBase<VecIn> &x,
          CONTACTBENCH_MAYBE_UNUSED const Eigen::MatrixBase<VecOut> &x_out_) {
    CONTACTBENCH_RUNTIME_ERROR("Unimplemented.");
  }

  template <typename VecIn, typename VecOut>
  void project(
      CONTACTBENCH_MAYBE_UNUSED const Eigen::MatrixBase<VecIn> &x,
      CONTACTBENCH_MAYBE_UNUSED const Eigen::MatrixBase<VecOut> &x_out_) const {
    CONTACTBENCH_RUNTIME_ERROR("Unimplemented.");
  }

  template <typename NewScalar> FrictionConstraint<NewScalar> cast() const {
    FrictionConstraint<NewScalar> newConstraint;
    newConstraint.n_cons_ = n_cons_;
    return newConstraint;
  }
};

template <typename T> struct PyramidCone : FrictionConstraint<T> {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);

  int n_cons_;
  T mu_;
  int n_facets_ = 4;
  Matrix2s Uxy_;
  Vector4s e_;
  Matrix24s D_, UD_;
  Matrix36s D_lcp_, UD_lcp_;

  PyramidCone() { n_cons_ = 3; };
  PyramidCone(T mu) {
    n_cons_ = 3;
    mu_ = mu;
    e_.setOnes();
    D_ << 1., -1., -1., 1., 1., 1., -1., -1.;
    D_ /= std::sqrt(2);
    D_lcp_.setZero();
    D_lcp_(2, 4) = 1.;
    D_lcp_.template block<2, 4>(0, 0) = D_;
    Uxy_ = Matrix2s::Identity();
    UD_ = Uxy_ * D_;
    UD_lcp_.setZero();
    UD_lcp_(2, 4) = 1.;
    UD_lcp_.template block<2, 4>(0, 0) = UD_;
  }
  template <typename VecIn>
  PyramidCone(T mu, const Eigen::MatrixBase<VecIn> &ux) : PyramidCone(mu) {
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

  template <typename VecIn>
  void setOrientation(const Eigen::MatrixBase<VecIn> &ux);

  template <typename VecIn, typename VecOut>
  void computeCoord(const Eigen::MatrixBase<VecIn> &x,
                    const Eigen::MatrixBase<VecOut> &x_out) const;

  template <typename VecIn, typename VecOut>
  void project(const Eigen::MatrixBase<VecIn> &x,
               const Eigen::MatrixBase<VecOut> &x_out) const;

  template <typename VecIn, typename VecOut>
  void projectDual(const Eigen::MatrixBase<VecIn> &x,
                   const Eigen::MatrixBase<VecOut> &x_out) const;

  template <typename VecIn>
  bool isInside(const Eigen::MatrixBase<VecIn> &x, T thresh = 1e-3) const;

  template <typename VecIn>
  bool isInsideDual(const Eigen::MatrixBase<VecIn> &x, T thresh = 1e-3) const;

  template <typename VecIn, typename VecOut>
  void projectHorizontal(const Eigen::MatrixBase<VecIn> &x,
                         const Eigen::MatrixBase<VecOut> &x_out_) const;

  template <typename VecIn, typename VecOut>
  void projectHorizontal(const Eigen::MatrixBase<VecIn> &x, double x_n,
                         const Eigen::MatrixBase<VecOut> &x_out_) const;

  template <typename VecIn, typename VecIn2>
  T computeConicComplementarity(const Eigen::MatrixBase<VecIn> &lam,
                                const Eigen::MatrixBase<VecIn2> &v) const;

  template <typename VecIn, typename VecIn2>
  T computeSignoriniComplementarity(const Eigen::MatrixBase<VecIn> &lam,
                                    const Eigen::MatrixBase<VecIn2> &v) const;

  template <typename VecIn, typename VecOut>
  void computeDeSaxceCorrection(const Eigen::MatrixBase<VecIn> &v,
                                const Eigen::MatrixBase<VecOut> &v_out_) const;

  template <typename VecIn, typename VecIn2>
  T computeContactComplementarity(const Eigen::MatrixBase<VecIn> &lam,
                                  const Eigen::MatrixBase<VecIn2> &v) const;

  template <typename VecIn, typename VecIn2>
  T computeLinearComplementarity(const Eigen::MatrixBase<VecIn> &lam,
                                 const Eigen::MatrixBase<VecIn2> &v) const;

  template <typename VecIn, typename VecOut>
  void computeCoordinatesInD(const Eigen::MatrixBase<VecIn> &x,
                             const Eigen::MatrixBase<VecOut> &x_out) const;

  template <typename MatOut>
  void computeRotMat2D(const T theta,
                       const Eigen::MatrixBase<MatOut> &R_out) const;

  // template <typename VecIn>
  // void computeTangentPyramid(const Eigen::MatrixBase<VecIn> &x) {}

  // template <typename VecIn>
  // void computeInscribedPyramid(const Eigen::MatrixBase<VecIn> &x) {}

  // const double &getTangentPyramidFriction() const { return mu_; }

  // const double &getInscribedPyramidFriction() const { return mu_; }

  // const Matrix36s &getLCPTangentBasisMatrix() const { return UD_lcp_; }

  // const Matrix36s &getLCPInscribedBasisMatrix() const { return UD_lcp_; }

  bool operator==(const PyramidCone &rhs) const { return mu_ == rhs.mu_; }

  template <typename NewScalar> PyramidCone<NewScalar> cast() const {
    PyramidCone<NewScalar> newCone;

    newCone.n_cons_ = n_cons_;
    newCone.mu_ = static_cast<NewScalar>(mu_);
    newCone.n_facets_ = n_facets_;
    newCone.Uxy_ = Uxy_.template cast<NewScalar>();
    newCone.e_ = e_.template cast<NewScalar>();
    newCone.D_ = D_.template cast<NewScalar>();
    newCone.D_lcp_ = D_lcp_.template cast<NewScalar>();
    newCone.UD_ = UD_.template cast<NewScalar>();
    newCone.Uxy_ = Uxy_.template cast<NewScalar>();
    newCone.UD_lcp_ = UD_lcp_.template cast<NewScalar>();

    return newCone;
  }

protected:
  Matrix2s D_red_inv_;
};

template <typename T> struct IceCreamCone : FrictionConstraint<T> {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);

  int n_cons_;
  T mu_;
  // PyramidCone<T> tangentPyramid_;
  // PyramidCone<T> inscribedPyramid_;
  // Vector4s e_;

  IceCreamCone() { n_cons_ = 3; };
  IceCreamCone(T mu) {
    mu_ = mu;
    n_cons_ = 3;
    // tangentPyramid_ = PyramidCone<T>(mu);
    // inscribedPyramid_ = PyramidCone<T>(mu / std::sqrt(2.));
    // e_ = tangentPyramid_.e_;
  }

  template <typename VecIn, typename VecOut>
  static void project(const T &mu, const Eigen::MatrixBase<VecIn> &x,
                      const Eigen::MatrixBase<VecOut> &x_out);

  template <typename VecIn, typename VecOut>
  void project(const Eigen::MatrixBase<VecIn> &x,
               const Eigen::MatrixBase<VecOut> &x_out) const;

  template <typename VecIn, typename VecOut>
  void projectDual(const Eigen::MatrixBase<VecIn> &x,
                   const Eigen::MatrixBase<VecOut> &x_out) const;

  template <typename VecIn>
  static bool isInside(const T &mu, const Eigen::MatrixBase<VecIn> &x,
                       T thresh = 1e-6);

  template <typename VecIn>
  bool isInside(const Eigen::MatrixBase<VecIn> &x, T thresh = 1e-3) const;

  template <typename VecIn>
  bool isInsideDual(const Eigen::MatrixBase<VecIn> &x, T thresh = 1e-3) const;

  template <typename VecIn>
  static bool isOnBorder(const T &mu, const Eigen::MatrixBase<VecIn> &x,
                         T thresh = 1e-3);

  template <typename VecIn>
  bool isOnBorder(const Eigen::MatrixBase<VecIn> &x, T thresh = 1e-3) const;

  template <typename VecIn>
  bool isOnBorderDual(const Eigen::MatrixBase<VecIn> &x, T thresh = 1e-3) const;

  template <typename VecIn, typename VecOut>
  static void projectHorizontal(const T &mu, const Eigen::MatrixBase<VecIn> &x,
                                const Eigen::MatrixBase<VecOut> &x_out_);

  template <typename VecIn, typename VecOut>
  void projectHorizontal(const Eigen::MatrixBase<VecIn> &x,
                         const Eigen::MatrixBase<VecOut> &x_out_) const;

  template <typename VecIn, typename VecOut>
  void projectHorizontal(const Eigen::MatrixBase<VecIn> &x, double x_n,
                         const Eigen::MatrixBase<VecOut> &x_out_) const;

  template <typename VecIn, typename VecIn2>
  T computeConicComplementarity(const Eigen::MatrixBase<VecIn> &lam,
                                const Eigen::MatrixBase<VecIn2> &v) const;

  template <typename VecIn, typename VecIn2>
  T computeSignoriniComplementarity(const Eigen::MatrixBase<VecIn> &lam,
                                    const Eigen::MatrixBase<VecIn2> &v) const;

  template <typename VecIn, typename VecOut>
  void computeDeSaxceCorrection(const Eigen::MatrixBase<VecIn> &v,
                                const Eigen::MatrixBase<VecOut> &v_out_) const;

  template <typename VecIn, typename VecIn2>
  T computeContactComplementarity(const Eigen::MatrixBase<VecIn> &lam,
                                  const Eigen::MatrixBase<VecIn2> &v) const;

  bool operator==(const IceCreamCone &rhs) const { return mu_ == rhs.mu_; }

  // template <typename VecIn>
  // void computeTangentPyramid(const Eigen::MatrixBase<VecIn> &lam);

  // template <typename VecIn>
  // void computeInscribedPyramid(const Eigen::MatrixBase<VecIn> &lam);

  // template <typename VecIn, typename VecOut>
  // void computeVelInTangentBasis(const Eigen::MatrixBase<VecIn> &v,
  //                               const Eigen::MatrixBase<VecOut> &v_out)
  //                               const;

  // template <typename VecIn, typename VecOut>
  // void computeVelInInscribedBasis(const Eigen::MatrixBase<VecIn> &v,
  //                                 const Eigen::MatrixBase<VecOut> &v_out)
  //                                 const;

  // const double &getTangentPyramidFriction() const {
  //   return tangentPyramid_.mu_;
  // }

  // const double &getInscribedPyramidFriction() const {
  //   return inscribedPyramid_.mu_;
  // }

  // const Matrix2s &getTangentBasisMatrix() const;

  // const Matrix2s &getInscribedBasisMatrix() const;

  // const Matrix36s &getLCPTangentBasisMatrix() const;

  // const Matrix36s &getLCPInscribedBasisMatrix() const;

  template <typename NewScalar> IceCreamCone<NewScalar> cast() const {
    IceCreamCone<NewScalar> newCone;

    newCone.n_cons_ = n_cons_;
    newCone.mu_ = static_cast<NewScalar>(mu_);

    return newCone;
  }

  // protected:
  //   Matrix2s D_tan_;
  //   Matrix2s D_ins_;
};

template <typename T> struct Box : FrictionConstraint<T> {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);

  int n_cons_;

  VectorXs mus_;

  Box(){};
  Box(VectorXs mus) {
    mus_ = mus;
    n_cons_ = mus_.size();
  }

  template <typename VecIn, typename VecOut>
  void project(const Eigen::MatrixBase<VecIn> &x,
               const Eigen::MatrixBase<VecOut> &x_out) const;

  template <typename VecIn>
  bool isInside(const Eigen::MatrixBase<VecIn> &x, T thresh = 1e-3) const;

  bool operator==(const Box &rhs) { return mus_ == rhs.mus_; }

  template <typename NewScalar> Box<NewScalar> cast() const {
    Box<NewScalar> newBox;

    newBox.n_cons_ = n_cons_;
    newBox.mus_ = mus_.template cast<NewScalar>();

    return newBox;
  }
};

} // end namespace contactbench

#include "contactbench/friction-constraint.hxx"

#endif
