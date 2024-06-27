#ifndef CONTACT_BENCH_DELASSUS_H
#define CONTACT_BENCH_DELASSUS_H

#include <Eigen/Core>
#include "contactbench/fwd.hpp"
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/contact-cholesky.hpp>
#include <pinocchio/algorithm/contact-info.hpp>
#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/dense/helpers.hpp>

namespace contactbench {

template <typename T> struct DelassusBase {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);

  int nc_;

  MatrixXs G_;

  VectorXs R_reg_;

  DelassusBase(){};
  DelassusBase(MatrixXs G) {
    G_ = G;
    nc_ = (int)G_.cols() / 3;
  };
  virtual ~DelassusBase() = default;

  virtual void evaluateDel() = 0;

  virtual void evaluateDiagDel() = 0;

  virtual void applyOnTheRight(const Eigen::Ref<const VectorXs> &x,
                               Eigen::Ref<VectorXs> x_out_) const = 0;

  virtual void applyPerContactOnTheRight(int i,
                                         const Eigen::Ref<const VectorXs> &x,
                                         Eigen::Ref<VectorXs> x_out_) const = 0;

  virtual void
  applyPerContactNormalOnTheRight(int i, const Eigen::Ref<const VectorXs> &x,
                                  T &x_out_) const = 0;

  virtual void
  applyPerContactTangentOnTheRight(int i, const Eigen::Ref<const VectorXs> &x,
                                   Eigen::Ref<VectorXs> x_out_) const = 0;

  // virtual T computeLargestEigenValue(const int max_it = 10,
  //                                    const T rel_tol = 1e-8) = 0;

  virtual void computeChol(T mu) = 0;

  virtual void computeChol(const Eigen::Ref<const VectorXs> &mus) = 0;

  virtual void updateChol(T mu) = 0;

  virtual void updateChol(const Eigen::Ref<const VectorXs> &mus) = 0;

  virtual void solve(const Eigen::Ref<const VectorXs> &x,
                     Eigen::Ref<VectorXs> x_out_) const = 0;

  virtual void solveInPlace(Eigen::Ref<VectorXs> x) const = 0;

  virtual bool isDense() const = 0;
};

template <typename T> struct DelassusDense : DelassusBase<T> {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);
  using Base = DelassusBase<T>;
  using Base::G_;
  using Base::nc_;
  using Base::R_reg_;

  // int nc_;

  // MatrixXs G_;

  // T mu_;

  MatrixXs G_llt_;
  Eigen::LLT<MatrixXs> llt_;

  DelassusDense(){};

  DelassusDense(const Eigen::Ref<const MatrixXs> &G);

  void evaluateDel();

  void evaluateDiagDel();

  void applyOnTheRight(const Eigen::Ref<const VectorXs> &x,
                       Eigen::Ref<VectorXs> x_out) const;

  void applyPerContactOnTheRight(int i, const Eigen::Ref<const VectorXs> &x,
                                 Eigen::Ref<VectorXs> x_out_) const;

  void applyPerContactNormalOnTheRight(int i,
                                       const Eigen::Ref<const VectorXs> &x,
                                       T &x_out_) const;

  void applyPerContactTangentOnTheRight(int i,
                                        const Eigen::Ref<const VectorXs> &x,
                                        Eigen::Ref<VectorXs> x_out_) const;

  // T computeLargestEigenValue(const int max_it = 10, const T tol = 1e-8);

  void computeChol(T mu);

  void computeChol(const Eigen::Ref<const VectorXs> &mus);

  void updateChol(T mu);

  void updateChol(const Eigen::Ref<const VectorXs> &mus);

  void solve(const Eigen::Ref<const VectorXs> &x,
             Eigen::Ref<VectorXs> x_out) const;

  void solveInPlace(Eigen::Ref<VectorXs> x) const;

  virtual bool isDense() const{
    return true;
  }

protected:
  VectorXs vpow_, Gvpow_, err_vpow_;
};

namespace {
namespace pin = pinocchio;
}

template <typename T> struct DelassusPinocchio : DelassusBase<T> {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  static constexpr int Options = Eigen::ColMajor;
  using Model = pin::ModelTpl<T>;
  using Data = pin::DataTpl<T>;
  using RigidConstraintModel = pin::RigidConstraintModelTpl<T, Options>;
  using RigidConstraintData = pin::RigidConstraintDataTpl<T, Options>;
  using RigidConstraintModelVector =
      CONTACTBENCH_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintModel);
  using RigidConstraintDataVector =
      CONTACTBENCH_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintData);
  using ContactCholeskyDecomposition =
      pin::ContactCholeskyDecompositionTpl<T, Options>;
  // pin::cholesky::ContactCholeskyDecompositionTpl<T, Options>;

  Model model_;
  Data data_;
  RigidConstraintModelVector contact_models_;
  RigidConstraintDataVector contact_datas_;
  ContactCholeskyDecomposition contact_chol_;

  // int nc_;

  // MatrixXs G_;

  // T mu_;
  using Base = DelassusBase<T>;
  using Base::G_;
  using Base::nc_;
  using Base::R_reg_;
  DelassusPinocchio(){};

  DelassusPinocchio(const Model &model, const Data &data,
                    const RigidConstraintModelVector &contact_models,
                    const RigidConstraintDataVector &contact_datas);

  void evaluateDel();

  void evaluateDiagDel();

  // template <typename VecIn, typename VecOut>
  void applyOnTheRight(const Eigen::Ref<const VectorXs> &x,
                       Eigen::Ref<VectorXs> x_out) const;

  void applyPerContactOnTheRight(int i, const Eigen::Ref<const VectorXs> &x,
                                 Eigen::Ref<VectorXs> x_out_) const;

  void applyPerContactNormalOnTheRight(int i,
                                       const Eigen::Ref<const VectorXs> &x,
                                       T &x_out_) const;

  void applyPerContactTangentOnTheRight(int i,
                                        const Eigen::Ref<const VectorXs> &x,
                                        Eigen::Ref<VectorXs> x_out_) const;

  // T computeLargestEigenValue(const int max_it = 10, const T tol = 1e-8);

  void computeChol(T mu);

  void computeChol(const Eigen::Ref<const VectorXs> &mus);

  void updateChol(T mu);

  void updateChol(const Eigen::Ref<const VectorXs> &mus);

  // template <typename VecIn, typename VecOut>
  void solve(const Eigen::Ref<const VectorXs> &x,
             Eigen::Ref<VectorXs> x_out) const;

  void solveInPlace(Eigen::Ref<VectorXs> x) const;

  virtual bool isDense() const{
    return false;
  }
};


template<typename _Vector>
  struct PowerIterationWrapper: pinocchio::PowerIterationAlgoTpl<_Vector> 
  {
    typedef typename PINOCCHIO_EIGEN_PLAIN_TYPE(_Vector) Vector;
    typedef typename Vector::Scalar Scalar;
    typedef pinocchio::PowerIterationAlgoTpl<_Vector> Base;

    explicit PowerIterationWrapper(const Eigen::DenseIndex size,
                                   const int max_it = 10,
                                   const Scalar rel_tol = Scalar(1e-8))
    : Base(size, max_it, rel_tol)
    {
    }

    void run(const DelassusBase<Scalar>* delassus)
    {
      if(delassus->isDense())
      {
        run(dynamic_cast<const DelassusDense<Scalar>&>(*delassus));
      }
      else
      {
        run(dynamic_cast<const DelassusPinocchio<Scalar>&>(*delassus));
      }
    }

    void run(const DelassusPinocchio<Scalar>& delassus)
    { 
      Base::run(delassus.contact_chol_.getDelassusCholeskyExpression());
    }

    void run(const DelassusDense<Scalar>& delassus)
    { 
      Base::run(delassus.G_);
    }

  }; // struct PowerIterationAlgoTpl

} // end namespace contactbench

#include "contactbench/delassus-wrapper.hxx"

#endif

#if CONTACTBENCH_ENABLE_TEMPLATE_INSTANTIATION
#include "contactbench/delassus-wrapper.txx"
#endif // CONTACTBENCH_ENABLE_TEMPLATE_INSTANTIATION
