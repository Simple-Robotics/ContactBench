#ifndef CONTACT_BENCH_MATH_H
#define CONTACT_BENCH_MATH_H

#include <Eigen/Core>

#define CONTACTBENCH_EIGEN_TYPEDEFS_WITH_OPTIONS(Scalar, Options)              \
  using MatrixXs = Eigen::Matrix<Scalar, -1, -1, Options>;                     \
  using RowMatrixXs = Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>;          \
  using Matrix2s = Eigen::Matrix<Scalar, 2, 2, Options>;                       \
                                                                               \
  using Matrix3s = Eigen::Matrix<Scalar, 3, 3, Options>;                       \
  using Matrix4s = Eigen::Matrix<Scalar, 4, 4, Options>;                       \
  using Matrix24s = Eigen::Matrix<Scalar, 2, 4, Options>;                      \
  using Matrix34s = Eigen::Matrix<Scalar, 3, 4, Options>;                      \
  using Matrix36s = Eigen::Matrix<Scalar, 3, 6, Options>;                      \
  template <int N> using Vector = Eigen::Matrix<Scalar, N, 1, Options>;        \
  using Vector2s = Vector<2>;                                                  \
  using Vector3s = Vector<3>;                                                  \
  using Vector4s = Vector<4>;                                                  \
  using VectorXs = Vector<Eigen::Dynamic>;                                     \
  using MatrixRef = Eigen::Ref<MatrixXs>;                                      \
  using VectorRef = Eigen::Ref<VectorXs>

#define CONTACTBENCH_EIGEN_TYPEDEFS(Scalar)                                    \
  CONTACTBENCH_EIGEN_TYPEDEFS_WITH_OPTIONS(Scalar, Eigen::ColMajor)

namespace contactbench {

template <typename Scalar, int Options = Eigen::ColMajor> struct math_types {
  CONTACTBENCH_EIGEN_TYPEDEFS_WITH_OPTIONS(Scalar, Options);
};

} // namespace contactbench

#endif
