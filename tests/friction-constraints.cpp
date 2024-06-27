// #define BOOST_TEST_MODULE FRICTION_CONSTRAINTS
#include "contactbench/friction-constraint.hpp"
#include "vector"

#include <iostream>
#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>

BOOST_AUTO_TEST_SUITE(FRICTION_CONSTRAINTS)

namespace cb = contactbench;
using T = double;
CONTACTBENCH_EIGEN_TYPEDEFS(T);

BOOST_AUTO_TEST_CASE(ice_cream_cone_init) {
  cb::IceCreamCone<T> cone(.84);
  BOOST_CHECK(!std::isnan(cone.mu_));
}

// BOOST_AUTO_TEST_CASE(ice_cream_projection) {
//   cb::IceCreamCone<T> cone(.84);
//   Eigen::Vector3d x = Eigen::Vector3d::Zero();
//   Eigen::Vector3d x_out;
//   cone.project(x, x_out);
//   BOOST_CHECK(x_out.isApprox(x));
//   x(2) = 1.;
//   x(0) = 1.08;
//   cone.project(x, x_out);
//   BOOST_CHECK(std::abs(x_out(0) - (x_out(2) * cone.mu_)) < 1e-3);
//   for (int i = 0; i < 10; i++) {
//     x = Eigen::Vector3d::Random();
//     cone = cb::IceCreamCone<T>(1. / (1 + std::rand()));
//     Eigen::Vector3d x_out2;
//     cone.project(x, x_out);
//     cone.project(x_out, x_out2);
//     BOOST_CHECK(x_out.isApprox(x_out2));
//     BOOST_CHECK(std::abs(x_out.dot(x_out - x)) < 1e-6);
//   }
// }

// BOOST_AUTO_TEST_CASE(ice_cream_dual_projection) {
//   cb::IceCreamCone<T> cone(.84);
//   Eigen::Vector3d x = Eigen::Vector3d::Zero();
//   Eigen::Vector3d x_out;
//   cone.projectDual(x, x_out);
//   BOOST_CHECK(x_out.isApprox(x));
//   x(2) = 1.;
//   x(0) = 1e3;
//   cone.projectDual(x, x_out);
//   BOOST_CHECK(std::abs(x_out(0) - (x_out(2) * 1. / cone.mu_)) < 1e-3);
//   for (int i = 0; i < 10; i++) {
//     x = Eigen::Vector3d::Random();
//     cone = cb::IceCreamCone<T>(1. / (1 + std::rand()));
//     Eigen::Vector3d x_out2;
//     cone.projectDual(x, x_out);
//     cone.projectDual(x_out, x_out2);
//     BOOST_CHECK(x_out.isApprox(x_out2));
//     BOOST_CHECK(std::abs(x_out.dot(x_out - x)) < 1e-6);
//   }
// }

// BOOST_AUTO_TEST_CASE(ice_cream_circle_projection) {
//   cb::IceCreamCone<T> cone(.37);
//   Eigen::Vector3d x = Eigen::Vector3d::Zero();
//   Eigen::Vector3d x_out;
//   cone.projectHorizontal(x, x_out);
//   BOOST_CHECK(x_out.isApprox(x));
//   x(2) = 1.59;
//   x(0) = 3.46;
//   cone.projectHorizontal(x, x_out);
//   BOOST_CHECK(x_out(0) == x(2) * cone.mu_);
//   for (int i = 0; i < 10; i++) {
//     x = Eigen::Vector3d::Random();
//     x[2] = std::abs(x[2]);
//     cone = cb::IceCreamCone<T>(1. / (1 + std::rand()));
//     Eigen::Vector3d x_out2;
//     cone.projectHorizontal(x, x_out);
//     cone.projectHorizontal(x_out, x_out2);
//     BOOST_CHECK(x_out.isApprox(x_out2));
//     x_out2[0] = 0.;
//     x_out2[1] = 0.;
//     BOOST_CHECK(std::abs(x_out2.dot(x_out - x)) < 1e-6);
//   }
// }

// BOOST_AUTO_TEST_CASE(ice_cream_circle_projection2) {
//   cb::IceCreamCone<T> cone(.37);
//   Eigen::Vector2d x = Eigen::Vector2d::Zero();
//   double xn = 0.;
//   Eigen::Vector2d x_out;
//   cone.projectHorizontal(x, xn, x_out);
//   BOOST_CHECK(x_out.isApprox(x));
//   xn = 1.59;
//   x(0) = 3.46;
//   cone.projectHorizontal(x, xn, x_out);
//   BOOST_CHECK(x_out(0) == xn * cone.mu_);
//   for (int i = 0; i < 10; i++) {
//     x = Eigen::Vector2d::Random();
//     xn = (1. / (1. + std::rand())) * 10.;
//     cone = cb::IceCreamCone<T>(1. / (1 + std::rand()));
//     Eigen::Vector2d x_out2;
//     cone.projectHorizontal(x, xn, x_out);
//     cone.projectHorizontal(x_out, xn, x_out2);
//     BOOST_CHECK(x_out.isApprox(x_out2));
//   }
// }

// BOOST_AUTO_TEST_CASE(ice_cream_circle_contact_complementarity) {
//   cb::IceCreamCone<T> cone(.37);
//   Eigen::Vector3d x = Eigen::Vector3d::Zero();
//   Eigen::Vector3d v = Eigen::Vector3d::Zero();
//   T comp = cone.computeContactComplementarity(x, v);
//   BOOST_CHECK(comp == 0.);
// }

// BOOST_AUTO_TEST_CASE(ice_cream_circle_signorini_complementarity) {
//   cb::IceCreamCone<T> cone(.37);
//   Eigen::Vector3d x = Eigen::Vector3d::Zero();
//   Eigen::Vector3d v = Eigen::Vector3d::Zero();
//   T comp = cone.computeSignoriniComplementarity(x, v);
//   BOOST_CHECK(comp == 0.);
// }

// BOOST_AUTO_TEST_CASE(ice_cream_circle_conic_complementarity) {
//   MatrixXs A = MatrixXs::Zero(3, 3);
//   VectorXs b = VectorXs::Zero(3);
//   cb::IceCreamCone<T> cone(.37);
//   Eigen::Vector3d x = Eigen::Vector3d::Zero();
//   Eigen::Vector3d v = Eigen::Vector3d::Zero();
//   T comp = cone.computeConicComplementarity(x, v);
//   BOOST_CHECK(comp == 0.);
// }

// BOOST_AUTO_TEST_CASE(pyramid_cone_init) {
//   cb::PyramidCone<T> cone(.84);
//   BOOST_CHECK(!std::isnan(cone.mu_));
// }

// BOOST_AUTO_TEST_CASE(pyramid_cone_horizontal_projection) {
//   cb::PyramidCone<T> cone(.37);
//   Eigen::Vector3d x = Eigen::Vector3d::Zero();
//   Eigen::Vector3d x_out;
//   cone.projectHorizontal(x, x_out);
//   BOOST_CHECK(x_out.isApprox(x));
//   x(2) = 1.59;
//   x(0) = 3.46;
//   cone.projectHorizontal(x, x_out);
//   BOOST_CHECK(x_out(0) == x(2) * cone.mu_);
// }

// BOOST_AUTO_TEST_CASE(pyramid_cone_horizontal_projection2) {
//   cb::PyramidCone<T> cone(.37);
//   Eigen::Vector2d x = Eigen::Vector2d::Zero();
//   double xn = 0.;
//   Eigen::Vector2d x_out;
//   cone.projectHorizontal(x, xn, x_out);
//   BOOST_CHECK(x_out.isApprox(x));
//   xn = 1.59;
//   x(0) = 3.46;
//   cone.projectHorizontal(x, xn, x_out);
//   BOOST_CHECK(x_out(0) == xn * cone.mu_);
// }

// BOOST_AUTO_TEST_CASE(pyramid_computeCoordinatesInD) {
//   double mu = .37;
//   cb::PyramidCone<T> cone(mu);
//   Eigen::Vector3d x = Eigen::Vector3d::Zero();
//   Eigen::Vector4d x_out;
//   cone.computeCoordinatesInD(x, x_out);
//   BOOST_CHECK(x_out.isZero(0.));
//   x(2) = 1.59;
//   x(0) = mu * x(2);
//   cone.computeCoordinatesInD(x, x_out);
//   BOOST_CHECK(x_out(1) == 0.);
//   BOOST_CHECK(x_out(2) == 0.);
//   BOOST_CHECK(std::abs(x_out(0) - x_out(3)) < 1e-5);
// }

BOOST_AUTO_TEST_SUITE_END()
