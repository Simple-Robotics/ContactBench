// #define BOOST_TEST_MODULE FRICTION_CONSTRAINTS
#include "contactbench/friction-constraint.hpp"
#include "vector"

#include <iostream>
#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>
#include <pinocchio/autodiff/cppad.hpp>

BOOST_AUTO_TEST_SUITE(FRICTION_CONSTRAINTS_CPPAD)

namespace cb = contactbench;
using CppAD::AD;
using CppAD::NearEqual;

typedef double Scalar;
typedef AD<Scalar> ADScalar;
using T = ADScalar;
CONTACTBENCH_EIGEN_TYPEDEFS(ADScalar);

BOOST_AUTO_TEST_CASE(ice_cream_cone_init_cppad) {
  cb::IceCreamCone<T> cone(.84);
  BOOST_CHECK(!CppAD::isnan(cone.mu_));
}

BOOST_AUTO_TEST_CASE(pyramid_cone_cast) {
  cb::PyramidCone<Scalar> originalCone(.84);
  cb::PyramidCone<ADScalar> castedCone = originalCone.cast<ADScalar>();
  BOOST_CHECK(!CppAD::isnan(castedCone.mu_));
}

BOOST_AUTO_TEST_CASE(ice_cream_cone_cast) {
  cb::IceCreamCone<Scalar> originalCone(.84);
  cb::IceCreamCone<ADScalar> castedCone = originalCone.cast<ADScalar>();
  BOOST_CHECK(!CppAD::isnan(castedCone.mu_));
}

BOOST_AUTO_TEST_CASE(ice_cream_projection_cppad) {
  cb::IceCreamCone<T> cone(.84);
  Vector3s x = Vector3s::Zero();
  Vector3s x_out;
  cone.project(x, x_out);
  BOOST_CHECK(x_out.isApprox(x));
  x(2) = 1.;
  x(0) = 1.08;
  cone.project(x, x_out);
  BOOST_CHECK(fabs(x_out(0) - (x_out(2) * cone.mu_)) < 1e-3);
  for (int i = 0; i < 10; i++) {
    x = Vector3s::Random();
    cone = cb::IceCreamCone<T>(1. / (1 + std::rand()));
    Vector3s x_out2;
    cone.project(x, x_out);
    cone.project(x_out, x_out2);
    BOOST_CHECK(x_out.isApprox(x_out2));
    BOOST_CHECK(fabs(x_out.dot(x_out - x)) < 1e-6);
  }
}

BOOST_AUTO_TEST_CASE(ice_cream_dual_projection_cppad) {
  cb::IceCreamCone<T> cone(.84);
  Vector3s x = Vector3s::Zero();
  Vector3s x_out;
  cone.projectDual(x, x_out);
  BOOST_CHECK(x_out.isApprox(x));
  x(2) = 1.;
  x(0) = 1e3;
  cone.projectDual(x, x_out);
  BOOST_CHECK(fabs(x_out(0) - (x_out(2) * 1. / cone.mu_)) < 1e-3);
  for (int i = 0; i < 10; i++) {
    x = Vector3s::Random();
    cone = cb::IceCreamCone<T>(1. / (1 + std::rand()));
    Vector3s x_out2;
    cone.projectDual(x, x_out);
    cone.projectDual(x_out, x_out2);
    BOOST_CHECK(x_out.isApprox(x_out2));
    BOOST_CHECK(fabs(x_out.dot(x_out - x)) < 1e-6);
  }
}

BOOST_AUTO_TEST_CASE(ice_cream_circle_projection_cppad) {
  cb::IceCreamCone<T> cone(.37);
  Vector3s x = Vector3s::Zero();
  Vector3s x_out;
  cone.projectHorizontal(x, x_out);
  BOOST_CHECK(x_out.isApprox(x));
  x(2) = 1.59;
  x(0) = 3.46;
  cone.projectHorizontal(x, x_out);
  BOOST_CHECK(x_out(0) == x(2) * cone.mu_);
  for (int i = 0; i < 10; i++) {
    x = Vector3s::Random();
    x[2] = fabs(x[2]);
    cone = cb::IceCreamCone<T>(1. / (1 + std::rand()));
    Vector3s x_out2;
    cone.projectHorizontal(x, x_out);
    cone.projectHorizontal(x_out, x_out2);
    BOOST_CHECK(x_out.isApprox(x_out2));
    x_out2[0] = 0.;
    x_out2[1] = 0.;
    BOOST_CHECK(fabs(x_out2.dot(x_out - x)) < 1e-6);
  }
}

BOOST_AUTO_TEST_CASE(ice_cream_circle_projection_backward_xout_wrt_x_cppad) {
  cb::IceCreamCone<T> cone(.37);
  VectorXs x = VectorXs::Zero(3);
  VectorXs x_out = VectorXs::Zero(3);
  x(2) = 1.59;
  x(1) = 0.2;
  x(0) = 3.5;
  CppAD::Independent(x);
  cone.projectHorizontal(x, x_out);
  CppAD::ADFun<Scalar> f(x, x_out);
  CPPAD_TESTVECTOR(Scalar) x_(static_cast<size_t>(x.size()));
  for (size_t i = 0; i < x.size(); ++i) {
    x_[i] = CppAD::Value(x[i]);
  }

  // CPPAD_TESTVECTOR(Scalar) y = f.Forward(0, x_);  // not needed
  CPPAD_TESTVECTOR(Scalar) J = f.Jacobian(x_);

  // compare with finite differences using x_out, could also use y
  double eps = 1e-6;
  Vector3s x_out2;
  Vector3s x2 = x;
  for (int i = 0; i < 3; ++i) {
    x2(i) += eps;
    cone.projectHorizontal(x2, x_out2);
    BOOST_CHECK_CLOSE(x_out2(0), (x_out[0] + eps * J[0]), 1e-3);
    BOOST_CHECK_CLOSE(x_out2(1), (x_out[1] + eps * J[1]), 1e-3);
    BOOST_CHECK_CLOSE(x_out2(2), (x_out[2] + eps * J[2]), 1e-3);
    x2(i) = x(i);
  }

  // check at different point in the same if branch of projectHorizontal
  x_[0] = -x_[0];
  x(0) = -x(0);
  CPPAD_TESTVECTOR(Scalar) y = f.Forward(0, x_); // for FD
  J = f.Jacobian(x_);
  x2 = x;
  for (int i = 0; i < 3; ++i) {
    x2(i) += eps;
    cone.projectHorizontal(x2, x_out2);
    BOOST_CHECK_CLOSE(x_out2(0), (y[0] + eps * J[0]), 1e-3);
    BOOST_CHECK_CLOSE(x_out2(1), (y[1] + eps * J[1]), 1e-3);
    BOOST_CHECK_CLOSE(x_out2(2), (y[2] + eps * J[2]), 1e-3);
    x2(i) = x(i);
  }

  // check at different point in the else statement of projectHorizontal
  // NOTE: This here is failing with our current implementation of
  // projectHorizontal. We can fix it by using the static if_then_else
  // from pinocchio (using CppAD::CondExpOp).
  // x_[0] = 0.3;
  // x(0) = 0.3;
  // cone.projectHorizontal(x, x_out);
  // y = f.Forward(0, x_);
  // J = f.Jacobian(x_);

  // // compare with finite differences
  // x2 = x;
  // for (int i = 0; i < 3; ++i) {
  //   x2(i) += eps;
  //   cone.projectHorizontal(x2, x_out2);
  //   std::cout << "x2: " << x2.transpose() << std::endl;
  //   BOOST_CHECK_CLOSE(x_out2(0), (y[0] + eps * J[0]), 1e-3);
  //   BOOST_CHECK_CLOSE(x_out2(1), (y[1] + eps * J[1]), 1e-3);
  //   BOOST_CHECK_CLOSE(x_out2(2), (y[2] + eps * J[2]), 1e-3);
  //   x2(i) = x(i);
  // }
}

BOOST_AUTO_TEST_CASE(ice_cream_circle_projection2_cppad) {
  cb::IceCreamCone<T> cone(.37);
  Vector2s x = Vector2s::Zero();
  double xn = 0.;
  Vector2s x_out;
  cone.projectHorizontal(x, xn, x_out);
  BOOST_CHECK(x_out.isApprox(x));
  xn = 1.59;
  x(0) = 3.46;
  cone.projectHorizontal(x, xn, x_out);
  BOOST_CHECK(x_out(0) == xn * cone.mu_);
  for (int i = 0; i < 10; i++) {
    x = Vector2s::Random();
    xn = (1. / (1. + std::rand())) * 10.;
    cone = cb::IceCreamCone<T>(1. / (1 + std::rand()));
    Vector2s x_out2;
    cone.projectHorizontal(x, xn, x_out);
    cone.projectHorizontal(x_out, xn, x_out2);
    BOOST_CHECK(x_out.isApprox(x_out2));
  }
}

BOOST_AUTO_TEST_CASE(ice_cream_circle_projection2_backward_cppad) {
  cb::IceCreamCone<T> cone(.37);
  VectorXs x = VectorXs::Zero(2);
  CppAD::Independent(x);
  double xn = 0.;
  VectorXs x_out = VectorXs::Zero(2);
  cone.projectHorizontal(x, xn, x_out);
  BOOST_CHECK(x_out.isApprox(x));

  CppAD::ADFun<Scalar> f(x, x_out);
  CPPAD_TESTVECTOR(Scalar) x_(2);
  x_[0] = 0;
  x_[1] = 0;

  // Perform Forward and Jacobian calculations using CPPAD_TESTVECTOR
  CPPAD_TESTVECTOR(Scalar) y = f.Forward(0, x_);
  CPPAD_TESTVECTOR(Scalar) J = f.Jacobian(x_);
}

BOOST_AUTO_TEST_CASE(ice_cream_circle_contact_complementarity_cppad) {
  cb::IceCreamCone<T> cone(.37);
  Vector3s x = Vector3s::Zero();
  Vector3s v = Vector3s::Zero();
  T comp = cone.computeContactComplementarity(x, v);
  BOOST_CHECK(comp == 0.);
}

BOOST_AUTO_TEST_CASE(ice_cream_circle_signorini_complementarity_cppad) {
  cb::IceCreamCone<T> cone(.37);
  Vector3s x = Vector3s::Zero();
  Vector3s v = Vector3s::Zero();
  T comp = cone.computeSignoriniComplementarity(x, v);
  BOOST_CHECK(comp == 0.);
}

BOOST_AUTO_TEST_CASE(ice_cream_circle_conic_complementarity_cppad) {
  MatrixXs A = MatrixXs::Zero(3, 3);
  VectorXs b = VectorXs::Zero(3);
  cb::IceCreamCone<T> cone(.37);
  Vector3s x = Vector3s::Zero();
  Vector3s v = Vector3s::Zero();
  T comp = cone.computeConicComplementarity(x, v);
  BOOST_CHECK(comp == 0.);
}

BOOST_AUTO_TEST_CASE(pyramid_cone_init_cppad) {
  cb::PyramidCone<T> cone(.84);
  BOOST_CHECK(!CppAD::isnan(cone.mu_));
}

BOOST_AUTO_TEST_CASE(pyramid_cone_horizontal_projection_cppad) {
  cb::PyramidCone<T> cone(.37);
  Vector3s x = Vector3s::Zero();
  Vector3s x_out;
  cone.projectHorizontal(x, x_out);
  BOOST_CHECK(x_out.isApprox(x));
  x(2) = 1.59;
  x(0) = 3.46;
  cone.projectHorizontal(x, x_out);
  BOOST_CHECK(x_out(0) == x(2) * cone.mu_);
}

BOOST_AUTO_TEST_CASE(pyramid_cone_horizontal_projection2_cppad) {
  cb::PyramidCone<T> cone(.37);
  Vector2s x = Vector2s::Zero();
  double xn = 0.;
  Vector2s x_out;
  cone.projectHorizontal(x, xn, x_out);
  BOOST_CHECK(x_out.isApprox(x));
  xn = 1.59;
  x(0) = 3.46;
  cone.projectHorizontal(x, xn, x_out);
  BOOST_CHECK(x_out(0) == xn * cone.mu_);
}

BOOST_AUTO_TEST_CASE(pyramid_computeCoordinatesInD_cppad) {
  double mu = .37;
  cb::PyramidCone<T> cone(mu);
  Vector3s x = Vector3s::Zero();
  Vector4s x_out;
  cone.computeCoordinatesInD(x, x_out);
  BOOST_CHECK(x_out.isZero(0.));
  x(2) = 1.59;
  x(0) = mu * x(2);
  cone.computeCoordinatesInD(x, x_out);
  BOOST_CHECK(x_out(1) == 0.);
  BOOST_CHECK(x_out(2) == 0.);
  BOOST_CHECK(fabs(x_out(0) - x_out(3)) < 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()
