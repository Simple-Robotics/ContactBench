#include "contactbench/contact-problem.hpp"
#include "contactbench/delassus-wrapper.hpp"
#include "contactbench/solvers.hpp"
#include "vector"

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>
#include <pinocchio/parsers/sample-models.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/contact-cholesky.hpp>
#include <pinocchio/algorithm/contact-info.hpp>
#include <pinocchio/autodiff/cppad.hpp>

BOOST_AUTO_TEST_SUITE(PROBLEMS_CPPAD)

namespace cb = contactbench;
using isize = Eigen::Index;
using usize = std::make_unsigned<isize>::type;
using CppAD::AD;
// using CppAD::NearEqual;

typedef double Scalar;
typedef AD<Scalar> ADScalar;
using T = ADScalar;
CONTACTBENCH_EIGEN_TYPEDEFS(ADScalar);
static constexpr int Options = Eigen::ColMajor;

typedef pinocchio::ModelTpl<Scalar> Model;
typedef Model::Data Data;
typedef pinocchio::ModelTpl<T> ADModel;
typedef ADModel::Data ADData;

typedef pinocchio::RigidConstraintModelTpl<Scalar, Options>
    RigidConstraintModel;
typedef pinocchio::RigidConstraintModelTpl<ADScalar, Options>
    ADRigidConstraintModel;
typedef pinocchio::RigidConstraintDataTpl<Scalar, Options> RigidConstraintData;
typedef pinocchio::RigidConstraintDataTpl<ADScalar, Options>
    ADRigidConstraintData;

typedef Model::ConfigVectorType ConfigVectorType;
typedef ADModel::ConfigVectorType ADConfigVectorType;

typedef pinocchio::SE3Tpl<Scalar> SE3;

using RigidConstraintModelVector =
    CONTACTBENCH_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintModel);
using RigidConstraintDataVector =
    CONTACTBENCH_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintData);

using ADRigidConstraintModelVector =
    CONTACTBENCH_STD_VECTOR_WITH_EIGEN_ALLOCATOR(ADRigidConstraintModel);
using ADRigidConstraintDataVector =
    CONTACTBENCH_STD_VECTOR_WITH_EIGEN_ALLOCATOR(ADRigidConstraintData);

BOOST_AUTO_TEST_CASE(contact_problem_init_cppad) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(contact_problem_init_and_cast) {
  using MatrixXs = Eigen::Matrix<Scalar, -1, -1, Eigen::ColMajor>;
  using VectorXs = Eigen::Matrix<Scalar, -1, 1, Eigen::ColMajor>;

  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<Scalar> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<Scalar, cb::IceCreamCone> originalProblem(A, b, mus);
  cb::ContactProblem<T, cb::IceCreamCone> adProblem =
      originalProblem.cast<CppAD::AD<double>>();
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(contact_problem_init_with_delassus_cppad) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};
  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);
  ADModel ad_model = model.cast<ADScalar>();
  ADData ad_data(ad_model);
  RigidConstraintModelVector contact_models;
  ADRigidConstraintModelVector ad_contact_models;
  RigidConstraintDataVector contact_datas;
  ADRigidConstraintDataVector ad_contact_datas;
  for (int i = 0; i < nc; i++) {
    int id1 = 0;
    int id2 = 1;
    SE3 placement1 = SE3::Identity();
    SE3 placement2 = SE3::Identity();
    RigidConstraintModel contact_model_i(pinocchio::CONTACT_3D, model, id1,
                                         placement1, id2, placement2,
                                         pinocchio::LOCAL);
    RigidConstraintData contact_data_i(contact_model_i);
    contact_models.push_back(contact_model_i);
    contact_datas.push_back(contact_data_i);
  }
  for (int i = 0; i < nc; i++) {
    ADRigidConstraintModel ad_contact_model_i =
        contact_models[i].cast<ADScalar>();
    ADRigidConstraintData ad_contact_data_i(ad_contact_model_i);
    ad_contact_models.push_back(ad_contact_model_i);
    ad_contact_datas.push_back(ad_contact_data_i);
  }
  auto Del = std::make_shared<cb::DelassusPinocchio<T>>(
      ad_model, ad_data, ad_contact_models, ad_contact_datas);
  int nc_test = Del->nc_;
  cb::ContactProblem<T, cb::IceCreamCone> prob(Del, b, mus);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(contact_problem_contactComplementarity_cppad) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  T comp = prob.computeContactComplementarity(lam);
  BOOST_CHECK(!CppAD::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_contactComplementarity2_cppad) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  VectorXs v = VectorXs::Zero(3 * nc);
  T comp = prob.computeContactComplementarity(lam, v);
  BOOST_CHECK(!CppAD::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_signoriniComplementarity_cppad) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  T comp = prob.computeSignoriniComplementarity(lam);
  BOOST_CHECK(!CppAD::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_signoriniComplementarity2_cppad) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  VectorXs v = VectorXs::Zero(3 * nc);
  T comp = prob.computeSignoriniComplementarity(lam, v);
  BOOST_CHECK(!CppAD::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_conicComplementarity_cppad) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  T comp = prob.computeConicComplementarity(lam);
  BOOST_CHECK(!CppAD::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_conicComplementarity2_cppad) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  VectorXs v = VectorXs::Zero(3 * nc);
  T comp = prob.computeConicComplementarity(lam, v);
  BOOST_CHECK(!CppAD::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_perContactContactComplementarity_cppad) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  VectorXs v = A * lam + b;
  isize i = 8;
  T comp = prob.computePerContactContactComplementarity(
      (usize)i, lam.segment<3>(3 * i), v.segment<3>(3 * i));
  BOOST_CHECK(!CppAD::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_perContactSignoriniComplementarity_cppad) {
  int nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  VectorXs v = A * lam + b;
  isize i = 8;
  T comp = prob.computePerContactSignoriniComplementarity(
      (usize)i, lam.segment<3>(3 * i), v.segment<3>(3 * i));
  BOOST_CHECK(!CppAD::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_perContactConicComplementarity_cppad) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  VectorXs v = A * lam + b;
  isize i = 8;
  T comp = prob.computePerContactConicComplementarity(
      (usize)i, v.segment<3>(3 * i), lam.segment<3>(3 * i));
  BOOST_CHECK(!CppAD::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_setLCP_cppad) {
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::PyramidCone> prob(A, b, mus);
  prob.setLCP();
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(contact_problem_getEquivalentLCP_backward_b_wrt_g_cppad) {
  int nc = 1;
  MatrixXs G = MatrixXs::Zero(3, 3);
  G(0, 0) = 3.5;
  G(1, 1) = 3.5;
  G(2, 2) = 1.;

  VectorXs g = VectorXs::Zero(3);
  g(2) = -0.00981;
  std::vector<T> mus = {0.9};

  CppAD::Independent(g);

  cb::ContactProblem<ADScalar, cb::PyramidCone> ad_prob(G, g, mus);
  ad_prob.setLCP();
  ad_prob.computeLCP();

  CppAD::ADFun<Scalar> ad_fun(g, ad_prob.b_);

  CPPAD_TESTVECTOR(Scalar) x(static_cast<size_t>(g.size()));
  for (size_t i = 0; i < g.size(); ++i) {
    x[i] = CppAD::Value(g[i]);
  }

  CPPAD_TESTVECTOR(Scalar) forward_result = ad_fun.Forward(0, x);
  CPPAD_TESTVECTOR(Scalar) jacobian_result = ad_fun.Jacobian(x);

  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(ad_prob.b_.size(), g.size());
  double eps = 1e-6;
  for (int i = 0; i < g.size(); i++) {

    VectorXs x_fd = g;
    VectorXs x_plus = x_fd;
    VectorXs x_minus = x_fd;
    x_plus(i) += eps;
    x_minus(i) -= eps;

    CPPAD_TESTVECTOR(Scalar) x_plus_ad(static_cast<int>(x_plus.size()));
    for (int i = 0; i < x_plus.size(); ++i) {
      x_plus_ad[i] = CppAD::Value(x_plus[i]);
    }
    CPPAD_TESTVECTOR(Scalar) x_minus_ad(static_cast<int>(x_minus.size()));
    for (int i = 0; i < x_minus.size(); ++i) {
      x_minus_ad[i] = CppAD::Value(x_minus[i]);
    }

    CPPAD_TESTVECTOR(Scalar) forward_result_plus = ad_fun.Forward(0, x_plus_ad);
    CPPAD_TESTVECTOR(Scalar)
    forward_result_minus = ad_fun.Forward(0, x_minus_ad);

    Eigen::VectorXd forward_plus_eigen(forward_result_plus.size());
    Eigen::VectorXd forward_minus_eigen(forward_result_minus.size());
    for (int i = 0; i < forward_result_plus.size(); i++) {
      forward_plus_eigen(i) = forward_result_plus[i];
      forward_minus_eigen(i) = forward_result_minus[i];
    }

    J.col(i) = (forward_plus_eigen - forward_minus_eigen) / (2 * eps);
  }
  Eigen::Map<Eigen::MatrixXd> jacobian_result_eigen(J.data(), J.rows(),
                                                    J.cols());
  BOOST_CHECK(J.isApprox(jacobian_result_eigen, 1e-3));
}

BOOST_AUTO_TEST_CASE(contact_problem_getEquivalentLCP_backward_A_wrt_g_cppad) {
  int nc = 1;
  MatrixXs G = MatrixXs::Zero(3, 3);
  G(0, 0) = 3.5;
  G(1, 1) = 3.5;
  G(2, 2) = 1.;

  VectorXs g = VectorXs::Zero(3);
  g(2) = -0.00981;
  std::vector<T> mus = {0.9};

  CppAD::Independent(g);
  cb::ContactProblem<ADScalar, cb::PyramidCone> ad_prob(G, g, mus);
  ad_prob.setLCP();
  ad_prob.computeLCP();

  VectorXs AD_A_flat =
      Eigen::Map<VectorXs>(ad_prob.A_.data(), ad_prob.A_.size());

  CppAD::ADFun<Scalar> ad_fun(g, AD_A_flat);

  CPPAD_TESTVECTOR(Scalar) x(static_cast<size_t>(g.size()));
  for (size_t i = 0; i < g.size(); ++i) {
    x[i] = CppAD::Value(g[i]);
  }

  CPPAD_TESTVECTOR(Scalar) forward_result = ad_fun.Forward(0, x);
  CPPAD_TESTVECTOR(Scalar) jacobian_result = ad_fun.Jacobian(x);
  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(ad_prob.A_.size(), g.size());
  double eps = 1e-6;
  for (int i = 0; i < g.size(); i++) {

    VectorXs x_fd = g;
    VectorXs x_plus = x_fd;
    VectorXs x_minus = x_fd;
    x_plus(i) += eps;
    x_minus(i) -= eps;

    CPPAD_TESTVECTOR(Scalar) x_plus_ad(static_cast<int>(x_plus.size()));
    for (int i = 0; i < x_plus.size(); ++i) {
      x_plus_ad[i] = CppAD::Value(x_plus[i]);
    }
    CPPAD_TESTVECTOR(Scalar) x_minus_ad(static_cast<int>(x_minus.size()));
    for (int i = 0; i < x_minus.size(); ++i) {
      x_minus_ad[i] = CppAD::Value(x_minus[i]);
    }

    CPPAD_TESTVECTOR(Scalar) forward_result_plus = ad_fun.Forward(0, x_plus_ad);
    CPPAD_TESTVECTOR(Scalar)
    forward_result_minus = ad_fun.Forward(0, x_minus_ad);

    Eigen::VectorXd forward_plus_eigen(forward_result_plus.size());
    Eigen::VectorXd forward_minus_eigen(forward_result_minus.size());
    for (int i = 0; i < forward_result_plus.size(); i++) {
      forward_plus_eigen(i) = forward_result_plus[i];
      forward_minus_eigen(i) = forward_result_minus[i];
    }

    J.col(i) = (forward_plus_eigen - forward_minus_eigen) / (2 * eps);
  }
  Eigen::Map<Eigen::MatrixXd> jacobian_result_eigen(jacobian_result.data(),
                                                    J.rows(), J.cols());
  BOOST_CHECK(J.isApprox(jacobian_result_eigen, 1e-3));
}

BOOST_AUTO_TEST_CASE(contact_problem_getEquivalentLCP_backward_b_wrt_G_cppad) {
  int nc = 1;
  MatrixXs G = MatrixXs::Zero(3, 3);
  G(0, 0) = 3.5;
  G(1, 1) = 3.5;
  G(2, 2) = 1.;

  VectorXs g = VectorXs::Zero(3);
  g(2) = -0.00981;
  std::vector<T> mus = {0.9};

  VectorXs AD_G_flat(G.size());
  AD_G_flat = Eigen::Map<VectorXs>(G.data(), G.cols() * G.rows());

  CppAD::Independent(AD_G_flat);
  MatrixXs G_ = Eigen::Map<MatrixXs>(AD_G_flat.data(), G.rows(), G.cols());
  cb::ContactProblem<ADScalar, cb::PyramidCone> ad_prob(G_, g, mus);
  ad_prob.setLCP();
  ad_prob.computeLCP();

  CppAD::ADFun<Scalar> ad_fun(AD_G_flat, ad_prob.b_);
  CPPAD_TESTVECTOR(Scalar) x(static_cast<size_t>(G.size()));
  for (size_t i = 0; i < G.size(); ++i) {
    x[i] = CppAD::Value(AD_G_flat[i]);
  }

  CPPAD_TESTVECTOR(Scalar) forward_result = ad_fun.Forward(0, x);
  CPPAD_TESTVECTOR(Scalar) jacobian_result = ad_fun.Jacobian(x);

  Eigen::MatrixXd J = Eigen::MatrixXd::Zero(ad_prob.b_.size(), G.size());
  double eps = 1e-6;
  for (int i = 0; i < G.size(); i++) {

    VectorXs x_fd = Eigen::Map<VectorXs>(AD_G_flat.data(), AD_G_flat.size());
    VectorXs x_plus = x_fd;
    VectorXs x_minus = x_fd;
    x_plus(i) += eps;
    x_minus(i) -= eps;

    CPPAD_TESTVECTOR(Scalar) x_plus_ad(static_cast<int>(x_plus.size()));
    for (int i = 0; i < x_plus.size(); ++i) {
      x_plus_ad[i] = CppAD::Value(x_plus[i]);
    }
    CPPAD_TESTVECTOR(Scalar) x_minus_ad(static_cast<int>(x_minus.size()));
    for (int i = 0; i < x_minus.size(); ++i) {
      x_minus_ad[i] = CppAD::Value(x_minus[i]);
    }

    CPPAD_TESTVECTOR(Scalar) forward_result_plus = ad_fun.Forward(0, x_plus_ad);
    CPPAD_TESTVECTOR(Scalar)
    forward_result_minus = ad_fun.Forward(0, x_minus_ad);

    Eigen::VectorXd forward_plus_eigen(forward_result_plus.size());
    Eigen::VectorXd forward_minus_eigen(forward_result_minus.size());
    for (int j = 0; j < forward_result_plus.size(); j++) {
      forward_plus_eigen(j) = forward_result_plus[j];
      forward_minus_eigen(j) = forward_result_minus[j];
    }

    J.col(i) = (forward_plus_eigen - forward_minus_eigen) / (2 * eps);
  }
  Eigen::Map<Eigen::MatrixXd> jacobian_result_eigen(jacobian_result.data(),
                                                    J.rows(), J.cols());
  BOOST_CHECK(J.isApprox(jacobian_result_eigen, 1e-3));
}

// BOOST_AUTO_TEST_CASE(contact_problem_getEquivalentLCP_backward_A_wrt_G_cppad)
// { This currently assumes Row-Major , but the matrices when mapped to flat
// vectors are in Col-Major
//   int nc = 1;
//   MatrixXs G = MatrixXs::Zero(3, 3);
//   G(0, 0) = 3.5;
//   G(1, 1) = 3.5;
//   G(2, 2) = 1.;

//   VectorXs g = VectorXs::Zero(3);
//   g(2) = -0.00981;
//   std::vector<T> mus = {0.9};

//   VectorXs AD_G_flat(G.size());
//   AD_G_flat = Eigen::Map<VectorXs>(G.data(), G.cols() * G.rows());

//   CppAD::Independent(AD_G_flat);
//   MatrixXs G_ = Eigen::Map<MatrixXs>(AD_G_flat.data(), G.rows(), G.cols());
//   cb::ContactProblem<ADScalar, cb::PyramidCone> ad_prob(G_, g, mus);
//   ad_prob.setLCP();
//   ad_prob.computeTangentLCP();

//   VectorXs AD_A_flat =
//       Eigen::Map<VectorXs>(ad_prob.A_.data(), ad_prob.A_.size());

//   CppAD::ADFun<Scalar> ad_fun(AD_G_flat, AD_A_flat);

//   CPPAD_TESTVECTOR(Scalar) x(static_cast<size_t>(G.size()));
//   for (size_t i = 0; i < G.size(); ++i) {
//     x[i] = CppAD::Value(AD_G_flat[i]);
//   }

//   CPPAD_TESTVECTOR(Scalar) forward_result = ad_fun.Forward(0, x);
//   CPPAD_TESTVECTOR(Scalar) jacobian_result = ad_fun.Jacobian(x);

//   std::cout << "x: " << std::endl << x << std::endl;
//   std::cout << "forward_result: " << std::endl << forward_result <<
//   std::endl; std::cout << "jacobian_result: " << std::endl << jacobian_result
//   << std::endl;

//   // remember: storage order by default is column major in Eigen
//   Eigen::MatrixXd J = Eigen::MatrixXd::Zero(AD_A_flat.size(),
//   AD_G_flat.size()); double eps = 1e-3; for (int i = 0; i < AD_G_flat.size();
//   i++) {

//     VectorXs x_fd = Eigen::Map<VectorXs>(AD_G_flat.data(), AD_G_flat.size());
//     VectorXs x_plus = x_fd;
//     VectorXs x_minus = x_fd;
//     x_plus(i) += eps;
//     x_minus(i) -= eps;

//     CPPAD_TESTVECTOR(Scalar) x_plus_ad(static_cast<int>(x_plus.size()));
//     for (int j = 0; j < x_plus.size(); ++j) {
//         x_plus_ad[j] = CppAD::Value(x_plus[j]);
//     }
//     CPPAD_TESTVECTOR(Scalar) x_minus_ad(static_cast<int>(x_minus.size()));
//     for (int j = 0; j < x_minus.size(); ++j) {
//         x_minus_ad[j] = CppAD::Value(x_minus[j]);
//     }

//     CPPAD_TESTVECTOR(Scalar) forward_result_plus = ad_fun.Forward(0,
//     x_plus_ad); CPPAD_TESTVECTOR(Scalar) forward_result_minus =
//     ad_fun.Forward(0, x_minus_ad);

//     Eigen::VectorXd forward_plus_eigen(forward_result_plus.size());
//     Eigen::VectorXd forward_minus_eigen(forward_result_minus.size());
//     for (int j = 0; j < forward_result_plus.size(); j++) {
//       forward_plus_eigen(j) = forward_result_plus[j];
//       forward_minus_eigen(j) = forward_result_minus[j];
//     }
//     J.col(i) = (forward_plus_eigen - forward_minus_eigen) / (2 * eps);

//     std::cout << "i: " << i << std::endl;
//     std::cout << "x_plus: " << std::endl << x_plus << std::endl;
//     std::cout << "x_minus: " << std::endl << x_minus << std::endl;
//     std::cout << "forward_result_plus: " << std::endl << forward_result_plus
//     << std::endl; std::cout << "forward_result_minus: " << std::endl <<
//     forward_result_minus << std::endl;

//   }
//   Eigen::Map<Eigen::MatrixXd> jacobian_result_eigen(jacobian_result.data(),
//   J.rows(), J.cols()); std::cout << "J: " << std::endl << J << std::endl;
//   std::cout << "jacobian_result_eigen: " << std::endl <<
//   jacobian_result_eigen << std::endl;

//   BOOST_CHECK(J.isApprox(jacobian_result_eigen, 1e-3));
// }

// it seems i do not need the solver for the computeTangentLCP
BOOST_AUTO_TEST_CASE(contact_problem_getEquivalentLCP_cppad) {
  int nc = 1;
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::PyramidCone> prob(A, b, mus);
  cb::NCPPGSSolver<T, cb::PyramidCone> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  int maxIter = 10000;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  settings.statistics_ = true;
  solver.solve(prob, x0, settings);
  VectorXs lam = solver.getSolution();
  VectorXs v = A * lam + b;
  BOOST_CHECK((lam).isApprox(-b, 1e-3));
  prob.setLCP();
  prob.computeLCP();
  VectorXs lam_lcp = VectorXs::Zero(6 * nc);
  prob.computeLCPSolution(lam, lam_lcp);
  VectorXs lam_b = prob.UD_ * lam_lcp;
  BOOST_CHECK((lam_b).isApprox(lam, 1e-3));
  VectorXs v_lcp = prob.A_ * lam_lcp + prob.b_;
  T vt_norm = v.segment<2>(0).norm();
  BOOST_CHECK(fabs(vt_norm - lam_lcp(5)) < 1e-4);
  T comp_lcp = v_lcp.dot(lam_lcp);
  BOOST_CHECK(fabs(comp_lcp) < 1e-5);
}

BOOST_AUTO_TEST_CASE(contact_problem_getEquivalentLCP2_cppad) {
  int nc = 3;
  MatrixXs A = MatrixXs::Zero(9, 9);
  A << 3.19551124, -0.19308891, -0.2835278, 2.6922379, 0.69578161, 1.02167151,
      1.80252347, 0.19243591, 0.28256896, -0.19308891, 2.55347549, -1.02145194,
      -0.05337345, 2.30682153, -0.28234878, 0.19335315, 2.44569244, 1.02284979,
      -0.2835278, -1.02145194, 1.74922841, -0.28354156, -1.0215015, 0.99915555,
      -0.28338194, -1.02092643, 0.25000088, 2.6922379, -0.05337345, -0.28354156,
      2.55317623, 0.1923273, 1.02172108, 2.30663115, 0.05319295, 0.28258267,
      0.69578161, 2.30682153, -1.0215015, 0.1923273, 3.1959559, -0.28236248,
      -0.69673379, 2.69165754, 1.02289942, 1.02167151, -0.28234878, 0.99915555,
      1.02172108, -0.28236248, 1.74908269, 1.02114589, -0.28220352, 0.99992817,
      1.80252347, 0.19335315, -0.28338194, 2.30663115, -0.69673379, 1.02114589,
      3.19587359, -0.19269926, 0.28242359, 0.19243591, 2.44569244, -1.02092643,
      0.05319295, 2.69165754, -0.28220352, -0.19269926, 2.55157071, 1.02232356,
      0.28256896, 1.02284979, 0.25000088, 0.28258267, 1.02289942, 0.99992817,
      0.28242359, 1.02232356, 1.75077176;
  VectorXs b = VectorXs::Zero(9);
  b << 0.0005535632995660456, -0.00021503837242162205, -0.2002703738176001,
      0.0003651400035045993, 0.00011681395199159006, -0.20929694476483335,
      0.00010131506722595163, -0.00016176693151320753, -0.009610457228502855;
  std::vector<T> mus = {0.9, 0.9, 0.9};
  cb::ContactProblem<T, cb::PyramidCone> prob(A, b, mus);
  cb::NCPPGSSolver<T, cb::PyramidCone> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 1000;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  settings.statistics_ = true;
  solver.solve(prob, x0, settings);
  VectorXs lam = solver.getSolution();
  VectorXs v = A * lam + b;
  prob.setLCP();
  prob.computeLCP();
  VectorXs lam_lcp = VectorXs::Zero(6 * nc);
  prob.computeLCPSolution(lam, lam_lcp);
  VectorXs lam_b = prob.UD_ * lam_lcp;
  BOOST_CHECK((lam_b).isApprox(lam, 1e-3));
  VectorXs v_lcp = prob.A_ * lam_lcp + prob.b_;
  T vt_norm = v.segment<2>(0).norm();
  BOOST_CHECK(fabs(vt_norm - lam_lcp(5)) < 1e-4);
  T comp_lcp = v_lcp.dot(lam_lcp);
  BOOST_CHECK(fabs(comp_lcp) < 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()
