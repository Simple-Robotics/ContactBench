#include "contactbench/contact-problem.hpp"
#include "contactbench/solvers.hpp"
#include "vector"

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>

#include <proxsuite/serialization/archive.hpp>
#include <proxsuite/serialization/eigen.hpp>
#include <pinocchio/autodiff/cppad.hpp>
// #include <cppad/cppad.hpp>

BOOST_AUTO_TEST_SUITE(SOLVERS_CPPAD)

namespace cb = contactbench;
using CppAD::AD;
using CppAD::NearEqual;

typedef double Scalar;
typedef AD<Scalar> ADScalar;
using T = Scalar;
CONTACTBENCH_EIGEN_TYPEDEFS(T);

// BOOST_AUTO_TEST_CASE(NCP_PGS_solver_init_cppad) {
//   cb::NCPPGSSolver<T, cb::IceCreamCone> solver;
//   BOOST_CHECK(true);
// }

// BOOST_AUTO_TEST_CASE(NCP_PGS_solver_setProblem_cppad) {
//   MatrixXs A = MatrixXs::Random(3, 3);
//   VectorXs b = VectorXs::Zero(3);
//   std::vector<T> mus = {0.9};
//   cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
//   cb::NCPPGSSolver<T, cb::IceCreamCone> solver;
//   solver.setProblem(prob);
//   BOOST_CHECK(true);
// }

// BOOST_AUTO_TEST_CASE(NCP_PGS_solver_solve_cppad) {
//   MatrixXs A = MatrixXs::Random(3, 3);
//   VectorXs b = VectorXs::Zero(3);
//   std::vector<T> mus = {0.9};
//   cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
//   cb::NCPPGSSolver<T, cb::IceCreamCone> solver;
//   solver.setProblem(prob);
//   VectorXs x0 = VectorXs::Zero(3);
//   int maxIter = 10;
//   cb::ContactSolverSettings<T> settings;
//   settings.max_iter_ = maxIter;
//   settings.th_stop_ = 1e-12;
//   settings.rel_th_stop_ = 1e-12;
//   settings.statistics_ = true;
//   solver.solve(prob, x0, settings);
//   BOOST_CHECK(true);
// }

// BOOST_AUTO_TEST_CASE(NCP_PGS_solver_getSolution_cppad) {
//   MatrixXs A = MatrixXs::Zero(3, 3);
//   A(0, 0) = 3.5;
//   A(1, 1) = 3.5;
//   A(2, 2) = 1.;
//   VectorXs b = VectorXs::Zero(3);
//   b(2) = -0.00981;
//   std::vector<T> mus = {0.9};
//   cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
//   cb::NCPPGSSolver<T, cb::IceCreamCone> solver;
//   solver.setProblem(prob);
//   VectorXs x0 = VectorXs::Zero(3);
//   int maxIter = 10;
//   cb::ContactSolverSettings<T> settings;
//   settings.max_iter_ = maxIter;
//   settings.th_stop_ = 1e-12;
//   settings.rel_th_stop_ = 1e-12;
//   settings.statistics_ = true;
//   solver.solve(prob, x0, settings);
//   VectorXs lam = solver.getSolution();
//   BOOST_CHECK((lam).isApprox(-b, 1e-3));
// }

// BOOST_AUTO_TEST_CASE(NCP_PGS_solver_getSolution_backward_cppad) {
//   MatrixXs A = MatrixXs::Zero(3, 3);
//   A(0, 0) = 3.5;
//   A(1, 1) = 3.5;
//   A(2, 2) = 1.;
//   VectorXs b = VectorXs::Zero(3);
//   b(2) = -0.00981;
//   CppAD::Independent(b);
//   std::vector<T> mus = {0.9};
//   cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
//   cb::NCPPGSSolver<T, cb::IceCreamCone> solver;
//   solver.setProblem(prob);
//   VectorXs x0 = VectorXs::Zero(3);
//   int maxIter = 10;
//   cb::ContactSolverSettings<T> settings;
//   settings.max_iter_ = maxIter;
//   settings.th_stop_ = 1e-12;
//   settings.rel_th_stop_ = 1e-12;
//   settings.statistics_ = true;
//   solver.solve(prob, x0, settings);
//   VectorXs lam = solver.getSolution();

//   CppAD::ADFun<Scalar> f(b, lam);
//   CPPAD_TESTVECTOR(Scalar) x_(static_cast<size_t>(b.size()));

//   for (size_t i = 0; i < b.size(); ++i) {
//     // x_[i] = CppAD::Value(b[i]);
//     x_[i] = b[i];
//   }

//   CPPAD_TESTVECTOR(Scalar) y = f.Forward(0, x_);
//   CPPAD_TESTVECTOR(Scalar) J = f.Jacobian(x_);
// }

// BOOST_AUTO_TEST_CASE(NCP_PGS_solver_getSolution2_cppad) {
//   MatrixXs A = MatrixXs::Zero(6, 6);
//   A << 0.00746183, 0.01030469, -0.02459814, 0.00146251, 0.0106132,
//   -0.02493933,
//       0.01030469, 0.05200239, 0.00350998, 0.00977178, 0.05230589, 0.00358581,
//       -0.02459814, 0.00350998, 0.11827849, 0.00287095, 0.00237138,
//       0.11994601, 0.00146251, 0.00977178, 0.00287095, 0.00750077, 0.00957129,
//       0.00312712, 0.0106132, 0.05230589, 0.00237138, 0.00957129, 0.05263244,
//       0.00242249, -0.02493933, 0.00358581, 0.11994601, 0.00312712,
//       0.00242249, 0.12164495;
//   VectorXs b = VectorXs::Zero(6);
//   b << 5.16331117e-01, -3.47716091e-02, -2.44411637e+00, 2.49915611e-01,
//       -2.36700375e-02, -2.46724553e+00;
//   std::vector<T> mus = {0.9, 0.9};
//   cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
//   cb::NCPPGSSolver<T, cb::IceCreamCone> solver;
//   solver.setProblem(prob);
//   VectorXs x0 = VectorXs::Zero(6);
//   int maxIter = 1000;
//   cb::ContactSolverSettings<T> settings;
//   settings.max_iter_ = maxIter;
//   settings.th_stop_ = 1e-12;
//   settings.rel_th_stop_ = 1e-12;
//   settings.statistics_ = true;
//   solver.solve(prob, x0, settings, 0.8, 0.);
//   VectorXs lam = solver.getSolution();
//   VectorXs v = A * lam + b;
//   VectorXs sol = VectorXs::Zero(6);
//   T comp = prob.computeContactComplementarity(lam);
//   // BOOST_CHECK(fabs(comp) < 1e-3);
//   // BOOST_CHECK((lam).isApprox(sol, 1e-3));
// }

// BOOST_AUTO_TEST_CASE(PSG_solver_separate_backward) {
//   typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXd;
//   typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

//   int nc = 1;
//   MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
//   A << 5.47732006, -0.44641268, -0.85806775, -0.44641268, 4.73256402,
//       -1.83316893, -0.85806775, -1.83316893, 2.16266878;
//   VectorXs b = VectorXs::Zero(3 * nc);
//   b << 4.58088011e-16, 5.88878006e-16, -9.61358750e-01;

//   // dlam_dg
//   CppAD::Independent(b);
//   std::vector<T> mus = {0.95};
//   cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
//   cb::NCPPGSSolver<T, cb::IceCreamCone> solver;
//   solver.setProblem(prob);
//   VectorXs x0 = VectorXs::Zero(3 * nc);
//   int maxIter = 100;
//   cb::ContactSolverSettings<T> settings;
//   settings.max_iter_ = maxIter;
//   settings.th_stop_ = 1e-12;
//   settings.rel_th_stop_ = 1e-12;
//   settings.statistics_ = true;
//   solver.solve(prob, x0, settings);
//   VectorXs lam = solver.getSolution();
//   CppAD::ADFun<Scalar> ad_fun(b, lam);
//   CPPAD_TESTVECTOR(Scalar) x(static_cast<size_t>(b.size()));
//   for (size_t i = 0; i < b.size(); ++i) {
//     // x[i] = CppAD::Value(b[i]);
//     x[i] = b[i];
//   }
//   CPPAD_TESTVECTOR(Scalar) forward_result = ad_fun.Forward(0, x);
//   CPPAD_TESTVECTOR(Scalar) jacobian_result = ad_fun.Jacobian(x);

//   MatrixXd dlam_dg = MatrixXd::Zero(3 * nc, b.size());
//   dlam_dg = Eigen::Map<MatrixXd>(jacobian_result.data(), 3 * nc, b.size());

//   // dlam_dmu
//   VectorXs mus_ad = VectorXs::Zero(mus.size());
//   mus_ad = Eigen::Map<VectorXs>(mus.data(), mus.size());
//   CppAD::Independent(mus_ad);
//   std::vector<T> mus2(mus_ad.data(), mus_ad.data() + mus_ad.size());
//   prob = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus2);
//   solver = cb::NCPPGSSolver<T, cb::IceCreamCone>();
//   solver.setProblem(prob);
//   solver.solve(prob, x0, settings);
//   lam = solver.getSolution();
//   CppAD::ADFun<Scalar> ad_fun2(mus_ad, lam);
//   x = CPPAD_TESTVECTOR(Scalar)(static_cast<size_t>(mus_ad.size()));
//   for (size_t i = 0; i < mus_ad.size(); ++i) {
//     // x[i] = CppAD::Value(mus_ad[i]);
//     x[i] = mus_ad[i];
//   }
//   forward_result = ad_fun2.Forward(0, x);
//   jacobian_result = ad_fun2.Jacobian(x);

//   MatrixXd dlam_dmu = MatrixXd::Zero(3 * nc, mus.size());
//   dlam_dmu = Eigen::Map<MatrixXd>(jacobian_result.data(), 3 * nc,
//   mus.size());

//   // dlam_dDel
//   VectorXs A_ad = VectorXs::Zero(A.size());
//   A_ad = Eigen::Map<VectorXs>(A.data(), A.size());
//   CppAD::Independent(A_ad);
//   MatrixXs A2 = Eigen::Map<MatrixXs>(A_ad.data(), A.rows(), A.cols());
//   prob = cb::ContactProblem<T, cb::IceCreamCone>(A2, b, mus2);
//   solver = cb::NCPPGSSolver<T, cb::IceCreamCone>();
//   solver.setProblem(prob);
//   solver.solve(prob, x0, settings);
//   lam = solver.getSolution();

//   CppAD::ADFun<Scalar> ad_fun3(A_ad, lam);
//   x = CPPAD_TESTVECTOR(Scalar)(static_cast<size_t>(A_ad.size()));
//   for (size_t i = 0; i < A_ad.size(); ++i) {
//     // x[i] = CppAD::Value(A_ad(i));
//     x[i] = A_ad(i);
//   }
//   forward_result = ad_fun3.Forward(0, x);
//   jacobian_result = ad_fun3.Jacobian(x);

//   MatrixXd dlam_dDel = MatrixXd::Zero(3 * nc, A.size());
//   dlam_dDel = Eigen::Map<MatrixXd>(jacobian_result.data(), 3 * nc, A.size());

//   // comparison with finite differences
//   double delta = 1e-5;
//   VectorXs lam2 = VectorXs::Zero(3 * nc);
//   VectorXs dlam_dmuj = VectorXs::Zero(3 * nc);
//   VectorXs dlam_dDeljk = VectorXs::Zero(3 * nc);
//   VectorXs dlam_dgj = VectorXs::Zero(3 * nc);

//   cb::ContactProblem<T, cb::IceCreamCone> prob2;
//   cb::NCPPGSSolver<T, cb::IceCreamCone> solver2;

//   for (int j = 0; j < 3 * nc; j++) {
//     // test derivatives wrt mu
//     if (j % 3 == 0) {
//       mus[j / 3] += delta;
//       prob2 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
//       solver2 = cb::NCPPGSSolver<T, cb::IceCreamCone>();
//       solver2.setProblem(prob2);
//       solver2.solve(prob2, x0, settings);
//       lam2 = solver2.getSolution();
//       dlam_dmuj = (lam2 - lam) / delta;
//       VectorXd dlam_dmuj_eigen = VectorXd::Zero(3 * nc);
//       for (int i = 0; i < 3 * nc; ++i) {
//         // dlam_dmuj_eigen(i) = CppAD::Value(dlam_dmuj(i));
//         dlam_dmuj_eigen(i) = dlam_dmuj(i);
//       }
//       BOOST_CHECK(dlam_dmu.col(j / 3).isApprox(dlam_dmuj_eigen, 1e-4));
//       mus[j / 3] -= delta;
//     }
//     // test derivatives wrt g
//     b(j) += delta;
//     prob2 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
//     solver2 = cb::NCPPGSSolver<T, cb::IceCreamCone>();
//     solver2.setProblem(prob2);
//     solver2.solve(prob2, x0, settings);
//     lam2 = solver2.getSolution();
//     dlam_dgj = (lam2 - lam) / delta;
//     VectorXd dlam_dgj_eigen = VectorXd::Zero(3 * nc);
//     for (int i = 0; i < 3 * nc; ++i) {
//       dlam_dgj_eigen(i) = dlam_dgj(i);
//       // dlam_dgj_eigen(i) = CppAD::Value(dlam_dgj(i));
//     }

//     BOOST_CHECK(dlam_dg.col(j).isApprox(dlam_dgj_eigen, 1e-4));
//     b(j) -= delta;
//     // test derivatives wrt Del
//     for (int k = 0; k < 3 * nc; k++) {
//       A(j, k) += delta;
//       prob2 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
//       solver2 = cb::NCPPGSSolver<T, cb::IceCreamCone>();
//       solver2.setProblem(prob2);
//       solver2.solve(prob2, x0, settings);
//       lam2 = solver2.getSolution();
//       dlam_dDeljk = (lam2 - lam) / delta;
//       VectorXd dlam_dDeljk_eigen = VectorXd::Zero(3 * nc);
//       for (int i = 0; i < 3 * nc; ++i) {
//         // dlam_dDeljk_eigen(i) = CppAD::Value(dlam_dDeljk(i));
//         dlam_dDeljk_eigen(i) = dlam_dDeljk(i);
//       }
//       BOOST_CHECK(dlam_dDel.col(j * 3 + k).isApprox(dlam_dDeljk_eigen,
//       1e-4)); A(j, k) -= delta;
//     }
//   }
// }

// BOOST_AUTO_TEST_CASE(NCP_PSG_solver_jacobian) {
//   typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXd;
//   typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

//   int nc = 1;
//   MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
//   A << 5.47732006, -0.44641268, -0.85806775, -0.44641268, 4.73256402,
//       -1.83316893, -0.85806775, -1.83316893, 2.16266878;
//   VectorXs b = VectorXs::Zero(3 * nc);
//   b << 4.58088011e-16, 5.88878006e-16, -9.61358750e-01;
//   std::vector<T> mus = {0.95};

//   VectorXs flat_A = Eigen::Map<VectorXs>(A.data(), A.size());
//   VectorXs x(b.size() + flat_A.size() + mus.size());
//   x << b, flat_A, Eigen::Map<VectorXs>(mus.data(), mus.size());

//   CppAD::Independent(x);

//   VectorXs b_ = x.head(b.size());
//   MatrixXs A_ = Eigen::Map<MatrixXs>(x.data() + b.size(), A.rows(),
//   A.cols()); std::vector<T> mus_ =
//       std::vector<T>(x.data() + b.size() + flat_A.size(),
//                      x.data() + b.size() + flat_A.size() + mus.size());

//   cb::ContactProblem<T, cb::IceCreamCone> prob(A_, b_, mus_);
//   cb::NCPPGSSolver<T, cb::IceCreamCone> solver;
//   solver.setProblem(prob);
//   VectorXs x0 = VectorXs::Zero(3 * nc);
//   int maxIter = 100;
//   cb::ContactSolverSettings<T> settings;
//   settings.max_iter_ = maxIter;
//   settings.th_stop_ = 1e-12;
//   settings.rel_th_stop_ = 1e-12;
//   settings.statistics_ = true;
//   solver.solve(prob, x0, settings);
//   VectorXs lam = solver.getSolution();
//   CppAD::ADFun<Scalar> ad_fun(x, lam);
//   CPPAD_TESTVECTOR(Scalar) x_(static_cast<size_t>(x.size()));
//   for (size_t i = 0; i < x.size(); ++i) {
//     // x_[i] = CppAD::Value(x[i]);
//     x_[i] = x[i];
//   }
//   CPPAD_TESTVECTOR(Scalar) forward_result = ad_fun.Forward(0, x_);
//   CPPAD_TESTVECTOR(Scalar) jacobian_result = ad_fun.Jacobian(x_);

//   MatrixXd dlam_db = MatrixXd::Zero(3 * nc, b.size());
//   MatrixXd dlam_dA = MatrixXd::Zero(3 * nc, A.size());
//   MatrixXd dlam_dmu = MatrixXd::Zero(3 * nc, mus.size());

//   int b_size = b.size();
//   int A_size = flat_A.size();
//   int mu_size = mus.size();

//   for (int i = 0; i < 3 * nc; ++i) {
//     int start = i * (b_size + A_size + mu_size);
//     int end_db = start + b_size;
//     int end_A = end_db + A_size;

//     dlam_db.row(i) =
//         Eigen::Map<Eigen::VectorXd>(jacobian_result.data() + start, b_size);
//     dlam_dA.row(i) =
//         Eigen::Map<Eigen::VectorXd>(jacobian_result.data() + end_db, A_size);
//     dlam_dmu.row(i) =
//         Eigen::Map<Eigen::VectorXd>(jacobian_result.data() + end_A, mu_size);
//   }

//   VectorXs dlam_dmuj = VectorXs::Zero(3 * nc);
//   VectorXs dlam_dAjk = VectorXs::Zero(3 * nc);
//   VectorXs dlam_dbj = VectorXs::Zero(3 * nc);
//   VectorXs lam2 = VectorXs::Zero(3 * nc);

//   cb::ContactProblem<T, cb::IceCreamCone> prob2;
//   cb::NCPPGSSolver<T, cb::IceCreamCone> solver2;

//   double delta = 1e-5;

//   for (int j = 0; j < 3 * nc; j++) {
//     // test derivatives wrt mu
//     if (j % 3 == 0) {
//       mus[j / 3] += delta;
//       prob2 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
//       solver2 = cb::NCPPGSSolver<T, cb::IceCreamCone>();
//       solver2.setProblem(prob2);
//       solver2.solve(prob2, x0, settings);
//       lam2 = solver2.getSolution();
//       dlam_dmuj = (lam2 - lam) / delta;
//       VectorXd dlam_dmuj_eigen = VectorXd::Zero(3 * nc);
//       for (int i = 0; i < 3 * nc; ++i) {
//         dlam_dmuj_eigen(i) = dlam_dmuj(i);
//         // dlam_dmuj_eigen(i) = CppAD::Value(dlam_dmuj(i));
//       }
//       BOOST_CHECK((dlam_dmu.col(j).isApprox(dlam_dmuj_eigen, 1e-4)));
//       mus[j / 3] -= delta;
//     }

//     // test derivatives wrt b
//     b(j) += delta;
//     prob2 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
//     solver2 = cb::NCPPGSSolver<T, cb::IceCreamCone>();
//     solver2.setProblem(prob2);
//     solver2.solve(prob2, x0, settings);
//     lam2 = solver2.getSolution();
//     dlam_dbj = (lam2 - lam) / delta;
//     VectorXd dlam_dbj_eigen = VectorXd::Zero(3 * nc);
//     for (int i = 0; i < 3 * nc; ++i) {
//       // dlam_dbj_eigen(i) = CppAD::Value(dlam_dbj(i));
//       dlam_dbj_eigen(i) = dlam_dbj(i);
//     }
//     BOOST_CHECK(dlam_db.col(j).isApprox(dlam_dbj_eigen, 1e-4));
//     b(j) -= delta;

//     // test derivatives wrt A: default storage is column major for Eigen
//     for (int k = 0; k < 3 * nc; k++) {
//       A(k, j) += delta;
//       prob2 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
//       solver2 = cb::NCPPGSSolver<T, cb::IceCreamCone>();
//       solver2.setProblem(prob2);
//       solver2.solve(prob2, x0, settings);
//       lam2 = solver2.getSolution();
//       dlam_dAjk = (lam2 - lam) / delta;
//       VectorXd dlam_dAjk_eigen = VectorXd::Zero(3 * nc);
//       for (int i = 0; i < 3 * nc; ++i) {
//         dlam_dAjk_eigen(i) = dlam_dAjk(i);
//         // dlam_dAjk_eigen(i) = CppAD::Value(dlam_dAjk(i));
//       }
//       BOOST_CHECK((dlam_dA.col(j * 3 + k)).isApprox(dlam_dAjk_eigen, 1e-3));
//       A(k, j) -= delta;
//     }
//   }
// }

BOOST_AUTO_TEST_CASE(NCP_PGS_solver_vjp2) {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXd;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

  int nc = 1;
  // define problem as scalar type double
  MatrixXd A = MatrixXd::Zero(3 * nc, 3 * nc);
  A << 5.47732006, -0.44641268, -0.85806775, -0.44641268, 4.73256402,
      -1.83316893, -0.85806775, -1.83316893, 2.16266878;
  VectorXd b = VectorXd::Zero(3 * nc);
  b << 4.58088011e-16, 5.88878006e-16, -9.61358750e-01;
  std::vector<Scalar> mus = {0.95};

  cb::ContactProblem<Scalar, cb::IceCreamCone> prob(A, b, mus);
  cb::NCPPGSSolver<Scalar, cb::IceCreamCone> solver;
  solver.setProblem(prob);
  VectorXd x0 = VectorXd::Zero(3 * nc);
  int maxIter = 100;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  settings.statistics_ = true;
  solver.solve(prob, x0, settings);
  VectorXd lam = solver.getSolution();
  VectorXd dL_dlam = VectorXd::Zero(3 * nc);
  double delta = 1e-5;
  VectorXd lam3 = VectorXd::Zero(3 * nc);
  VectorXd dlam_dmuj = VectorXd::Zero(3 * nc);
  VectorXd dlam_dgj = VectorXd::Zero(3 * nc);
  VectorXd dlam_dDeljk = VectorXd::Zero(3 * nc);
  VectorXd dlami_dmus = VectorXd::Zero(nc);
  VectorXd dlami_dg = VectorXd::Zero(3 * nc);
  MatrixXd dlami_dDel = MatrixXd::Zero(3 * nc, 3 * nc);
  cb::ContactProblem<Scalar, cb::IceCreamCone> prob3;
  cb::NCPPGSSolver<Scalar, cb::IceCreamCone> solver3;
  for (int i = 0; i < 3 * nc; i++) {
    // std::cout << "i:" << i << std::endl;
    dL_dlam.setZero();
    dL_dlam(i) = 1.;
    solver.vjp_cppad(prob, dL_dlam, settings);
    dlami_dmus = solver.getdLdmus();
    dlami_dg = solver.getdLdg();
    dlami_dDel = solver.getdLdDel();
    for (int j = 0; j < 3 * nc; j++) {
      // test derivatives wrt mu
      if (j % 3 == 0) {
        mus[j / 3] += delta;
        prob3 = cb::ContactProblem<Scalar, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::NCPPGSSolver<Scalar, cb::IceCreamCone>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings);
        lam3 = solver3.getSolution();
        dlam_dmuj = (lam3 - lam) / delta;
        BOOST_CHECK(std::abs(dlam_dmuj(i) - dlami_dmus(j / 3)) < 1e-3);
        mus[j / 3] -= delta;
      }
      // test derivatives wrt g
      b(j) += delta;
      prob3 = cb::ContactProblem<Scalar, cb::IceCreamCone>(A, b, mus);
      solver3 = cb::NCPPGSSolver<Scalar, cb::IceCreamCone>();
      solver3.setProblem(prob3);
      solver3.solve(prob3, x0, settings);
      lam3 = solver3.getSolution();
      dlam_dgj = (lam3 - lam) / delta;
      BOOST_CHECK(std::abs(dlam_dgj(i) - dlami_dg(j)) < 1e-3);
      b(j) -= delta;
      // test derivatives wrt Del
      for (int k = 0; k < 3 * nc; k++) {
        // std::cout << "k:" << k << std::endl;
        A(j, k) += delta;
        prob3 = cb::ContactProblem<Scalar, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::NCPPGSSolver<Scalar, cb::IceCreamCone>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings);
        lam3 = solver3.getSolution();
        dlam_dDeljk = (lam3 - lam) / delta;
        BOOST_CHECK(std::abs(dlam_dDeljk(i) - dlami_dDel(j, k)) < 1e-3);
        A(j, k) -= delta;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_vjp2) {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXd;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

  int nc = 1;
  // define problem as scalar type double
  MatrixXd A = MatrixXd::Zero(3 * nc, 3 * nc);
  A << 5.47732006, -0.44641268, -0.85806775, -0.44641268, 4.73256402,
      -1.83316893, -0.85806775, -1.83316893, 2.16266878;
  VectorXd b = VectorXd::Zero(3 * nc);
  b << 4.58088011e-16, 5.88878006e-16, -9.61358750e-01;
  std::vector<Scalar> mus = {0.95};

  cb::ContactProblem<Scalar, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPPGSSolver<Scalar> solver;
  solver.setProblem(prob);
  VectorXd x0 = VectorXd::Zero(3 * nc);
  int maxIter = 100;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  settings.statistics_ = true;
  solver.solve(prob, x0, settings);
  VectorXd lam = solver.getSolution();
  VectorXd dL_dlam = VectorXd::Zero(3 * nc);
  double delta = 1e-5;
  VectorXd lam3 = VectorXd::Zero(3 * nc);
  VectorXd dlam_dmuj = VectorXd::Zero(3 * nc);
  VectorXd dlam_dgj = VectorXd::Zero(3 * nc);
  VectorXd dlam_dDeljk = VectorXd::Zero(3 * nc);
  VectorXd dlami_dmus = VectorXd::Zero(nc);
  VectorXd dlami_dg = VectorXd::Zero(3 * nc);
  MatrixXd dlami_dDel = MatrixXd::Zero(3 * nc, 3 * nc);
  cb::ContactProblem<Scalar, cb::IceCreamCone> prob3;
  cb::CCPPGSSolver<Scalar> solver3;
  for (int i = 0; i < 3 * nc; i++) {
    // std::cout << "i:" << i << std::endl;
    dL_dlam.setZero();
    dL_dlam(i) = 1.;
    solver.vjp_cppad(prob, dL_dlam, settings);
    dlami_dmus = solver.getdLdmus();
    dlami_dg = solver.getdLdg();
    dlami_dDel = solver.getdLdDel();
    for (int j = 0; j < 3 * nc; j++) {
      // test derivatives wrt mu
      if (j % 3 == 0) {
        mus[j / 3] += delta;
        prob3 = cb::ContactProblem<Scalar, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::CCPPGSSolver<Scalar>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings);
        lam3 = solver3.getSolution();
        dlam_dmuj = (lam3 - lam) / delta;
        BOOST_CHECK(std::abs(dlam_dmuj(i) - dlami_dmus(j / 3)) < 1e-3);
        mus[j / 3] -= delta;
      }
      // test derivatives wrt g
      b(j) += delta;
      prob3 = cb::ContactProblem<Scalar, cb::IceCreamCone>(A, b, mus);
      solver3 = cb::CCPPGSSolver<Scalar>();
      solver3.setProblem(prob3);
      solver3.solve(prob3, x0, settings);
      lam3 = solver3.getSolution();
      dlam_dgj = (lam3 - lam) / delta;
      BOOST_CHECK(std::abs(dlam_dgj(i) - dlami_dg(j)) < 1e-3);
      b(j) -= delta;
      // test derivatives wrt Del
      for (int k = 0; k < 3 * nc; k++) {
        // std::cout << "k:" << k << std::endl;
        A(j, k) += delta;
        prob3 = cb::ContactProblem<Scalar, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::CCPPGSSolver<Scalar>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings);
        lam3 = solver3.getSolution();
        dlam_dDeljk = (lam3 - lam) / delta;
        BOOST_CHECK(std::abs(dlam_dDeljk(i) - dlami_dDel(j, k)) < 1e-3);
        A(j, k) -= delta;
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
