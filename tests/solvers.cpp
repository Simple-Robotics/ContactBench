#include "contactbench/contact-problem.hpp"
#include "contactbench/solvers.hpp"
#include "vector"

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>

#include <proxsuite/serialization/archive.hpp>
#include <proxsuite/serialization/eigen.hpp>
#include <Eigen/Eigenvalues>

BOOST_AUTO_TEST_SUITE(SOLVERS)

namespace cb = contactbench;
using T = double;
CONTACTBENCH_EIGEN_TYPEDEFS(T);

// BOOST_AUTO_TEST_SUITE(solvers)
BOOST_AUTO_TEST_CASE(NCP_PGS_solver_init) {
  cb::NCPPGSSolver<T, cb::IceCreamCone> solver;
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(NCP_PGS_solver_setProblem) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::NCPPGSSolver<T, cb::IceCreamCone> solver;
  solver.setProblem(prob);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(NCP_PGS_solver_solve) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::NCPPGSSolver<T, cb::IceCreamCone> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  int maxIter = 10;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(NCP_PGS_solver_getSolution) {
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::NCPPGSSolver<T, cb::IceCreamCone> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  int maxIter = 10;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  VectorXs lam = solver.getSolution();
  BOOST_CHECK((lam).isApprox(-b, 1e-3));
}

BOOST_AUTO_TEST_CASE(NCP_PGS_solver_getSolution2) {
  MatrixXs A = MatrixXs::Zero(6, 6);
  A << 0.00746183, 0.01030469, -0.02459814, 0.00146251, 0.0106132, -0.02493933,
      0.01030469, 0.05200239, 0.00350998, 0.00977178, 0.05230589, 0.00358581,
      -0.02459814, 0.00350998, 0.11827849, 0.00287095, 0.00237138, 0.11994601,
      0.00146251, 0.00977178, 0.00287095, 0.00750077, 0.00957129, 0.00312712,
      0.0106132, 0.05230589, 0.00237138, 0.00957129, 0.05263244, 0.00242249,
      -0.02493933, 0.00358581, 0.11994601, 0.00312712, 0.00242249, 0.12164495;
  VectorXs b = VectorXs::Zero(6);
  b << 5.16331117e-01, -3.47716091e-02, -2.44411637e+00, 2.49915611e-01,
      -2.36700375e-02, -2.46724553e+00;
  std::vector<T> mus = {0.9, 0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::NCPPGSSolver<T, cb::IceCreamCone> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(6);
  int maxIter = 1000;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  settings.statistics_ = true;
  solver.solve(prob, x0, settings, 0.8, 0.);
  VectorXs lam = solver.getSolution();
  VectorXs v = A * lam + b;
  VectorXs sol = VectorXs::Zero(6);
  double comp = prob.computeContactComplementarity(lam);
  CONTACTBENCH_UNUSED(comp);
  // BOOST_CHECK(std::abs(comp) < 1e-3);
  // BOOST_CHECK((lam).isApprox(sol, 1e-3));
}

BOOST_AUTO_TEST_CASE(LCP_QP_solver_computeSmallestEigenValue) {
  int nc = 1;
  CONTACTBENCH_UNUSED(nc);
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::PyramidCone> prob(A, b, mus);
  cb::LCPQPSolver<T> solver2;
  solver2.setProblem(prob);
  double eps = 1e-9;
  int maxiter = 20;
  MatrixXs H = MatrixXs::Zero(6, 6);
  H(0, 0) = 3.5;
  H(1, 1) = 3.5;
  H(2, 2) = 1.;
  double l_min = solver2.computeSmallestEigenValue(H, eps, maxiter);
  BOOST_CHECK(std::abs(l_min - H(3, 3)) < 1e-2);
  H(3, 3) = -1.894;
  l_min = solver2.computeSmallestEigenValue(H, eps, maxiter);
  BOOST_CHECK(std::abs(l_min - H(3, 3)) < 1e-2);
}

BOOST_AUTO_TEST_CASE(LCP_QP_solver_computeSmallestEigenValue_randomMatrix) {
  int nc = 1;
  CONTACTBENCH_UNUSED(nc);
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  std::vector<T> mus = {0.9};
  // need to initialize the lcpqp solver with a problem to use
  // method to compute smallest eigenvalue
  cb::ContactProblem<T, cb::PyramidCone> prob(A, b, mus);
  cb::LCPQPSolver<T> solver2;
  solver2.setProblem(prob);
  double eps = 1e-9;
  int maxiter = 200;
  // compute smallest eigenvalue of a random matrix
  MatrixXs H = MatrixXs::Random(6, 6);
  H = H * H.transpose();
  double l_min = solver2.computeSmallestEigenValue(H, eps, maxiter);
  double l_max = solver2.computeLargestEigenValue(H, eps, maxiter);

  Eigen::SelfAdjointEigenSolver<MatrixXs> eigensolver(H);
  if (eigensolver.info() != Eigen::Success)
    std::cout << "eigensolver failed to compute eigenvalues." << std::endl;
  VectorXs eigenvalues = eigensolver.eigenvalues();
  double l_min_eigen = eigenvalues.minCoeff();
  double l_max_eigen = eigenvalues.maxCoeff();
  // std::cout << "l_min_eigen  " << l_min_eigen << std::endl;
  // std::cout << "l_min  " << l_min << std::endl;
  // std::cout << "l_max_eigen  " << l_max_eigen << std::endl;
  // std::cout << "l_max  " << l_max << std::endl;

  BOOST_CHECK(std::abs(l_min - l_min_eigen) < 1e-2);
  BOOST_CHECK(std::abs(l_max - l_max_eigen) < 1e-2);
}

BOOST_AUTO_TEST_CASE(LCP_QP_solver_getSolution) {
  int nc = 1;
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::PyramidCone> prob(A, b, mus);
  cb::NCPPGSSolver<T, cb::PyramidCone> solver1;
  cb::LCPQPSolver<T> solver2;
  solver1.setProblem(prob);
  solver2.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  int maxIter = 1000;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver1.solve(prob, x0, settings);
  solver2.solve(prob, x0, settings);
  VectorXs lam1 = solver1.getSolution();
  VectorXs lam2 = solver2.getSolution();
  VectorXs lam_lcp1 = VectorXs::Zero(6 * nc);
  VectorXs lam_lcp2 = VectorXs::Zero(6 * nc);
  prob.computeLCPSolution(lam1, lam_lcp1);
  prob.computeLCPSolution(lam2, lam_lcp2);
  VectorXs v_lcp1 = prob.A_ * lam_lcp1 + prob.b_;
  VectorXs v_lcp2 = prob.A_ * lam_lcp2 + prob.b_;
  BOOST_CHECK((lam1).isApprox(-b, 1e-3));
  BOOST_CHECK((lam2).isApprox(-b, 1e-3));
  BOOST_CHECK(std::abs(lam_lcp1.dot(v_lcp1)) < 1e-5);
  BOOST_CHECK(std::abs(lam_lcp2.dot(v_lcp2)) < 1e-5);
  T linear_comp1 = prob.computeLinearComplementarity(lam1);
  T linear_comp2 = prob.computeLinearComplementarity(lam2);
  BOOST_CHECK(linear_comp1 < 1e-5);
  BOOST_CHECK(linear_comp2 < 1e-5);
}

BOOST_AUTO_TEST_CASE(LCP_QP_solver_getSolution2) {
  int nc = 1;
  MatrixXs A = MatrixXs::Zero(3, 3);
  A << 5.32596138, -0.72388089, -0.9707955, -0.72388089, 4.11915949,
      -2.07305584, -0.9707955, -2.07305584, 2.88477717;
  VectorXs b = VectorXs::Zero(3);
  b << 0.00065288, 0.00139417, -0.00799683;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::PyramidCone> prob(A, b, mus);
  cb::NCPPGSSolver<T, cb::PyramidCone> solver1;
  cb::LCPQPSolver<T> solver2;
  solver1.setProblem(prob);
  solver2.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  int maxIter = 100;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver1.solve(prob, x0, settings);
  maxIter = 100;
  solver2.solve(prob, x0, settings);
  VectorXs lam1 = solver1.getSolution();
  VectorXs lam2 = solver2.getSolution();
  MatrixXs H = solver2.getQPH();
  VectorXs g = solver2.getQPg();
  MatrixXs C = solver2.getQPC();
  VectorXs l = solver2.getQPl();
  double l_min = solver2.computeSmallestEigenValue(H, 1e-12, 100);
  double l_min2 = solver2.computeSmallestEigenValue(H, 1e-12, 10);
  double l_max = solver2.computeLargestEigenValue(H, 1e-12, 100);
  double l_max2 = solver2.computeLargestEigenValue(H, 1e-12, 10);
  CONTACTBENCH_UNUSED(l_min);
  CONTACTBENCH_UNUSED(l_min2);
  CONTACTBENCH_UNUSED(l_max);
  CONTACTBENCH_UNUSED(l_max2);
  // std::cout << l_min << std::endl;
  // std::cout << l_min2 << std::endl;
  // std::cout << l_max << std::endl;
  // std::cout << l_max2 << std::endl;
  if (false) {
    proxsuite::serialization::saveToJSON(H, "H_lcp");
    proxsuite::serialization::saveToJSON(g, "g_lcp");
    proxsuite::serialization::saveToJSON(C, "C_lcp");
    proxsuite::serialization::saveToJSON(l, "l_lcp");
  }
  VectorXs lam_lcp1 = VectorXs::Zero(6 * nc);
  VectorXs lam_lcp2 = VectorXs::Zero(6 * nc);
  prob.computeLCPSolution(lam1, lam_lcp1);
  prob.computeLCPSolution(lam2, lam_lcp2);
  VectorXs v_lcp1 = prob.A_ * lam_lcp1 + prob.b_;
  VectorXs v_lcp2 = prob.A_ * lam_lcp2 + prob.b_;
  BOOST_CHECK(lam1.isApprox(lam2, 5e-2));
  BOOST_CHECK(std::abs(lam_lcp1.dot(v_lcp1)) < 1e-5);
  BOOST_CHECK(std::abs(lam_lcp2.dot(v_lcp2)) < 1e-5);
  T linear_comp1 = prob.computeLinearComplementarity(lam1);
  T linear_comp2 = prob.computeLinearComplementarity(lam2);
  BOOST_CHECK(linear_comp1 < 1e-5);
  BOOST_CHECK(linear_comp2 < 1e-5);
  double cost_pgs = 0.5 * lam_lcp1.dot(H * lam_lcp1) + lam_lcp1.dot(g);
  BOOST_CHECK(std::abs(cost_pgs) < 1e-4);
  maxIter = 100;
  solver2.solve(prob, lam1, settings);
  VectorXs lam3 = solver2.getSolution();
  VectorXs lam_lcp3 = VectorXs::Zero(6 * nc);
  prob.computeLCPSolution(lam3, lam_lcp3);
  VectorXs v_lcp3 = prob.A_ * lam_lcp3 + prob.b_;
  BOOST_CHECK((lam3).isApprox(lam1, 5e-2));
  BOOST_CHECK(std::abs(lam_lcp3.dot(v_lcp3)) < 1e-5);
  T linear_comp3 = prob.computeLinearComplementarity(lam3);
  BOOST_CHECK(linear_comp3 < 1e-5);
}

BOOST_AUTO_TEST_CASE(LCP_QP_solver_getSolution3) {
  int nc = 1;
  MatrixXs A = MatrixXs::Zero(3, 3);
  A << 5.47732006, -0.44641268, -0.85806775, -0.44641268, 4.73256402,
      -1.83316893, -0.85806775, -1.83316893, 2.16266878;
  VectorXs b = VectorXs::Zero(3);
  b << 4.58088011e-16, 5.88878006e-16, -9.61358750e-01;
  std::vector<T> mus = {0.95};
  cb::ContactProblem<T, cb::PyramidCone> prob(A, b, mus);
  cb::NCPPGSSolver<T, cb::PyramidCone> solver1;
  cb::LCPQPSolver<T> solver2;
  solver1.setProblem(prob);
  solver2.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  int maxIter = 100;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver1.solve(prob, x0, settings);
  maxIter = 100;
  solver2.solve(prob, x0, settings);
  VectorXs lam1 = solver1.getSolution();
  VectorXs lam2 = solver2.getSolution();
  MatrixXs H = solver2.getQPH();
  VectorXs g = solver2.getQPg();
  MatrixXs C = solver2.getQPC();
  VectorXs l = solver2.getQPl();
  double l_min = solver2.computeSmallestEigenValue(H, 1e-12, 100);
  double l_min2 = solver2.computeSmallestEigenValue(H, 1e-12, 10);
  double l_max = solver2.computeLargestEigenValue(H, 1e-12, 100);
  double l_max2 = solver2.computeLargestEigenValue(H, 1e-12, 10);
  CONTACTBENCH_UNUSED(l_min);
  CONTACTBENCH_UNUSED(l_min2);
  CONTACTBENCH_UNUSED(l_max);
  CONTACTBENCH_UNUSED(l_max2);
  // std::cout << l_min << std::endl;
  // std::cout << l_min2 << std::endl;
  // std::cout << l_max << std::endl;
  // std::cout << l_max2 << std::endl;
  if (false) {
    proxsuite::serialization::saveToJSON(H, "H_lcp");
    proxsuite::serialization::saveToJSON(g, "g_lcp");
    proxsuite::serialization::saveToJSON(C, "C_lcp");
    proxsuite::serialization::saveToJSON(l, "l_lcp");
  }
  VectorXs lam_lcp1 = VectorXs::Zero(6 * nc);
  VectorXs lam_lcp2 = VectorXs::Zero(6 * nc);
  prob.computeLCPSolution(lam1, lam_lcp1);
  prob.computeLCPSolution(lam2, lam_lcp2);
  VectorXs v_lcp1 = prob.A_ * lam_lcp1 + prob.b_;
  VectorXs v_lcp2 = prob.A_ * lam_lcp2 + prob.b_;
  BOOST_CHECK(lam1.isApprox(lam2, 5e-2));
  BOOST_CHECK(std::abs(lam_lcp1.dot(v_lcp1)) < 1e-5);
  BOOST_CHECK(std::abs(lam_lcp2.dot(v_lcp2)) < 1e-5);
  T linear_comp1 = prob.computeLinearComplementarity(lam1);
  T linear_comp2 = prob.computeLinearComplementarity(lam2);
  BOOST_CHECK(linear_comp1 < 1e-5);
  BOOST_CHECK(linear_comp2 < 1e-5);
  double cost_pgs = 0.5 * lam_lcp1.dot(H * lam_lcp1) + lam_lcp1.dot(g);
  BOOST_CHECK(std::abs(cost_pgs) < 1e-4);
  maxIter = 100;
  solver2.solve(prob, lam1, settings);
  VectorXs lam3 = solver2.getSolution();
  VectorXs lam_lcp3 = VectorXs::Zero(6 * nc);
  prob.computeLCPSolution(lam3, lam_lcp3);
  VectorXs v_lcp3 = prob.A_ * lam_lcp3 + prob.b_;
  BOOST_CHECK((lam3).isApprox(lam1, 5e-2));
  BOOST_CHECK(std::abs(lam_lcp3.dot(v_lcp3)) < 1e-5);
  T linear_comp3 = prob.computeLinearComplementarity(lam3);
  BOOST_CHECK(linear_comp3 < 1e-5);
}

BOOST_AUTO_TEST_CASE(NCP_StagProj_solver_init) {
  // std::cout << "NCP_StagProj_solver_init" << std::endl;
  cb::NCPStagProjSolver<cb::IceCreamCone> solver;
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(NCP_StagProj_solver_setProblem) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::NCPStagProjSolver<cb::IceCreamCone> solver;
  solver.setProblem(prob);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(NCP_StagProj_solver_solveNormal) {
  int nc = 1;
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::NCPStagProjSolver<cb::IceCreamCone> solver;
  solver.setProblem(prob);
  std::vector<int> ind_n;
  for (int i = 0; i < nc; i++) {
    ind_n.push_back(i * 3 + 2);
  }
  MatrixXs G_n = MatrixXs::Zero(nc, nc);
  prob.Del_->evaluateDel();
  solver.evaluateNormalDelassus(prob);
  solver.evaluateTangentDelassus(prob);
  G_n = prob.Del_->G_(ind_n, ind_n);
  double rho = 1e-6;
  double eigval_max_n = solver.computeLargestEigenValueNormal(G_n);
  double eigval_min = rho;
  solver.eigval_min_ = eigval_min;
  solver.eigval_max_n_ = eigval_max_n;
  double rho_n = std::sqrt(eigval_max_n * eigval_min) *
                 (std::pow(eigval_max_n / eigval_min, 0.4));
  solver.rho_n_ = rho_n;
  int maxIter = 100;
  solver.computeGntild(G_n, rho + rho_n);
  MatrixXs G_n_tild = G_n + (rho + rho_n) * MatrixXs::Identity(nc, nc);
  solver.computeGninv(G_n_tild);
  MatrixXs G_n_llt = G_n_tild;
  Eigen::LLT<Eigen::Ref<MatrixXs>> llt_n(G_n_llt);
  llt_n.compute(G_n_tild);
  MatrixXs G_n_inv = G_n_tild.inverse();
  VectorXs gn = b(ind_n);
  VectorXs lam_n = VectorXs::Zero(nc);
  VectorXs lam0 = VectorXs::Zero(nc);
  solver.solveNormal(G_n, gn, llt_n, lam0, lam_n, maxIter, 1e-6, 1e-8, 1e-6,
                     1.);
  VectorXs gamma_n = solver.getDualNormal();
  VectorXs dual_feas = G_n * lam_n + gn + gamma_n;
  // std::cout << lam_n << std::endl;
  // std::cout << prob.G_ << std::endl;
  // std::cout << G_n << std::endl;
  // std::cout << b << std::endl;
  // std::cout << gn << std::endl;
  // std::cout << solver.stop_n_ << std::endl;
  // std::cout << solver.rel_stop_n_ << std::endl;
  // std::cout << solver.prim_feas_n_ << std::endl;
  // std::cout << solver.dual_feas_n_ << std::endl;
  BOOST_CHECK(dual_feas.lpNorm<Eigen::Infinity>() < 1e-3);
  // BOOST_CHECK(lam_n.isApprox(-b(ind_n), 1e-3));
}

BOOST_AUTO_TEST_CASE(NCP_StagProj_solver_solveTangent) {
  int nc = 1;
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::NCPStagProjSolver<cb::IceCreamCone> solver;
  solver.setProblem(prob);
  std::vector<int> ind_t, ind_n;
  for (int i = 0; i < nc; i++) {
    ind_t.push_back(i * 3);
    ind_t.push_back(i * 3 + 1);
    ind_n.push_back(i * 3 + 2);
  }
  MatrixXs G_t = MatrixXs::Zero(2 * nc, 2 * nc);
  MatrixXs G_tn = MatrixXs::Zero(2 * nc, nc);
  prob.Del_->evaluateDel();
  G_t = prob.Del_->G_(ind_t, ind_t);
  G_tn = prob.Del_->G_(ind_t, ind_n);
  solver.evaluateNormalDelassus(prob);
  solver.evaluateTangentDelassus(prob);
  double rho = 1e-6;
  double eigval_max_t = solver.computeLargestEigenValueTangent(G_t);
  double eigval_min = rho;
  solver.eigval_min_ = eigval_min;
  solver.eigval_max_t_ = eigval_max_t;
  double rho_t = std::sqrt(eigval_max_t * eigval_min) *
                 (std::pow(eigval_max_t / eigval_min, 0.4));
  solver.rho_t_ = rho_t;
  int maxIter = 100;
  solver.computeGttild(G_t, rho + rho_t);
  MatrixXs G_t_tild = G_t + (rho + rho_t) * MatrixXs::Identity(2 * nc, 2 * nc);
  solver.computeGtinv(G_t_tild);
  MatrixXs G_t_llt = G_t_tild;
  Eigen::LLT<Eigen::Ref<MatrixXs>> llt_t(G_t_llt);
  llt_t.compute(G_t_tild);
  MatrixXs G_t_inv = G_t_tild.inverse();
  VectorXs gt = b(ind_t) - G_tn * b(ind_n);
  VectorXs lam_t = VectorXs::Zero(2 * nc);
  VectorXs lam0 = VectorXs::Zero(2 * nc);
  solver.solveTangent(G_t, gt, llt_t, prob, -b(ind_n), lam0, lam_t, maxIter);
  VectorXs gamma_t = solver.getDualTangent();
  VectorXs dual_feas = G_t * lam_t + gt + gamma_t;
  BOOST_CHECK(dual_feas.lpNorm<Eigen::Infinity>() < 1e-3);
  BOOST_CHECK((lam_t + b(ind_t)).lpNorm<Eigen::Infinity>() < 1e-3);
}

BOOST_AUTO_TEST_CASE(NCP_StagProj_solver_solve) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::NCPStagProjSolver<cb::IceCreamCone> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  int maxIter = 10;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(NCP_StagProj_solver_getSolution) {
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::NCPStagProjSolver<cb::IceCreamCone> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  int maxIter = 10;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  VectorXs lam = solver.getSolution();
  BOOST_CHECK((lam).isApprox(-b, 1e-3));
}

BOOST_AUTO_TEST_CASE(NCP_StagProj_solver_getSolution2) {
  MatrixXs A = MatrixXs::Zero(6, 6);
  A << 0.00746183, 0.01030469, -0.02459814, 0.00146251, 0.0106132, -0.02493933,
      0.01030469, 0.05200239, 0.00350998, 0.00977178, 0.05230589, 0.00358581,
      -0.02459814, 0.00350998, 0.11827849, 0.00287095, 0.00237138, 0.11994601,
      0.00146251, 0.00977178, 0.00287095, 0.00750077, 0.00957129, 0.00312712,
      0.0106132, 0.05230589, 0.00237138, 0.00957129, 0.05263244, 0.00242249,
      -0.02493933, 0.00358581, 0.11994601, 0.00312712, 0.00242249, 0.12164495;
  VectorXs b = VectorXs::Zero(6);
  b << 5.16331117e-01, -3.47716091e-02, -2.44411637e+00, 2.49915611e-01,
      -2.36700375e-02, -2.46724553e+00;

  std::vector<T> mus = {0.9, 0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::NCPStagProjSolver<cb::IceCreamCone> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(6);
  int maxIter = 5;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  settings.statistics_ = true;
  solver.solve(prob, x0, settings);
  VectorXs lam = solver.getSolution();
  VectorXs v = A * lam + b;
  VectorXs sol = VectorXs::Zero(6);
  double comp = prob.computeContactComplementarity(lam);
  BOOST_CHECK(std::abs(comp) < 1e-3);
  // BOOST_CHECK((lam).isApprox(sol, 1e-3));
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_init) {
  cb::CCPPGSSolver<T> solver;
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_setProblem) {
  // std::cout << "CCP_PGS_solver_setProblem" << std::endl;
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPPGSSolver<T> solver;
  solver.setProblem(prob);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_solve) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPPGSSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  int maxIter = 10;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_getSolution) {
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPPGSSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  int maxIter = 200;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  VectorXs lam = solver.getSolution();
  BOOST_CHECK((lam).isApprox(-b, 1e-3));
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_getSolution2) {
  // breaking contact
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
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPPGSSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(9);
  int maxIter = 200;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  VectorXs lam = solver.getSolution();
  VectorXs sol = VectorXs::Zero(9);
  sol << -0.04659727125606961, -0.0006406377552907818, 0.07386375452370315,
      0.012217208053733344, 0.042661426660780384, 0.10433162765826219, 0.0, 0.0,
      0.0;
  double comp = lam.dot(A * lam + b);
  BOOST_CHECK(std::abs(comp) < 1e-3);
  BOOST_CHECK((lam).isApprox(sol, 1e-3));
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_polish1) {
  // 1 sticking contact (ball on floor)
  int nc = 1;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  A << 3.5, 0., 0., 0., 3.5, 0., 0., 0., 1.;
  VectorXs b = VectorXs::Zero(3 * nc);
  b << 0., 0., -0.00481;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPPGSSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 200;
  bool polish = true;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings, 0., polish);
  // std::cout << " constraint matrix:   " <<
  // prob.getLinearInConstraintMatrix()
  // << std::endl;
  VectorXs lam = solver.getSolution();
  polish = false;
  solver.solve(prob, x0, settings, 0., polish);
  VectorXs lam2 = solver.getSolution();
  // VectorXs sol = VectorXs::Zero(3*nc);
  // sol << -0.04659727125606961;
  VectorXs c = A * lam + b;
  double comp = lam.dot(A * lam + b);
  // std::cout << " comp:   " << comp << std::endl;
  BOOST_CHECK(std::abs(comp) < 1e-3);
  BOOST_CHECK((lam).isApprox(lam2, 1e-3));
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_polish2) {
  // 1 sliding contact (ball dragged on floor)
  int nc = 1;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  A << 3.5, 0., 0., 0., 3.5, 0., 0., 0., 1.;
  VectorXs b = VectorXs::Zero(3 * nc);
  b << 0., 0.0132435, -0.00981;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPPGSSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 200;
  bool polish = true;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings, 0., polish);
  // std::cout << " constraint matrix:   " <<
  // prob.getLinearInConstraintMatrix()
  // << std::endl;
  VectorXs lam = solver.getSolution();
  polish = false;
  solver.solve(prob, x0, settings, 0., polish);
  VectorXs lam2 = solver.getSolution();
  // VectorXs sol = VectorXs::Zero(3*nc);
  // sol << -0.04659727125606961;
  VectorXs c = A * lam + b;
  double comp = lam.dot(A * lam + b);
  // std::cout << " comp:   " << comp << std::endl;
  BOOST_CHECK(std::abs(comp) < 1e-3);
  BOOST_CHECK((lam).isApprox(lam2, 1e-3));
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_polish3) {
  // 1 breaking contact (lifted ball on floor)
  int nc = 1;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  A << 3.5, 0., 0., 0., 3.5, 0., 0., 0., 1.;
  VectorXs b = VectorXs::Zero(3 * nc);
  b << 0., 0., 0.00019;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPPGSSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 200;
  bool polish = true;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings, 0., polish);
  // std::cout << " constraint matrix:   " <<
  // prob.getLinearInConstraintMatrix()
  // << std::endl;
  VectorXs lam = solver.getSolution();
  // std::cout << " lam:   " << lam << std::endl;
  polish = false;
  solver.solve(prob, x0, settings, 0., polish);
  VectorXs lam2 = solver.getSolution();
  // std::cout << " lam2:   " << lam2 << std::endl;
  // VectorXs sol = VectorXs::Zero(3*nc);
  // sol << -0.04659727125606961;
  VectorXs c = A * lam + b;
  double comp = lam.dot(c);
  // std::cout << " c:   " << c << std::endl;
  // std::cout << " comp:   " << comp << std::endl;
  BOOST_CHECK(std::abs(comp) < 1e-3);
  VectorXs c2 = A * lam2 + b;
  double comp2 = lam2.dot(c2);
  CONTACTBENCH_UNUSED(comp2);
  // std::cout << " c2:   " << c2 << std::endl;
  // std::cout << " comp2:   " << comp2 << std::endl;
  BOOST_CHECK((lam - lam2).isMuchSmallerThan(1e-3, 1e-3));
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_polish4) {
  // return;
  // 2 sticking and 1 breaking contact
  // std::cout << " testing polish " << std::endl;
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
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPPGSSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(9);
  int maxIter = 200;
  bool polish = true;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings, 0., polish);
  // std::cout << " constraint matrix:   " <<
  // prob.getLinearInConstraintMatrix()
  // << std::endl;
  VectorXs lam = solver.getSolution();
  VectorXs sol = VectorXs::Zero(9);
  sol << -0.04659727125606961, -0.0006406377552907818, 0.07386375452370315,
      0.012217208053733344, 0.042661426660780384, 0.10433162765826219, 0.0, 0.0,
      0.0;
  VectorXs c = A * lam + b;
  double comp = lam.dot(A * lam + b);
  // std::cout << " comp:   " << comp << std::endl;
  BOOST_CHECK(std::abs(comp) < 1e-3);
  BOOST_CHECK((lam).isApprox(sol, 1e-3));
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_polish5) {
  // return;
  // sliding contact
  // std::cout << " testing vjp 2" << std::endl;
  int nc = 4;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  A << 3.79313165, 1.46341469, 1.72273192, 1.15898547, 1.17073161, 1.37818536,
      4.08581475, -1.17073157, -1.37818531, 1.45166856, -1.46341465,
      -1.72273187, 1.46341469, 4.45166853, -1.37818541, -1.82926842, 4.08581472,
      1.72273182, 1.82926858, 1.15898543, -1.72273196, -1.46341454, 0.79313162,
      1.37818527, 1.72273192, -1.37818541, 4.00000018, 1.722732, -1.37818548,
      1.00000003, 1.72273193, -1.37818543, 1.00000014, 1.72273201, -1.37818549,
      -2., 1.15898547, -1.82926842, 1.722732, 4.45166847, -1.46341456,
      1.37818543, 0.79313162, 1.46341451, -1.37818538, 4.08581462, 1.82926837,
      -1.72273195, 1.17073161, 4.08581472, -1.37818548, -1.46341456, 3.79313172,
      1.7227319, 1.46341468, 1.45166856, -1.72273204, -1.17073149, 1.15898557,
      1.37818533, 1.37818536, 1.72273182, 1.00000003, 1.37818543, 1.7227319,
      3.99999989, 1.37818538, 1.72273184, -2., 1.37818544, 1.72273192,
      0.99999986, 4.08581475, 1.82926858, 1.72273193, 0.79313162, 1.46341468,
      1.37818538, 4.45166866, -1.46341463, -1.37818533, 1.15898553, -1.82926852,
      -1.72273188, -1.17073157, 1.15898543, -1.37818543, 1.46341451, 1.45166856,
      1.72273184, -1.46341463, 3.79313152, -1.72273198, 1.17073144, 4.08581465,
      1.37818528, -1.37818531, -1.72273196, 1.00000014, -1.37818538,
      -1.72273204, -2., -1.37818533, -1.72273198, 4.00000011, -1.37818539,
      -1.72273206, 0.99999997, 1.45166856, -1.46341454, 1.72273201, 4.08581462,
      -1.17073149, 1.37818544, 1.15898553, 1.17073144, -1.37818539, 3.79313159,
      1.4634145, -1.72273196, -1.46341465, 0.79313162, -1.37818549, 1.82926837,
      1.15898557, 1.72273192, -1.82926852, 4.08581465, -1.72273206, 1.4634145,
      4.45166861, 1.37818534, -1.72273187, 1.37818527, -2., -1.72273195,
      1.37818533, 0.99999986, -1.72273188, 1.37818528, 0.99999997, -1.72273196,
      1.37818534, 3.99999982;
  VectorXs b = VectorXs::Zero(3 * nc);
  b << -0.14457191, 1.30114761, -0.00981, -0.14457196, 1.3011476, -0.00981,
      -0.14457191, 1.30114755, -0.00981, -0.14457196, 1.30114755, -0.00981;
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPPGSSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 200;
  bool polish = true;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings, 0., polish);
  // std::cout << " constraint matrix:   " <<
  // prob.getLinearInConstraintMatrix()
  // << std::endl;
  VectorXs lam = solver.getSolution();
  VectorXs c = A * lam + b;
  double comp = lam.dot(A * lam + b);
  // std::cout << " comp:   " << comp << std::endl;
  BOOST_CHECK_SMALL(comp, 1e-3);
  //  polishing the already polished solution
  // solver._polish(prob, lam, maxIter, 1e-10, 1e-10, 0., false);
  // VectorXs lam2 = solver.getSolution();
  // VectorXs c2 = A * lam2 + b;
  // double comp2 = lam2.dot(A * lam2 + b);
  // std::cout << " comp:   " << comp2 << std::endl;
  // BOOST_CHECK_SMALL(comp2, 1e-3);
  // double delta  = 1e-8;
  // mus[0] +=  delta;
  // cb::ContactProblem<T, cb::IceCreamCone> prob2(A, b, mus);
  // solver.setProblem(prob2);
  // solver.solve(prob2, x0, maxIter, 1e-10, 1e-10, 0., polish);
  // VectorXs lam3 = solver.getSolution();
  // VectorXs c3 = A * lam3 + b;
  // double comp3 = lam.dot(A * lam3 + b);
  // std::cout << " comp:   " << comp << std::endl;
  // BOOST_CHECK_SMALL(comp3, 1e-3);
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_polish6) {
  // return;
  // 4 sliding contact points
  // std::cout << " testing vjp 2" << std::endl;
  int nc = 4;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  A << 4.1224, 1.5, 1.56, 1.1224, 1.5, 1.56, 4.1224, -1.5, -1.56, 1.1224, -1.5,
      -1.56, 1.5, 4.1224, -1.56, -1.5, 4.1224, 1.56, 1.5, 1.1224, -1.56, -1.5,
      1.1224, 1.56, 1.56, -1.56, 4., 1.56, -1.56, 1., 1.56, -1.56, 1., 1.56,
      -1.56, -2., 1.1224, -1.5, 1.56, 4.1224, -1.5, 1.56, 1.1224, 1.5, -1.56,
      4.1224, 1.5, -1.56, 1.5, 4.1224, -1.56, -1.5, 4.1224, 1.56, 1.5, 1.1224,
      -1.56, -1.5, 1.1224, 1.56, 1.56, 1.56, 1., 1.56, 1.56, 4., 1.56, 1.56,
      -2., 1.56, 1.56, 1., 4.1224, 1.5, 1.56, 1.1224, 1.5, 1.56, 4.1224, -1.5,
      -1.56, 1.1224, -1.5, -1.56, -1.5, 1.1224, -1.56, 1.5, 1.1224, 1.56, -1.5,
      4.1224, -1.56, 1.5, 4.1224, 1.56, -1.56, -1.56, 1., -1.56, -1.56, -2.,
      -1.56, -1.56, 4., -1.56, -1.56, 1., 1.1224, -1.5, 1.56, 4.1224, -1.5,
      1.56, 1.1224, 1.5, -1.56, 4.1224, 1.5, -1.56, -1.5, 1.1224, -1.56, 1.5,
      1.1224, 1.56, -1.5, 4.1224, -1.56, 1.5, 4.1224, 1.56, -1.56, 1.56, -2.,
      -1.56, 1.56, 1., -1.56, 1.56, 1., -1.56, 1.56, 4.;
  VectorXs b = VectorXs::Zero(3 * nc);
  b << 0., 1., -0.00981, 0., 1., -0.00981, 0., 1., -0.00981, 0., 1., -0.00981;
  std::vector<T> mus = {0.95, 0.95, 0.95, 0.95};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPPGSSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 200;
  bool polish = true;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings, 0., polish);
  // std::cout << " constraint matrix:   " <<
  // prob.getLinearInConstraintMatrix()
  // << std::endl;
  VectorXs lam = solver.getSolution();
  VectorXs c = A * lam + b;
  double comp = lam.dot(A * lam + b);
  // std::cout << " comp:   " << comp << std::endl;
  BOOST_CHECK_SMALL(comp, 1e-3);
  polish = false;
  solver.solve(prob, x0, settings, 0., polish);
  VectorXs lam_not_pol = solver.getSolution();
  BOOST_CHECK(lam.isApprox(lam_not_pol, 1e-3));
  VectorXs c_not_pol = A * lam_not_pol + b;
  BOOST_CHECK(c.isApprox(c_not_pol, 1e-3));
}

BOOST_AUTO_TEST_CASE(CCP_ADMM_solver_getSolution) {
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPADMMSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  int maxIter = 200;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  VectorXs lam = solver.getSolution();
  BOOST_CHECK((lam).isApprox(-b, 1e-3));
}

BOOST_AUTO_TEST_CASE(CCP_ADMM_solver_getSolution2) {
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
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPADMMSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(9);
  int maxIter = 200;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  VectorXs lam = solver.getSolution();
  VectorXs sol = VectorXs::Zero(9);
  sol << -0.04659727125606961, -0.0006406377552907818, 0.07386375452370315,
      0.012217208053733344, 0.042661426660780384, 0.10433162765826219, 0.0, 0.0,
      0.0;
  double comp = lam.dot(A * lam + b);
  BOOST_CHECK(std::abs(comp) < 1e-3);
  // BOOST_CHECK((lam).isApprox(sol, 1e-3));
}

BOOST_AUTO_TEST_CASE(CCP_ADMM_solver_getSolution3) {
  MatrixXs A = MatrixXs::Zero(24, 24);
  A << 4.96601249e+03, -1.24028947e+03, -1.03195111e+03, 2.30447824e+03,
      3.88616385e+03, 1.94994280e+03, 1.45951760e+03, -2.29620501e+03,
      -1.94994280e+03, -7.82661745e+02, 1.00173900e+03, 1.03195111e+03,
      5.00569796e+02, -2.37144192e+03, -1.94994280e+03, -2.11420699e+03,
      1.76122730e+02, 1.03195111e+03, 3.21393432e+03, 7.85766563e+01,
      -1.03195111e+03, -1.37897998e+03, 2.65716875e+03, 1.94994280e+03,
      -1.24028947e+03, 3.27878751e+03, -1.94994280e+03, -1.22978144e+03,
      9.10336032e+02, -1.03195111e+03, 5.27173946e+02, 3.88243507e+03,
      1.03195111e+03, -7.34602910e+02, 2.20542097e+03, 1.94994280e+03,
      -2.50168026e+02, 1.95904889e+03, 1.03195111e+03, 1.08048962e+03,
      6.09930563e+02, 1.94994280e+03, -1.65427288e+03, 6.61057234e+02,
      -1.94994280e+03, 2.96085554e+02, -8.51448938e+02, -1.03195111e+03,
      -1.03195111e+03, -1.94994280e+03, 4.00000000e+03, 7.58640766e+00,
      -2.20616011e+03, 1.00000000e+03, -9.66332306e+02, -1.98328058e+03,
      1.00000000e+03, 8.54218390e+02, -2.03408725e+03, -2.00000000e+03,
      -4.61925060e+01, -2.20568952e+03, -1.00000000e+03, 1.20284855e+02,
      -2.20289163e+03, -4.00000000e+03, -1.46024393e+02, -2.20133525e+03,
      2.00000000e+03, 1.35875647e+03, -1.73809690e+03, -1.00000000e+03,
      2.30447824e+03, -1.22978144e+03, 7.58640766e+00, 2.62243547e+03,
      -1.03160934e+01, 2.20616011e+03, 2.35801029e+03, -1.14747097e+03,
      -2.20616011e+03, 2.43083635e+03, 1.01105325e+03, -7.58640766e+00,
      6.22431064e+02, -4.85753216e+00, -2.20616011e+03, 6.31888052e+02,
      3.23596410e+01, -7.58640766e+00, 6.10596276e+02, -4.26485382e+01,
      7.58640766e+00, 4.98016371e+02, 3.73513100e+02, 2.20616011e+03,
      3.88616385e+03, 9.10336032e+02, -2.20616011e+03, -1.03160934e+01,
      5.62236453e+03, 7.58640766e+00, -1.57286069e+02, -3.43398435e+02,
      -7.58640766e+00, -3.77303655e+03, 1.25974787e+03, 2.20616011e+03,
      -4.76417309e+01, -2.37710954e+03, -7.58640766e+00, -3.02731719e+03,
      4.58022369e+02, 2.20616011e+03, 3.03673497e+03, 4.22323986e+02,
      -2.20616011e+03, -2.22929481e+03, 2.85514900e+03, 7.58640766e+00,
      1.94994280e+03, -1.03195111e+03, 1.00000000e+03, 2.20616011e+03,
      7.58640766e+00, 4.00000000e+03, 1.98328058e+03, -9.66332306e+02,
      -2.00000000e+03, 2.03408725e+03, 8.54218390e+02, 1.00000000e+03,
      2.20568952e+03, -4.61925060e+01, -4.00000000e+03, 2.20289163e+03,
      1.20284855e+02, -1.00000000e+03, 2.20133525e+03, -1.46024393e+02,
      -1.00000000e+03, 1.73809690e+03, 1.35875647e+03, 2.00000000e+03,
      1.45951760e+03, 5.27173946e+02, -9.66332306e+02, 2.35801029e+03,
      -1.57286069e+02, 1.98328058e+03, 3.19796591e+03, 1.18127965e+03,
      -1.98328058e+03, 2.94036113e+03, 2.48063125e+03, 9.66332306e+02,
      5.92616646e+02, 1.57459500e+03, -1.98328058e+03, 1.85590630e+03,
      3.74363693e+02, 9.66332306e+02, -7.34821615e+02, 3.21962367e+02,
      -9.66332306e+02, 1.08220524e+03, -4.75864330e+02, 1.98328058e+03,
      -2.29620501e+03, 3.88243507e+03, -1.98328058e+03, -1.14747097e+03,
      -3.43398435e+02, -9.66332306e+02, 1.18127965e+03, 5.04683409e+03,
      9.66332306e+02, 5.14699483e+02, 2.77304760e+03, 1.98328058e+03,
      -2.04376970e+02, 3.26141835e+03, 9.66332306e+02, 2.39017486e+03,
      6.90862704e+02, 1.98328058e+03, -2.92597948e+03, 7.54841051e+02,
      -1.98328058e+03, 1.10161392e+03, -1.85180864e+03, -9.66332306e+02,
      -1.94994280e+03, 1.03195111e+03, 1.00000000e+03, -2.20616011e+03,
      -7.58640766e+00, -2.00000000e+03, -1.98328058e+03, 9.66332306e+02,
      4.00000000e+03, -2.03408725e+03, -8.54218390e+02, 1.00000000e+03,
      -2.20568952e+03, 4.61925060e+01, 2.00000000e+03, -2.20289163e+03,
      -1.20284855e+02, -1.00000000e+03, -2.20133525e+03, 1.46024393e+02,
      -1.00000000e+03, -1.73809690e+03, -1.35875647e+03, -4.00000000e+03,
      -7.82661745e+02, -7.34602910e+02, 8.54218390e+02, 2.43083635e+03,
      -3.77303655e+03, 2.03408725e+03, 2.94036113e+03, 5.14699483e+02,
      -2.03408725e+03, 5.17264097e+03, 1.07097802e+03, -8.54218390e+02,
      6.26593896e+02, 2.51243533e+03, -2.03408725e+03, 3.34801704e+03,
      -5.85365984e+01, -8.54218390e+02, -2.20328620e+03, -9.53657547e+01,
      8.54218390e+02, 2.30406598e+03, -2.01557387e+03, 2.03408725e+03,
      1.00173900e+03, 2.20542097e+03, -2.03408725e+03, 1.01105325e+03,
      1.25974787e+03, 8.54218390e+02, 2.48063125e+03, 2.77304760e+03,
      -8.54218390e+02, 1.07097802e+03, 3.07215903e+03, 2.03408725e+03,
      2.77273379e+02, 1.73000918e+03, -8.54218390e+02, 1.36920008e+03,
      6.49469050e+02, 2.03408725e+03, -8.80592515e+02, 6.33526353e+02,
      -2.03408725e+03, 5.51837443e+02, -3.14612245e+02, 8.54218390e+02,
      1.03195111e+03, 1.94994280e+03, -2.00000000e+03, -7.58640766e+00,
      2.20616011e+03, 1.00000000e+03, 9.66332306e+02, 1.98328058e+03,
      1.00000000e+03, -8.54218390e+02, 2.03408725e+03, 4.00000000e+03,
      4.61925060e+01, 2.20568952e+03, -1.00000000e+03, -1.20284855e+02,
      2.20289163e+03, 2.00000000e+03, 1.46024393e+02, 2.20133525e+03,
      -4.00000000e+03, -1.35875647e+03, 1.73809690e+03, -1.00000000e+03,
      5.00569796e+02, -2.50168026e+02, -4.61925060e+01, 6.22431064e+02,
      -4.76417309e+01, 2.20568952e+03, 5.92616646e+02, -2.04376970e+02,
      -2.20568952e+03, 6.26593896e+02, 2.77273379e+02, 4.61925060e+01,
      2.62371780e+03, 6.27998201e+01, -2.20568731e+03, 2.67765446e+03,
      2.01197653e+02, 4.61924598e+01, 2.55703676e+03, -1.14591538e+02,
      -4.61924598e+01, 2.07043277e+03, 1.60852459e+03, 2.20568731e+03,
      -2.37144192e+03, 1.95904889e+03, -2.20568952e+03, -4.85753216e+00,
      -2.37710954e+03, -4.61925060e+01, 1.57459500e+03, 3.26141835e+03,
      4.61925060e+01, 2.51243533e+03, 1.73000918e+03, 2.20568952e+03,
      6.27998201e+01, 5.62109044e+03, 4.61924598e+01, 2.79711109e+03,
      2.77846444e+03, 2.20568731e+03, -2.87401898e+03, 2.81823628e+03,
      -2.20568731e+03, 1.89250788e+02, -3.31237839e+02, -4.61924598e+01,
      -1.94994280e+03, 1.03195111e+03, -1.00000000e+03, -2.20616011e+03,
      -7.58640766e+00, -4.00000000e+03, -1.98328058e+03, 9.66332306e+02,
      2.00000000e+03, -2.03408725e+03, -8.54218390e+02, -1.00000000e+03,
      -2.20568731e+03, 4.61924598e+01, 4.00000400e+03, -2.20288943e+03,
      -1.20284735e+02, 1.00000100e+03, -2.20133305e+03, 1.46024246e+02,
      1.00000100e+03, -1.73809517e+03, -1.35875511e+03, -2.00000200e+03,
      -2.11420699e+03, 1.08048962e+03, 1.20284855e+02, 6.31888052e+02,
      -3.02731719e+03, 2.20289163e+03, 1.85590630e+03, 2.39017486e+03,
      -2.20289163e+03, 3.34801704e+03, 1.36920008e+03, -1.20284855e+02,
      2.67765446e+03, 2.79711109e+03, -2.20288943e+03, 5.61348769e+03,
      1.63322710e+02, -1.20284735e+02, -3.85675521e+02, -1.17709143e+02,
      1.20284735e+02, 3.99592093e+03, -8.59925247e+02, 2.20288943e+03,
      1.76122730e+02, 6.09930563e+02, -2.20289163e+03, 3.23596410e+01,
      4.58022369e+02, 1.20284855e+02, 3.74363693e+02, 6.90862704e+02,
      -1.20284855e+02, -5.85365984e+01, 6.49469050e+02, 2.20289163e+03,
      2.01197653e+02, 2.77846444e+03, -1.20284735e+02, 1.63322710e+02,
      2.63132056e+03, 2.20288943e+03, 1.52773664e+02, 2.61412255e+03,
      -2.20288943e+03, -1.39932338e+03, 2.02213937e+03, 1.20284735e+02,
      1.03195111e+03, 1.94994280e+03, -4.00000000e+03, -7.58640766e+00,
      2.20616011e+03, -1.00000000e+03, 9.66332306e+02, 1.98328058e+03,
      -1.00000000e+03, -8.54218390e+02, 2.03408725e+03, 2.00000000e+03,
      4.61924598e+01, 2.20568731e+03, 1.00000100e+03, -1.20284735e+02,
      2.20288943e+03, 4.00000400e+03, 1.46024246e+02, 2.20133305e+03,
      -2.00000200e+03, -1.35875511e+03, 1.73809517e+03, 1.00000100e+03,
      3.21393432e+03, -1.65427288e+03, -1.46024393e+02, 6.10596276e+02,
      3.03673497e+03, 2.20133525e+03, -7.34821615e+02, -2.92597948e+03,
      -2.20133525e+03, -2.20328620e+03, -8.80592515e+02, 1.46024393e+02,
      2.55703676e+03, -2.87401898e+03, -2.20133305e+03, -3.85675521e+02,
      1.52773664e+02, 1.46024246e+02, 5.60926266e+03, -1.98131758e+02,
      -1.46024246e+02, 1.10968154e+02, 4.10663282e+03, 2.20133305e+03,
      7.85766563e+01, 6.61057234e+02, -2.20133525e+03, -4.26485382e+01,
      4.22323986e+02, -1.46024393e+02, 3.21962367e+02, 7.54841051e+02,
      1.46024393e+02, -9.53657547e+01, 6.33526353e+02, 2.20133525e+03,
      -1.14591538e+02, 2.81823628e+03, 1.46024246e+02, -1.17709143e+02,
      2.61412255e+03, 2.20133305e+03, -1.98131758e+02, 2.63554559e+03,
      -2.20133305e+03, -1.62601793e+03, 1.79814561e+03, -1.46024246e+02,
      -1.03195111e+03, -1.94994280e+03, 2.00000000e+03, 7.58640766e+00,
      -2.20616011e+03, -1.00000000e+03, -9.66332306e+02, -1.98328058e+03,
      -1.00000000e+03, 8.54218390e+02, -2.03408725e+03, -4.00000000e+03,
      -4.61924598e+01, -2.20568731e+03, 1.00000100e+03, 1.20284735e+02,
      -2.20288943e+03, -2.00000200e+03, -1.46024246e+02, -2.20133305e+03,
      4.00000400e+03, 1.35875511e+03, -1.73809517e+03, 1.00000100e+03,
      -1.37897998e+03, 2.96085554e+02, 1.35875647e+03, 4.98016371e+02,
      -2.22929481e+03, 1.73809690e+03, 1.08220524e+03, 1.10161392e+03,
      -1.73809690e+03, 2.30406598e+03, 5.51837443e+02, -1.35875647e+03,
      2.07043277e+03, 1.89250788e+02, -1.73809517e+03, 3.99592093e+03,
      -1.39932338e+03, -1.35875511e+03, 1.10968154e+02, -1.62601793e+03,
      1.35875511e+03, 3.76035935e+03, -1.45565383e+03, 1.73809517e+03,
      2.65716875e+03, -8.51448938e+02, -1.73809690e+03, 3.73513100e+02,
      2.85514900e+03, 1.35875647e+03, -4.75864330e+02, -1.85180864e+03,
      -1.35875647e+03, -2.01557387e+03, -3.14612245e+02, 1.73809690e+03,
      1.60852459e+03, -3.31237839e+02, -1.35875511e+03, -8.59925247e+02,
      2.02213937e+03, 1.73809517e+03, 4.10663282e+03, 1.79814561e+03,
      -1.73809517e+03, -1.45565383e+03, 4.48444889e+03, 1.35875511e+03,
      1.94994280e+03, -1.03195111e+03, -1.00000000e+03, 2.20616011e+03,
      7.58640766e+00, 2.00000000e+03, 1.98328058e+03, -9.66332306e+02,
      -4.00000000e+03, 2.03408725e+03, 8.54218390e+02, -1.00000000e+03,
      2.20568731e+03, -4.61924598e+01, -2.00000200e+03, 2.20288943e+03,
      1.20284735e+02, 1.00000100e+03, 2.20133305e+03, -1.46024246e+02,
      1.00000100e+03, 1.73809517e+03, 1.35875511e+03, 4.00000400e+03;
  VectorXs b = VectorXs::Zero(24);
  b << 0.0000000e+00, 0.0000000e+00, -9.8100000e-03, 0.0000000e+00,
      0.0000000e+00, -9.8100000e-03, 0.0000000e+00, 0.0000000e+00,
      -9.8100000e-03, 0.0000000e+00, 0.0000000e+00, -9.8100000e-03,
      0.0000000e+00, 0.0000000e+00, 6.9388939e-18, 0.0000000e+00, 0.0000000e+00,
      6.9388939e-18, 0.0000000e+00, 0.0000000e+00, 6.9388939e-18, 0.0000000e+00,
      0.0000000e+00, 6.9388939e-18;
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.95, 0.95, 0.95, 0.95};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPADMMSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(24);
  int maxIter = 100;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  settings.statistics_ = true;
  solver.solve(prob, x0, settings, 1e-6, 1., 0.);
  VectorXs lam = solver.getSolution();
  VectorXs sol = VectorXs::Zero(24);
  double comp = lam.dot(A * lam + b);
  BOOST_CHECK(std::abs(comp) < 1e-3);
  // BOOST_CHECK((lam).isApprox(sol, 1e-3));
}

BOOST_AUTO_TEST_CASE(CCP_ADMM_solver_getSolution4) {
  MatrixXs A = MatrixXs::Zero(12, 12);
  A << 4966.01249165, -1240.28946522, -1031.95110853, 2304.47823585,
      3886.16384948, 1949.94279712, 3101.26567494, 847.17130135, -1031.95113888,
      -649.26439802, 2922.42830527, 1949.94282247, -1240.28946522,
      3278.78750708, -1949.9427959, -1229.78144134, 910.33603352,
      -1031.95110621, -1764.54125972, 244.96873579, -1949.94277319, 67.16266426,
      -898.95556738, -1031.95111298, -1031.95110853, -1949.9427959,
      3999.99999644, 7.58640634, -2206.16011298, 999.99999644, 386.24666091,
      -2172.09889442, 1999.99997419, 866.07608026, -2029.06682359,
      -1000.00002581, 2304.47823585, -1229.78144134, 7.58640634, 2622.43547551,
      -10.3160952, 2206.16011297, 603.00131155, 105.05297493, 7.58640054,
      577.32186417, 232.87730159, 2206.16010695, 3886.16384948, 910.33603352,
      -2206.16011298, -10.3160952, 5622.36452322, 7.58640897, 2846.78851543,
      1138.38123324, -2206.16013868, -1420.06658107, 3332.42285314, 7.58644628,
      1949.94279712, -1031.95110621, 999.99999644, 2206.16011297, 7.58640897,
      3999.99999643, 2172.0988648, 386.24664759, -1000.00002588, 2029.06685565,
      866.07608543, 1999.99997412, 3101.26567494, -1764.54125972, 386.24666091,
      603.00131155, 2846.78851543, 2172.0988648, 5530.45135733, 517.11456951,
      386.24623677, 1395.34700283, 3307.86814296, 2172.09671685, 847.17130135,
      244.96873579, -2172.09889442, 105.05297493, 1138.38123324, 386.24664759,
      517.11456951, 2714.35693425, -2172.0967231, -797.50151956, 3037.92945963,
      386.2462716, -1031.95113888, -1949.94277319, 1999.99997419, 7.58640054,
      -2206.16013868, -1000.00002588, 386.24623677, -2172.0967231,
      4000.00395195, 866.07521885, -2029.06482046, 1000.00095188, -649.26439802,
      67.16266426, 866.07608026, 577.32186417, -1420.06658107, 2029.06685565,
      1395.34700283, -797.50151956, 866.07521885, 3084.73536801, -1083.16570071,
      2029.06480651, 2922.42830527, -898.95556738, -2029.06682359, 232.87730159,
      3332.42285314, 866.07608543, 3307.86814296, 3037.92945963, -2029.06482046,
      -1083.16570071, 5160.07292371, 866.07525138, 1949.94282247,
      -1031.95111298, -1000.00002581, 2206.16010695, 7.58644628, 1999.99997412,
      2172.09671685, 386.2462716, 1000.00095188, 2029.06480651, 866.07525138,
      4000.00395181;
  VectorXs b = VectorXs::Zero(12);
  b << -3.45372014e-08, 1.11640710e-07, -9.81012734e-03, -8.30347887e-08,
      8.22393506e-08, -9.81012765e-03, -9.59246610e-08, 6.67565704e-08,
      -1.27337139e-07, -1.08501617e-07, 4.34056213e-08, -1.27665779e-07;
  std::vector<T> mus = {0.9, 0.9, 0.95, 0.95};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPADMMSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(12);
  int maxIter = 200;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  VectorXs lam = solver.getSolution();
  VectorXs v = A * lam + b;
  VectorXs sol = VectorXs::Zero(12);
  double comp = lam.dot(A * lam + b);
  CONTACTBENCH_UNUSED(comp);
  // BOOST_CHECK(std::abs(comp) < 1e-3);
  // BOOST_CHECK(prob.isInside(lam, 1e-4));
  // BOOST_CHECK(prob.isInsideDual(v, 1e-4));
}

BOOST_AUTO_TEST_CASE(CCPADMM_regularization) {
  MatrixXs A = MatrixXs::Zero(12, 12);
  A << 1.69932559, -0.47651881, -0.8105002, 0.87268438, -0.47651881,
      -0.81103521, 1.69932559, 0.47584594, 0.81103643, 0.87268438, 0.47584594,
      0.81050142, -0.47651881, 2.40566234, -1.16187148, 0.47140398, 2.40566234,
      1.15928786, -0.47651881, 1.31357036, -1.16457793, 0.47140398, 1.31357036,
      1.15658141, -0.8105002, -1.16187148, 2.06840372, -0.80868605, -1.16187148,
      0.21907277, -0.8105002, -1.16396154, 0.58514108, -0.80868605, -1.16396154,
      -1.26418988, 0.87268438, 0.47140398, -0.80868605, 1.69053155, 0.47140398,
      -0.80822906, 0.87268438, -0.47082924, 0.80822802, 1.69053155, -0.47082924,
      0.80868501, -0.47651881, 2.40566234, -1.16187148, 0.47140398, 2.40566234,
      1.15928786, -0.47651881, 1.31357036, -1.16457793, 0.47140398, 1.31357036,
      1.15658141, -0.81103521, 1.15928786, 0.21907277, -0.80822906, 1.15928786,
      2.06006266, -0.81103521, 1.15605492, -1.26425928, -0.80822906, 1.15605492,
      0.5767306, 1.69932559, -0.47651881, -0.8105002, 0.87268438, -0.47651881,
      -0.81103521, 1.69932559, 0.47584594, 0.81103643, 0.87268438, 0.47584594,
      0.81050142, 0.47584594, 1.31357036, -1.16396154, -0.47082924, 1.31357036,
      1.15605492, 0.47584594, 2.40422499, -1.16134239, -0.47082924, 2.40422499,
      1.15867408, 0.81103643, -1.16457793, 0.58514108, 0.80822802, -1.16457793,
      -1.26425928, 0.81103643, -1.16134239, 2.06847329, 0.80822802, -1.16134239,
      0.21907293, 0.87268438, 0.47140398, -0.80868605, 1.69053155, 0.47140398,
      -0.80822906, 0.87268438, -0.47082924, 0.80822802, 1.69053155, -0.47082924,
      0.80868501, 0.47584594, 1.31357036, -1.16396154, -0.47082924, 1.31357036,
      1.15605492, 0.47584594, 2.40422499, -1.16134239, -0.47082924, 2.40422499,
      1.15867408, 0.81050142, 1.15658141, -1.26418988, 0.80868501, 1.15658141,
      0.5767306, 0.81050142, 1.15867408, 0.21907293, 0.80868501, 1.15867408,
      2.05999341;
  VectorXs b = VectorXs::Zero(12);
  b << -3.99839052e-25, -1.35624669e-21, -9.80999983e-03, 2.14993475e-24,
      -1.35624669e-21, -9.80999983e-03, -3.99839052e-25, 8.82015178e-21,
      -9.80999983e-03, 2.14993475e-24, 8.82015178e-21, -9.80999983e-03;
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPADMMSolver<T> solver;
  cb::CCPPGSSolver<T> solver2;
  solver.setProblem(prob);
  solver2.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(12);
  int maxIter = 1000;
  double eps_reg = 1e-1;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  settings.statistics_ = true;
  solver.solve(prob, x0, settings, 1e-6, 1., eps_reg);
  solver2.solve(prob, x0, settings, eps_reg);
  prob.Del_->evaluateDel();
  // VectorXs R_reg = eps_reg * prob.Del_->G_.diagonal();
  VectorXs R_reg = VectorXs::Constant(3 * 4, eps_reg);
  VectorXs lam = solver.getSolution();
  VectorXs lam2 = solver2.getSolution();
  VectorXs v = A * lam + b;
  VectorXs v2 = A * lam2 + b;
  VectorXs v_reg = v + R_reg.cwiseProduct(lam);
  VectorXs v_reg2 = v2 + R_reg.cwiseProduct(lam2);
  VectorXs sol = VectorXs::Zero(12);
  double comp = lam.dot(v_reg);
  double comp2 = lam2.dot(v_reg2);
  double cost = 0.5 * lam.dot(A * lam + R_reg.cwiseProduct(lam)) + lam.dot(b);
  double cost2 =
      0.5 * lam.dot(A * lam2 + R_reg.cwiseProduct(lam2)) + lam2.dot(b);
  CONTACTBENCH_UNUSED(cost);
  CONTACTBENCH_UNUSED(cost2);
  BOOST_CHECK(comp < 1e-6);
  BOOST_CHECK(comp2 < 1e-6);
  BOOST_CHECK(v.isApprox(v2, 1e-4));
}

BOOST_AUTO_TEST_CASE(CCP_ADMMPrimal_solver_getSolution) {
  // std::cout << "CCP_ADMMPrimal_solver_getSolution" << std::endl;
  //  ball on floor
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  MatrixXs M = MatrixXs::Zero(6, 6);
  M.diagonal() << 1., 1., 1., .004, .004, .004;
  VectorXs dqf = VectorXs::Zero(6);
  dqf(2) = -0.00981;
  VectorXs vstar = VectorXs::Zero(3);
  MatrixXs J = MatrixXs::Zero(3, 6);
  J << 1., 0., 0., 0., -0.1, 0., 0., 1., 0., 0.1, 0., 0., 0., 0., 1., 0., 0.,
      0.;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, M, J, dqf, vstar, mus);
  cb::CCPADMMPrimalSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  int maxIter = 200;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  VectorXs lam = solver.getDualSolution();
  BOOST_CHECK((lam).isApprox(-b, 1e-3));
}

BOOST_AUTO_TEST_CASE(CCP_ADMMPrimal_solver_getSolution2) {
  // std::cout << "CCP_ADMMPrimal_solver_getSolution2" << std::endl;
  //  cube on floor
  int nc = 4;
  int nv = 6;
  MatrixXs G = MatrixXs::Zero(3 * nc, 3 * nc);
  G << 4.1224, 1.5, 1.56, 1.1224, 1.5, 1.56, 4.1224, -1.5, -1.56, 1.1224, -1.5,
      -1.56, 1.5, 4.1224, -1.56, -1.5, 4.1224, 1.56, 1.5, 1.1224, -1.56, -1.5,
      1.1224, 1.56, 1.56, -1.56, 4., 1.56, -1.56, 1., 1.56, -1.56, 1., 1.56,
      -1.56, -2., 1.1224, -1.5, 1.56, 4.1224, -1.5, 1.56, 1.1224, 1.5, -1.56,
      4.1224, 1.5, -1.56, 1.5, 4.1224, -1.56, -1.5, 4.1224, 1.56, 1.5, 1.1224,
      -1.56, -1.5, 1.1224, 1.56, 1.56, 1.56, 1., 1.56, 1.56, 4., 1.56, 1.56,
      -2., 1.56, 1.56, 1., 4.1224, 1.5, 1.56, 1.1224, 1.5, 1.56, 4.1224, -1.5,
      -1.56, 1.1224, -1.5, -1.56, -1.5, 1.1224, -1.56, 1.5, 1.1224, 1.56, -1.5,
      4.1224, -1.56, 1.5, 4.1224, 1.56, -1.56, -1.56, 1., -1.56, -1.56, -2.,
      -1.56, -1.56, 4., -1.56, -1.56, 1., 1.1224, -1.5, 1.56, 4.1224, -1.5,
      1.56, 1.1224, 1.5, -1.56, 4.1224, 1.5, -1.56, -1.5, 1.1224, -1.56, 1.5,
      1.1224, 1.56, -1.5, 4.1224, -1.56, 1.5, 4.1224, 1.56, -1.56, 1.56, -2.,
      -1.56, 1.56, 1., -1.56, 1.56, 1., -1.56, 1.56, 4.;
  VectorXs g = VectorXs::Zero(3 * nc);
  g(2) = -0.00981;
  g(5) = -0.00981;
  g(8) = -0.00981;
  g(11) = -0.00981;
  MatrixXs M = MatrixXs::Zero(nv, nv);
  M.diagonal() << 1., 1., 1., .00666667, .00666667, .00666667;
  VectorXs dqf = VectorXs::Zero(nv);
  dqf(2) = -0.00981;
  VectorXs vstar = VectorXs::Zero(3 * nc);
  MatrixXs J = MatrixXs::Zero(3 * nc, nv);
  J << 1., 0., 0., 0., -0.104, 0.1, 0., 1., 0., 0.104, 0., 0.1, 0., 0., 1.,
      -0.1, -0.1, 0., 1., 0., 0., 0., -0.104, -0.1, 0., 1., 0., 0.104, 0., 0.1,
      0., 0., 1., 0.1, -0.1, 0., 1., 0., 0., 0., -0.104, 0.1, 0., 1., 0., 0.104,
      0., -0.1, 0., 0., 1., -0.1, 0.1, 0., 1., 0., 0., 0., -0.104, -0.1, 0., 1.,
      0., 0.104, 0., -0.1, 0., 0., 1., 0.1, 0.1, 0.;
  std::vector<T> mus = {0.95, .95, .95, .95};
  cb::ContactProblem<T, cb::IceCreamCone> prob(G, g, M, J, dqf, vstar, mus);
  cb::CCPADMMPrimalSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 200;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  VectorXs lam = solver.getDualSolution();
  VectorXs sol = VectorXs::Zero(3 * nc);
  sol(2) = 2.4524975e-3;
  sol(5) = 2.4524975e-3;
  sol(8) = 2.4524975e-3;
  sol(11) = 2.4524975e-3;
  BOOST_CHECK((lam).isApprox(sol, 1e-3));
  //  cube sliding on floor
  g(1) = 1.;
  g(4) = 1.;
  g(7) = 1.;
  g(10) = 1.;
  dqf(1) = 1.;
  cb::ContactProblem<T, cb::IceCreamCone> prob2(G, g, M, J, dqf, vstar, mus);
  solver.setProblem(prob2);
  solver.solve(prob2, x0, settings);
  // std::cout << " lam1 " << solver.getDualSolution() << std::endl;
  lam = solver.getDualSolution();
  // std::cout << " lam2 " << lam << std::endl;
  cb::CCPADMMSolver<T> solver2;
  solver2.setProblem(prob2);
  solver2.solve(prob2, x0, settings);
  sol = solver2.getSolution();
  BOOST_CHECK((lam).isApprox(sol, 1e-3));
}

constexpr double PI = 3.1415926535897932;

BOOST_AUTO_TEST_CASE(CCP_NewtonPrimal_solver_gradients) {
  //  ball on floor
  int nc = 1;
  int nv = 6;
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  MatrixXs M = MatrixXs::Zero(6, 6);
  M.diagonal() << 1., 1., 1., .004, .004, .004;
  VectorXs dqf = VectorXs::Zero(6);
  dqf(2) = -0.00981;
  VectorXs vstar = VectorXs::Zero(3);
  MatrixXs J = MatrixXs::Zero(3, 6);
  J << 1., 0., 0., 0., -0.1, 0., 0., 1., 0., 0.1, 0., 0., 0., 0., 1., 0., 0.,
      0.;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, M, J, dqf, vstar, mus);
  prob.Del_->computeChol(1e-6);
  prob.Del_->evaluateDel();
  VectorXs R = VectorXs::Zero(3 * nc);
  for (int i = 0; i < nc; i++) {
    R(3 * i + 2) =
        std::max(prob.Del_->G_.template block<3, 3>(3 * i, 3 * i).norm() /
                     (4. * PI * PI),
                 0.);
    R(3 * i) = 1e-3 * R(3 * i + 2);
    R(3 * i + 1) = R(3 * i);
  }
  cb::CCPNewtonPrimalSolver<T> solver;
  solver.setProblem(prob);
  solver.setCompliance(prob, 1e-6);
  VectorXs dq = VectorXs::Zero(nv);
  // VectorXs y  = VectorXs::Zero(3*nc);
  VectorXs ddq = VectorXs::Zero(nv);
  // solver.complianceMap(prob, R, prob.J_ * dq, y);
  VectorXs grad_dq_lr = VectorXs::Zero(nv);
  VectorXs grad_dq_lr_fd = VectorXs::Zero(nv);
  solver.computeRegularizationGrad(prob, R, dq, grad_dq_lr);
  double delta = 1e-6;
  for (int i = 0; i < nv; i++) {
    ddq.setZero();
    ddq(i) = delta;
    grad_dq_lr_fd(i) = (solver.regularizationCost(prob, R, dq + ddq) -
                        solver.regularizationCost(prob, R, dq - ddq)) /
                       (2 * delta);
  }
  // std::cout << " FD gradients" << grad_dq_lr_fd << std::endl;
  // std::cout << " analytc gradients" << grad_dq_lr << std::endl;
  BOOST_CHECK((grad_dq_lr_fd - grad_dq_lr).norm() < 1e-3);
}

BOOST_AUTO_TEST_CASE(CCP_NewtonPrimal_solver_gradients2) {
  //  cube on floor
  int nc = 4;
  int nv = 6;
  MatrixXs G = MatrixXs::Zero(3 * nc, 3 * nc);
  G << 4.1224, 1.5, 1.56, 1.1224, 1.5, 1.56, 4.1224, -1.5, -1.56, 1.1224, -1.5,
      -1.56, 1.5, 4.1224, -1.56, -1.5, 4.1224, 1.56, 1.5, 1.1224, -1.56, -1.5,
      1.1224, 1.56, 1.56, -1.56, 4., 1.56, -1.56, 1., 1.56, -1.56, 1., 1.56,
      -1.56, -2., 1.1224, -1.5, 1.56, 4.1224, -1.5, 1.56, 1.1224, 1.5, -1.56,
      4.1224, 1.5, -1.56, 1.5, 4.1224, -1.56, -1.5, 4.1224, 1.56, 1.5, 1.1224,
      -1.56, -1.5, 1.1224, 1.56, 1.56, 1.56, 1., 1.56, 1.56, 4., 1.56, 1.56,
      -2., 1.56, 1.56, 1., 4.1224, 1.5, 1.56, 1.1224, 1.5, 1.56, 4.1224, -1.5,
      -1.56, 1.1224, -1.5, -1.56, -1.5, 1.1224, -1.56, 1.5, 1.1224, 1.56, -1.5,
      4.1224, -1.56, 1.5, 4.1224, 1.56, -1.56, -1.56, 1., -1.56, -1.56, -2.,
      -1.56, -1.56, 4., -1.56, -1.56, 1., 1.1224, -1.5, 1.56, 4.1224, -1.5,
      1.56, 1.1224, 1.5, -1.56, 4.1224, 1.5, -1.56, -1.5, 1.1224, -1.56, 1.5,
      1.1224, 1.56, -1.5, 4.1224, -1.56, 1.5, 4.1224, 1.56, -1.56, 1.56, -2.,
      -1.56, 1.56, 1., -1.56, 1.56, 1., -1.56, 1.56, 4.;
  VectorXs g = VectorXs::Zero(3 * nc);
  g(2) = -0.00981;
  g(5) = -0.00981;
  g(8) = -0.00981;
  g(11) = -0.00981;
  MatrixXs M = MatrixXs::Zero(nv, nv);
  M.diagonal() << 1., 1., 1., .00666667, .00666667, .00666667;
  VectorXs dqf = VectorXs::Zero(nv);
  dqf(2) = -0.00981;
  VectorXs vstar = VectorXs::Zero(3 * nc);
  MatrixXs J = MatrixXs::Zero(3 * nc, nv);
  J << 1., 0., 0., 0., -0.104, 0.1, 0., 1., 0., 0.104, 0., 0.1, 0., 0., 1.,
      -0.1, -0.1, 0., 1., 0., 0., 0., -0.104, -0.1, 0., 1., 0., 0.104, 0., 0.1,
      0., 0., 1., 0.1, -0.1, 0., 1., 0., 0., 0., -0.104, 0.1, 0., 1., 0., 0.104,
      0., -0.1, 0., 0., 1., -0.1, 0.1, 0., 1., 0., 0., 0., -0.104, -0.1, 0., 1.,
      0., 0.104, 0., -0.1, 0., 0., 1., 0.1, 0.1, 0.;
  std::vector<T> mus = {0.95, .95, .95, .95};
  cb::ContactProblem<T, cb::IceCreamCone> prob(G, g, M, J, dqf, vstar, mus);
  prob.Del_->computeChol(1e-6);
  prob.Del_->evaluateDel();
  VectorXs R = VectorXs::Zero(3 * nc);
  for (int i = 0; i < nc; i++) {
    R(3 * i + 2) =
        std::max(prob.Del_->G_.template block<3, 3>(3 * i, 3 * i).norm() /
                     (4. * PI * PI),
                 0.);
    R(3 * i) = 1e-3 * R(3 * i + 2);
    R(3 * i + 1) = R(3 * i);
  }
  cb::CCPNewtonPrimalSolver<T> solver;
  solver.setProblem(prob);
  solver.setCompliance(prob, 1e-6);
  VectorXs dq = VectorXs::Zero(nv);
  // VectorXs y  = VectorXs::Zero(3*nc);
  VectorXs ddq = VectorXs::Zero(nv);
  // solver.complianceMap(prob, R, prob.J_ * dq, y);
  VectorXs grad_dq_lr = VectorXs::Zero(nv);
  VectorXs grad_dq_lr_fd = VectorXs::Zero(nv);
  solver.computeRegularizationGrad(prob, R, dq, grad_dq_lr);
  double delta = 1e-6;
  for (int i = 0; i < nv; i++) {
    ddq.setZero();
    ddq(i) = delta;
    grad_dq_lr_fd(i) = (solver.regularizationCost(prob, R, dq + ddq) -
                        solver.regularizationCost(prob, R, dq - ddq)) /
                       (2 * delta);
  }
  // std::cout << " FD gradients" << grad_dq_lr_fd << std::endl;
  // std::cout << " analytc gradients" << grad_dq_lr << std::endl;
  BOOST_CHECK((grad_dq_lr_fd - grad_dq_lr).norm() < 1e-3);
}

BOOST_AUTO_TEST_CASE(CCP_NewtonPrimal_solver_hessians) {
  //  ball on floor
  int nc = 1;
  int nv = 6;
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  MatrixXs M = MatrixXs::Zero(6, 6);
  M.diagonal() << 1., 1., 1., .004, .004, .004;
  VectorXs dqf = VectorXs::Zero(6);
  dqf(2) = -0.00981;
  VectorXs vstar = VectorXs::Zero(3);
  MatrixXs J = MatrixXs::Zero(3, 6);
  J << 1., 0., 0., 0., -0.1, 0., 0., 1., 0., 0.1, 0., 0., 0., 0., 1., 0., 0.,
      0.;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, M, J, dqf, vstar, mus);
  prob.Del_->computeChol(1e-6);
  prob.Del_->evaluateDel();
  VectorXs R = VectorXs::Zero(3 * nc);
  for (int i = 0; i < nc; i++) {
    R(3 * i + 2) =
        std::max(prob.Del_->G_.template block<3, 3>(3 * i, 3 * i).norm() /
                     (4. * PI * PI),
                 0.);
    R(3 * i) = 1e-3 * R(3 * i + 2);
    R(3 * i + 1) = R(3 * i);
  }
  cb::CCPNewtonPrimalSolver<T> solver;
  solver.setProblem(prob);
  solver.setCompliance(prob, 1e-6);
  VectorXs dq = VectorXs::Zero(nv);
  VectorXs y = VectorXs::Zero(3 * nc);
  VectorXs y_tilde = VectorXs::Zero(3 * nc);
  VectorXs dy = VectorXs::Zero(3 * nc);
  VectorXs dy2 = VectorXs::Zero(3 * nc);
  solver.complianceMap(prob, R, prob.J_ * dq, y);
  y_tilde = (R.cwiseSqrt().array() * y.array()).matrix();
  MatrixXs hess_y_lr = MatrixXs::Zero(3 * nc, 3 * nc);
  MatrixXs hess_y_lr_fd = MatrixXs::Zero(3 * nc, 3 * nc);
  solver.computeHessReg(prob, R, y, y_tilde, hess_y_lr);
  double delta = 1e-6;
  for (int i = 0; i < 3 * nc; i++) {
    for (int j = 0; j < 3 * nc; j++) {
      dy.setZero();
      dy(i) = delta;
      dy2.setZero();
      dy2(j) = delta;
      hess_y_lr_fd(i, j) =
          (solver.regularizationCost(prob, R, dq, y + dy + dy2) -
           solver.regularizationCost(prob, R, dq, y + dy) -
           solver.regularizationCost(prob, R, dq, y + dy2) +
           solver.regularizationCost(prob, R, dq, y)) /
          (delta * delta);
    }
  }
  // std::cout << " FD hess" << hess_y_lr_fd << std::endl;
  // std::cout << " analytc hess" << hess_y_lr << std::endl;
  // BOOST_CHECK((hess_y_lr).isApprox(hess_y_lr_fd, 1e-3));
}

BOOST_AUTO_TEST_CASE(CCP_NewtonPrimal_solver_getSolution) {
  // std::cout << "CCP_NewtonPrimal_solver_getSolution" << std::endl;
  //  ball on floor
  int nc = 1;
  int nv = 6;
  CONTACTBENCH_UNUSED(nv);
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  MatrixXs M = MatrixXs::Zero(6, 6);
  M.diagonal() << 1., 1., 1., .004, .004, .004;
  VectorXs dqf = VectorXs::Zero(6);
  dqf(2) = -0.00981;
  VectorXs vstar = VectorXs::Zero(3);
  MatrixXs J = MatrixXs::Zero(3, 6);
  J << 1., 0., 0., 0., -0.1, 0., 0., 1., 0., 0.1, 0., 0., 0., 0., 1., 0., 0.,
      0.;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, M, J, dqf, vstar, mus);
  prob.Del_->computeChol(1e-6);
  prob.Del_->evaluateDel();
  VectorXs R = VectorXs::Zero(3 * nc);
  for (int i = 0; i < nc; i++) {
    R(3 * i + 2) =
        std::max(prob.Del_->G_.template block<3, 3>(3 * i, 3 * i).norm() /
                     (4. * PI * PI),
                 0.);
    R(3 * i) = 1e-3 * R(3 * i + 2);
    R(3 * i + 1) = R(3 * i);
  }
  cb::CCPNewtonPrimalSolver<T> solver;
  // std::cout << "entering set" << std::endl;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  int maxIter = 200;
  // std::cout << "entering solve" << std::endl;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  VectorXs lam = solver.getDualSolution();
  // std::cout << " lam " << lam << std::endl;
  VectorXs v =
      (prob.Del_->G_ * lam) + (R.array() * lam.array()).matrix() + prob.g_;
  // std::cout << " v " << v << std::endl;
  // std::cout << " comp " << lam.dot(v) << std::endl;
  BOOST_CHECK(lam.dot(v) < 1e-4);
}

BOOST_AUTO_TEST_CASE(CCP_NewtonPrimalSolver_getSolution2) {
  // std::cout << "CCP_NewtonPrimalSolver_getSolution2" << std::endl;
  //  cube on floor
  // std::cout << " static cube " << std::endl;
  int nc = 4;
  int nv = 6;
  MatrixXs G = MatrixXs::Zero(3 * nc, 3 * nc);
  G << 4.1224, 1.5, 1.56, 1.1224, 1.5, 1.56, 4.1224, -1.5, -1.56, 1.1224, -1.5,
      -1.56, 1.5, 4.1224, -1.56, -1.5, 4.1224, 1.56, 1.5, 1.1224, -1.56, -1.5,
      1.1224, 1.56, 1.56, -1.56, 4., 1.56, -1.56, 1., 1.56, -1.56, 1., 1.56,
      -1.56, -2., 1.1224, -1.5, 1.56, 4.1224, -1.5, 1.56, 1.1224, 1.5, -1.56,
      4.1224, 1.5, -1.56, 1.5, 4.1224, -1.56, -1.5, 4.1224, 1.56, 1.5, 1.1224,
      -1.56, -1.5, 1.1224, 1.56, 1.56, 1.56, 1., 1.56, 1.56, 4., 1.56, 1.56,
      -2., 1.56, 1.56, 1., 4.1224, 1.5, 1.56, 1.1224, 1.5, 1.56, 4.1224, -1.5,
      -1.56, 1.1224, -1.5, -1.56, -1.5, 1.1224, -1.56, 1.5, 1.1224, 1.56, -1.5,
      4.1224, -1.56, 1.5, 4.1224, 1.56, -1.56, -1.56, 1., -1.56, -1.56, -2.,
      -1.56, -1.56, 4., -1.56, -1.56, 1., 1.1224, -1.5, 1.56, 4.1224, -1.5,
      1.56, 1.1224, 1.5, -1.56, 4.1224, 1.5, -1.56, -1.5, 1.1224, -1.56, 1.5,
      1.1224, 1.56, -1.5, 4.1224, -1.56, 1.5, 4.1224, 1.56, -1.56, 1.56, -2.,
      -1.56, 1.56, 1., -1.56, 1.56, 1., -1.56, 1.56, 4.;
  VectorXs g = VectorXs::Zero(3 * nc);
  g(2) = -0.00981;
  g(5) = -0.00981;
  g(8) = -0.00981;
  g(11) = -0.00981;
  MatrixXs M = MatrixXs::Zero(nv, nv);
  M.diagonal() << 1., 1., 1., .00666667, .00666667, .00666667;
  VectorXs dqf = VectorXs::Zero(nv);
  dqf(2) = -0.00981;
  VectorXs vstar = VectorXs::Zero(3 * nc);
  MatrixXs J = MatrixXs::Zero(3 * nc, nv);
  J << 1., 0., 0., 0., -0.104, 0.1, 0., 1., 0., 0.104, 0., 0.1, 0., 0., 1.,
      -0.1, -0.1, 0., 1., 0., 0., 0., -0.104, -0.1, 0., 1., 0., 0.104, 0., 0.1,
      0., 0., 1., 0.1, -0.1, 0., 1., 0., 0., 0., -0.104, 0.1, 0., 1., 0., 0.104,
      0., -0.1, 0., 0., 1., -0.1, 0.1, 0., 1., 0., 0., 0., -0.104, -0.1, 0., 1.,
      0., 0.104, 0., -0.1, 0., 0., 1., 0.1, 0.1, 0.;
  std::vector<T> mus = {0.95, .95, .95, .95};
  cb::ContactProblem<T, cb::IceCreamCone> prob(G, g, M, J, dqf, vstar, mus);
  cb::CCPNewtonPrimalSolver<T> solver;
  solver.setProblem(prob);
  prob.Del_->computeChol(1e-6);
  prob.Del_->evaluateDel();
  VectorXs R = VectorXs::Zero(3 * nc);
  for (int i = 0; i < nc; i++) {
    R(3 * i + 2) =
        std::max(prob.Del_->G_.template block<3, 3>(3 * i, 3 * i).norm() /
                     (4. * PI * PI),
                 0.);
    R(3 * i) = 1e-3 * R(3 * i + 2);
    R(3 * i + 1) = R(3 * i);
  }
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 200;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  settings.statistics_ = true;
  solver.solve(prob, x0, settings, 1e-6);
  // VectorXs sol = VectorXs::Zero(3 * nc);
  // sol(2) = 2.4524975e-3;
  // sol(5) = 2.4524975e-3;
  // sol(8) = 2.4524975e-3;
  // sol(11) = 2.4524975e-3;
  VectorXs lam = solver.getDualSolution();
  // std::cout << " lam " << lam << std::endl;
  VectorXs v =
      (prob.Del_->G_ * lam) + (R.array() * lam.array()).matrix() + prob.g_;
  // std::cout << " v " << v << std::endl;
  // std::cout << " comp " << lam.dot(v) << std::endl;
  BOOST_CHECK(lam.dot(v) < 1e-4);
  // BOOST_CHECK((lam).isApprox(sol, 1e-3));
  //  cube sliding on floor
  // std::cout << " sliding cube " << std::endl;
  g(1) = 1.;
  g(4) = 1.;
  g(7) = 1.;
  g(10) = 1.;
  dqf(1) = 1.;
  cb::ContactProblem<T, cb::IceCreamCone> prob2(G, g, M, J, dqf, vstar, mus);
  solver.setProblem(prob2);
  solver.solve(prob2, x0, settings, 1e-6);
  lam = solver.getDualSolution();
  // std::cout << " stop " << solver.stop_ << std::endl;
  // std::cout << " n iter  " << solver.n_iter_ << std::endl;
  // std::cout << " solver stats comp  " << std::endl;
  // for (T compi : solver.stats_.comp_) {
  //   std::cout << compi << std::endl;
  // }
  // std::cout << " lam " << lam << std::endl;
  v = (prob2.Del_->G_ * lam) + (R.array() * lam.array()).matrix() + prob2.g_;
  // std::cout << " v " << v << std::endl;
  // std::cout << " comp " << lam.dot(v) << std::endl;
  BOOST_CHECK(lam.dot(v) < 1e-4);
}

BOOST_AUTO_TEST_CASE(CCP_NewtonPrimal_solver_getSolution3) {
  // std::cout << "CCP_ADMMPrimal_solver_getSolution2" << std::endl;
  //  ball in allegro hand
  int nc = 3;
  int nv = 6;
  MatrixXs G = MatrixXs::Zero(3 * nc, 3 * nc);
  G << 3.54487830e-01, -1.56125113e-17, 2.61292724e-17, -9.70611328e-02,
      -1.85088901e-01, -8.90919316e-02, 1.14699643e-01, -7.02513097e-03,
      -1.06622888e-02, -6.93889390e-18, 3.54487830e-01, -2.25514052e-17,
      4.10505520e-02, 5.45144842e-02, 2.87804517e-02, -3.49983171e-02,
      2.55919028e-01, 9.93151662e-02, 2.98155597e-17, -3.55618313e-17,
      1.00000000e-01, -4.47572434e-02, -8.22342741e-02, -3.51327955e-02,
      1.81067978e-02, -9.82310059e-02, 4.77633348e-03, -9.70611328e-02,
      4.10505520e-02, -4.47572434e-02, 3.54983658e-01, 6.24500451e-17,
      -1.24900090e-16, 2.44785519e-01, 7.15439467e-02, 7.11600891e-02,
      -1.85088901e-01, 5.45144842e-02, -8.22342741e-02, 6.93889390e-17,
      3.54983658e-01, -5.89805982e-17, -1.50344004e-01, 1.85587565e-01,
      -6.00911948e-02, -8.90919316e-02, 2.87804517e-02, -3.51327955e-02,
      -1.28369537e-16, -5.89805982e-17, 1.00000000e-01, -9.07855696e-02,
      2.08012100e-02, 3.64045330e-02, 1.14699643e-01, -3.49983171e-02,
      1.81067978e-02, 2.44785519e-01, -1.50344004e-01, -9.07855696e-02,
      3.54971923e-01, 6.93889390e-18, -1.56125113e-17, -7.02513097e-03,
      2.55919028e-01, -9.82310059e-02, 7.15439467e-02, 1.85587565e-01,
      2.08012100e-02, 6.93889390e-18, 3.54971923e-01, 1.21430643e-16,
      -1.06622888e-02, 9.93151662e-02, 4.77633348e-03, 7.11600891e-02,
      -6.00911948e-02, 3.64045330e-02, -1.56125113e-17, 1.21430643e-16,
      1.00000000e-01;
  VectorXs g = VectorXs::Zero(3 * nc);
  g << 0.00165058, -0.00281272, -0.01007722, 0.00732922, 0.02622216,
      -0.00979829, -0.00032159, 0.00971973, -0.00294113;
  MatrixXs M = MatrixXs::Zero(nv, nv);
  M << 1.00000000e+01, 0.00000000e+00, -4.51028104e-17, 0.00000000e+00,
      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+01,
      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
      -4.51028104e-17, 0.00000000e+00, 1.00000000e+01, 0.00000000e+00,
      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
      0.00000000e+00, 1.00000000e-02, -2.71050543e-20, 1.11808349e-19,
      0.00000000e+00, 0.00000000e+00, 0.00000000e+00, -2.71050543e-20,
      1.00000000e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
      0.00000000e+00, 1.11808349e-19, 0.00000000e+00, 1.00000000e-02;
  VectorXs dqf = VectorXs::Zero(nv);
  dqf << 0.01418917, 0.0018922, -0.01088471, -0.06665827, 0.29799905,
      -0.19428938;
  VectorXs vstar = VectorXs::Zero(3 * nc);
  vstar << -0.00000000e+00, -0.00000000e+00, 8.93573682e-05, -0.00000000e+00,
      -0.00000000e+00, 9.91813224e-05, -0.00000000e+00, -0.00000000e+00,
      9.89489163e-05;
  MatrixXs J = MatrixXs::Zero(3 * nc, nv);
  J << 9.96397344e-01, 6.93355535e-02, -4.88355881e-02, 2.99043575e-03,
      -4.94988086e-02, -9.26302045e-03, -5.92790133e-02, 9.81208352e-01,
      1.83619632e-01, 5.02650444e-02, 3.49775589e-03, -2.46359850e-03,
      6.06492557e-02, -1.80063188e-01, 9.81783538e-01, -6.93889390e-18,
      -3.46944695e-18, -3.14418630e-18, 3.54640809e-01, 8.79834458e-01,
      -3.16419379e-01, -6.50772313e-03, 1.92481898e-02, 4.62276250e-02,
      1.28876251e-01, -3.81183170e-01, -9.15472721e-01, 1.79079092e-02,
      4.44280386e-02, -1.59778834e-02, -9.26078187e-01, 2.83885043e-01,
      -2.48572874e-01, 8.67361738e-19, -2.08166817e-17, -1.49077799e-17,
      9.80480833e-01, 1.29814147e-01, 1.47667270e-01, -5.61006120e-03,
      -1.28175722e-02, 4.85175874e-02, 1.11101883e-01, 2.53839726e-01,
      -9.60844298e-01, 4.95091292e-02, 6.55493220e-03, 7.45642108e-03,
      -1.62215002e-01, 9.58495530e-01, 2.34462390e-01, 2.08166817e-17,
      3.46944695e-18, 1.73472348e-18;
  std::vector<T> mus = {0.9, .9, .9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(G, g, M, J, dqf, vstar, mus);
  VectorXs R = VectorXs::Zero(3 * nc);
  for (int i = 0; i < nc; i++) {
    R(3 * i + 2) =
        std::max(prob.Del_->G_.template block<3, 3>(3 * i, 3 * i).norm() /
                     (4. * PI * PI),
                 0.);
    R(3 * i) = 1e-3 * R(3 * i + 2);
    R(3 * i + 1) = R(3 * i);
  }
  cb::CCPNewtonPrimalSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 10000;
  double eps_abs = 1e-6;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = eps_abs;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  VectorXs lam = solver.getDualSolution();
  VectorXs sol = VectorXs::Zero(3 * nc);
}

BOOST_AUTO_TEST_CASE(Raisim_solver_init) {
  cb::RaisimSolver<T> solver;
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(Raisim_solver_setProblem) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(Raisim_solver_computeGinv) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  solver.computeGinv(A);
  BOOST_CHECK(!std::isnan(solver.getGinv(0).sum()));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_getAinv) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  solver.computeGinv(A);
  Eigen::Matrix3d Ainv = solver.getGinv(0);
  BOOST_CHECK(!std::isnan(Ainv.sum()));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_computeGlam) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs lam = VectorXs::Zero(3);
  lam(2) = 1.;
  solver.computeGlam(A, lam);
  BOOST_CHECK(!std::isnan(solver.getGlam(0, 0).sum()));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_updateAlam) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs lam = VectorXs::Zero(3);
  solver.computeGlam(A, lam);
  Eigen::Vector3d lamj = Eigen::Vector3d::Zero();
  solver.updateGlam(0, A, lamj);
  BOOST_CHECK(!std::isnan(solver.getGlam(0, 0).sum()));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_getAlam) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs lam = VectorXs::Zero(3);
  solver.computeGlam(A, lam);
  Eigen::Vector3d Alamij = solver.getGlam(0, 0);
  BOOST_CHECK(!std::isnan(Alamij.sum()));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_setAlam) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs lam = VectorXs::Zero(3);
  solver.computeGlam(A, lam);
  Eigen::Vector3d Alamij = Eigen::Vector3d::Zero();
  solver.setGlam(0, 0, Alamij);
  BOOST_CHECK(!std::isnan(solver.getGlam(0, 0).sum()));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_computeC) {
  // std::cout << "Raisim_solver_computeC" << std::endl;
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs lam = VectorXs::Zero(3);
  solver.computeC(A, b, lam);
  BOOST_CHECK(!std::isnan(solver.getC(0).sum()));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_updateC) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs lam = VectorXs::Zero(3);
  solver.computeC(A, b, lam);
  Eigen::Vector3d lamj = Eigen::Vector3d::Zero();
  solver.updateC(0, A, b, lamj);
  BOOST_CHECK(!std::isnan(solver.getC(0).sum()));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_getC) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs lam = VectorXs::Zero(3);
  solver.computeC(A, b, lam);
  Eigen::Vector3d ci = solver.getC(0);
  BOOST_CHECK(!std::isnan(ci.sum()));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_computeLamV0) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  Eigen::Vector3d lam = Eigen::Vector3d::Zero(3);
  Eigen::Vector3d lam_out = Eigen::Vector3d::Zero();
  solver.computeGinv(A);
  solver.computeC(A, b, lam);
  Eigen::Matrix3d Ainvi = solver.getGinv(0);
  Eigen::Vector3d ci = solver.getC(0);
  solver.computeLamV0(Ainvi, ci, lam_out);
  BOOST_CHECK(!std::isnan(lam_out.sum()));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_computeH1Grad) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  Eigen::Vector3d grad_out = Eigen::Vector3d::Zero();
  solver.computeH1Grad(A, grad_out);
  BOOST_CHECK(!std::isnan(grad_out.sum()));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_computeH2Grad) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs lam = VectorXs::Zero(3);
  Eigen::Vector3d grad_out = Eigen::Vector3d::Zero();
  solver.computeH2Grad(mus[0], lam, grad_out);
  BOOST_CHECK(!std::isnan(grad_out.sum()));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_computeEta) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs lam = VectorXs::Zero(3);
  Eigen::Vector3d eta_out = Eigen::Vector3d::Zero();
  solver.computeEta(A, mus[0], lam, eta_out);
  BOOST_CHECK(!std::isnan(eta_out.sum()));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_computeGbar) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  Eigen::Matrix<T, 3, 2> Abar = Eigen::Matrix<T, 3, 2>::Zero();
  solver.computeGbar(A, Abar);
  BOOST_CHECK(!std::isnan(Abar.sum()));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_computeCbar) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  Eigen::Vector3d cbar = Eigen::Vector3d::Zero();
  solver.computeCbar(A, b, cbar);
  BOOST_CHECK(!std::isnan(cbar.sum()));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_computeGradTheta) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs lam = VectorXs::Zero(3);
  T grad_theta = solver.computeGradTheta(A, b, mus[0], lam);
  BOOST_CHECK(!std::isnan(grad_theta));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_computeTheta) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  T theta = solver.computeTheta(x0);
  BOOST_CHECK(!std::isnan(theta));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_computeR) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  T theta = 0.;
  T r = solver.computeR(A, b, mus[0], theta);
  BOOST_CHECK(!std::isnan(r));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_computeLamZ) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  T r = 1.;
  T lamz = solver.computeLamZ(mus[0], r);
  BOOST_CHECK(!std::isnan(lamz));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_computeLam) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  T r = 1.;
  T theta = 0.;
  T lamZ = 1.;
  VectorXs lam = VectorXs::Zero(3);
  solver.computeLam(r, theta, lamZ, lam);
  BOOST_CHECK(!std::isnan(lam.sum()));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_bisection) {
  MatrixXs A = MatrixXs::Random(3, 3);
  VectorXs b = VectorXs::Zero(3);
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  int maxIter = 10;
  CONTACTBENCH_UNUSED(maxIter);
  solver.computeGinv(A);
  Eigen::Vector3d lam = Eigen::Vector3d::Zero(3);
  solver.computeC(A, b, lam);
  Eigen::Vector3d lam_v0;
  solver.computeLamV0(solver.getGinv(0), solver.getC(0), lam_v0);
  solver.bisectionStep(A, solver.getGinv(0), solver.getC(0), mus[0], lam_v0,
                       lam);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(Raisim_solver_solve) {
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  int maxIter = 10;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(Raisim_solver_getSolution) {
  MatrixXs A = MatrixXs::Zero(3, 3);
  A(0, 0) = 3.5;
  A(1, 1) = 3.5;
  A(2, 2) = 1.;
  VectorXs b = VectorXs::Zero(3);
  b(2) = -0.00981;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3);
  int maxIter = 200;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  VectorXs lam = solver.getSolution();
  BOOST_CHECK((lam).isApprox(-b, 1e-3));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_getSolution2) {
  MatrixXs A = MatrixXs::Zero(6, 6);
  A << 0.00746183, 0.01030469, -0.02459814, 0.00146251, 0.0106132, -0.02493933,
      0.01030469, 0.05200239, 0.00350998, 0.00977178, 0.05230589, 0.00358581,
      -0.02459814, 0.00350998, 0.11827849, 0.00287095, 0.00237138, 0.11994601,
      0.00146251, 0.00977178, 0.00287095, 0.00750077, 0.00957129, 0.00312712,
      0.0106132, 0.05230589, 0.00237138, 0.00957129, 0.05263244, 0.00242249,
      -0.02493933, 0.00358581, 0.11994601, 0.00312712, 0.00242249, 0.12164495;
  VectorXs b = VectorXs::Zero(6);
  b << 5.16331117e-01, -3.47716091e-02, -2.44411637e+00, 2.49915611e-01,
      -2.36700375e-02, -2.46724553e+00;
  std::vector<T> mus = {0.9, 0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(6);
  int maxIter = 200;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  settings.statistics_ = true;
  solver.solve(prob, x0, settings, 0., 1., .7, 1e-2, .5, 1.3, 0.99, 1e-3);
  VectorXs lam = solver.getSolution();
  VectorXs v = A * lam + b;
  VectorXs sol = VectorXs::Zero(6);
  double comp = prob.computeContactComplementarity(lam);
  BOOST_CHECK(!std::isnan(comp));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_getSolution3) {
  int nc = 4;
  MatrixXs A = MatrixXs::Zero(12, 12);
  A << 4.1224, 1.5, 1.56, 1.1224, 1.5, 1.56, 4.1224, -1.5, -1.56, 1.1224, -1.5,
      -1.56, 1.5, 4.1224, -1.56, -1.5, 4.1224, 1.56, 1.5, 1.1224, -1.56, -1.5,
      1.1224, 1.56, 1.56, -1.56, 4., 1.56, -1.56, 1., 1.56, -1.56, 1., 1.56,
      -1.56, -2., 1.1224, -1.5, 1.56, 4.1224, -1.5, 1.56, 1.1224, 1.5, -1.56,
      4.1224, 1.5, -1.56, 1.5, 4.1224, -1.56, -1.5, 4.1224, 1.56, 1.5, 1.1224,
      -1.56, -1.5, 1.1224, 1.56, 1.56, 1.56, 1., 1.56, 1.56, 4., 1.56, 1.56,
      -2., 1.56, 1.56, 1., 4.1224, 1.5, 1.56, 1.1224, 1.5, 1.56, 4.1224, -1.5,
      -1.56, 1.1224, -1.5, -1.56, -1.5, 1.1224, -1.56, 1.5, 1.1224, 1.56, -1.5,
      4.1224, -1.56, 1.5, 4.1224, 1.56, -1.56, -1.56, 1., -1.56, -1.56, -2.,
      -1.56, -1.56, 4., -1.56, -1.56, 1., 1.1224, -1.5, 1.56, 4.1224, -1.5,
      1.56, 1.1224, 1.5, -1.56, 4.1224, 1.5, -1.56, -1.5, 1.1224, -1.56, 1.5,
      1.1224, 1.56, -1.5, 4.1224, -1.56, 1.5, 4.1224, 1.56, -1.56, 1.56, -2.,
      -1.56, 1.56, 1., -1.56, 1.56, 1., -1.56, 1.56, 4.;
  VectorXs b = VectorXs::Zero(12);
  b << 0., 3., -0.00981, 0., 3., -0.00981, 0., 3., -0.00981, 0., 3., -0.00981;
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(12);
  x0 << 0.0, -0.000098039, 0.0001089322, 0.0, -0.00431635, 0.004795953, 0.0,
      -0.0001843271, 0.00020480, 0.0, -0.0042301, 0.004700119;
  VectorXs v0 = A * x0 + b;
  prob.Del_->evaluateDel();
  solver.computeGinv(prob.Del_->G_);
  solver.computeC(prob.Del_->G_, prob.g_, x0);
  Vector3s lam_v0, c_j, c_j2, lam_star;
  Vector2s lam_star_t_cor, v_star_t;
  Matrix3s G_j, Ginv_j;
  int maxIter = 200;
  double th = 1e-8;
  double beta1 = 1e-2;
  double beta2 = 0.5;
  double beta3 = 1.3;
  for (int i = 0; i < nc; i++) {
    c_j = solver.getC(i);
    G_j = prob.Del_->G_.block<3, 3>(3 * i, 3 * i);
    Ginv_j = solver.getGinv(i);
    solver.computeLamV0(Ginv_j, c_j, lam_v0);
    c_j2 = c_j + G_j * lam_v0;
    BOOST_CHECK(c_j2.norm() < 1e-6);
    solver.bisectionStep(G_j, Ginv_j, c_j, mus[CAST_UL(i)], lam_v0, lam_star,
                         maxIter, th, beta1, beta2, beta3);
    BOOST_CHECK(
        prob.contact_constraints_[CAST_UL(i)].isOnBorder(lam_star, 1e-5));
    c_j2 = c_j + G_j * lam_star;
    BOOST_CHECK(std::abs(c_j2(2)) < 1e-5);
    v_star_t = c_j2.head<2>();
    v_star_t.normalize();
    lam_star_t_cor = -lam_star.head<2>();
    lam_star_t_cor +=
        (-(mus[CAST_UL(i)] * mus[CAST_UL(i)] * lam_star(2) / G_j(2, 2)) *
         G_j.row(2).head<2>());
    lam_star_t_cor.normalize();
    BOOST_CHECK(v_star_t.isApprox(lam_star_t_cor, 1e-4));
  }
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings, 0., 1., 0.1, 1e-2, .5, 1.3, 0.99, 1e-6);
  VectorXs lam = solver.getSolution();
  VectorXs v = A * lam + b;
  double comp = prob.computeContactComplementarity(lam);
  CONTACTBENCH_UNUSED(comp);
  // BOOST_CHECK(std::abs(comp) < 1e-5);
  VectorXs lam_t_cor = VectorXs::Zero(2 * nc);
  VectorXs lam_t = VectorXs::Zero(2 * nc);
  VectorXs v_t = VectorXs::Zero(2 * nc);
  for (int i = 0; i < nc; i++) {
    lam_t_cor.segment<2>(2 * i) = A.row(3 * i + 2).segment<2>(3 * i);
    lam_t_cor.segment<2>(2 * i) *= (-(mus[CAST_UL(i)] * mus[CAST_UL(i)]) *
                                    lam(3 * i + 2) / A(3 * i + 2, 3 * i + 2));
    lam_t_cor.segment<2>(2 * i) += -lam.segment<2>(3 * i);
    lam_t_cor.segment<2>(2 * i).normalize();
    lam_t.segment<2>(2 * i) = -lam.segment<2>(3 * i);
    lam_t.segment<2>(2 * i).normalize();
    v_t.segment<2>(2 * i) = v.segment<2>(3 * i);
    v_t.segment<2>(2 * i).normalize();
  }
  // BOOST_CHECK(v_t.tail<2>().isApprox(lam_t_cor.tail<2>(), 1e-4));
}

BOOST_AUTO_TEST_CASE(Raisim_solver_getSolution4) {
  int nc = 4;
  MatrixXs A = MatrixXs::Zero(12, 12);
  A << 1.70260293, -0.47810656, -0.81113264, 0.87470029, -0.47808302,
      -0.81160892, 1.70263763, 0.4774236, 0.81159737, 0.87473499, 0.47744715,
      0.81112109, -0.47810656, 2.40658732, -1.16458464, 0.46986558, 2.40900655,
      1.15727014, -0.47813832, 1.3144438, -1.16816273, 0.46983382, 1.31686302,
      1.15369205, -0.81113264, -1.16458464, 2.07287338, -0.80996834,
      -1.16653817, 0.21909843, -0.81113907, -1.16749827, 0.58960997,
      -0.80997476, -1.1694518, -1.26416498, 0.87470029, 0.46986558, -0.80996834,
      1.69128967, 0.46984242, -0.80943495, 0.87467992, -0.46921415, 0.80941869,
      1.6912693, -0.4692373, 0.80995208, -0.47808302, 2.40900655, -1.16653817,
      0.46984242, 2.41142987, 1.15920541, -0.47811477, 1.31691684, -1.17011623,
      0.46981067, 1.31934017, 1.15562734, -0.81160892, 1.15727014, 0.21909843,
      -0.80943495, 1.15920541, 2.0556279, -0.81161538, 1.15321151, -1.26423198,
      -0.80944141, 1.15514678, 0.57229749, 1.70263763, -0.47813832, -0.81113907,
      0.87467992, -0.47811477, -0.81161538, 1.70267234, 0.47745531, 0.81160383,
      0.87471463, 0.47747886, 0.81112752, 0.4774236, 1.3144438, -1.16749827,
      -0.46921415, 1.31691684, 1.15321151, 0.47745531, 2.40507282, -1.16409878,
      -0.46918244, 2.40754587, 1.156611, 0.81159737, -1.16816273, 0.58960997,
      0.80941869, -1.17011623, -1.26423198, 0.81160383, -1.16409878, 2.07291549,
      0.80942515, -1.16605229, 0.21907354, 0.87473499, 0.46983382, -0.80997476,
      1.6912693, 0.46981067, -0.80944141, 0.87471463, -0.46918244, 0.80942515,
      1.69124894, -0.46920559, 0.80995851, 0.47744715, 1.31686302, -1.1694518,
      -0.4692373, 1.31934017, 1.15514678, 0.47747886, 2.40754587, -1.16605229,
      -0.46920559, 2.41002301, 1.15854629, 0.81112109, 1.15369205, -1.26416498,
      0.80995208, 1.15562734, 0.57229749, 0.81112752, 1.156611, 0.21907354,
      0.80995851, 1.15854629, 2.05553602;
  VectorXs b = VectorXs::Zero(12);
  b << 0.00554428, 0.00582766, -0.00701578, 0.00820943, 0.00582485, -0.0096867,
      0.00554418, 0.00275529, -0.00877465, 0.00820934, 0.00275248, -0.01144557;
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(12);
  x0 << -9.41930386e-05, -1.19316259e-04, 1.68906060e-04, -1.19463177e-04,
      -6.29330529e-04, 7.11743091e-04, -6.79642646e-03, 4.00495120e-03,
      8.76518412e-03, -4.56331805e-03, -2.79577045e-03, 5.94628258e-03;
  VectorXs v0 = A * x0 + b;
  prob.Del_->evaluateDel();
  solver.computeGinv(prob.Del_->G_);
  solver.computeC(prob.Del_->G_, prob.g_, x0);
  Vector3s lam_v0, c_j, c_j2, lam_star;
  Vector2s lam_star_t_cor, v_star_t;
  Matrix3s G_j, Ginv_j;
  int maxIter = 10000;
  double th = 1e-8;
  double beta1 = 1e-2;
  double beta2 = 0.5;
  double beta3 = 1.3;
  for (int i = 0; i < nc; i++) {
    c_j = solver.getC(i);
    G_j = prob.Del_->G_.block<3, 3>(3 * i, 3 * i);
    Ginv_j = solver.getGinv(i);
    solver.computeLamV0(Ginv_j, c_j, lam_v0);
    c_j2 = c_j + G_j * lam_v0;
    BOOST_CHECK(c_j2.norm() < 1e-6);
    solver.bisectionStep(G_j, Ginv_j, c_j, mus[CAST_UL(i)], lam_v0, lam_star,
                         maxIter, th, beta1, beta2, beta3);
    BOOST_CHECK(
        prob.contact_constraints_[CAST_UL(i)].isOnBorder(lam_star, 1e-5));
    c_j2 = c_j + G_j * lam_star;
    BOOST_CHECK(std::abs(c_j2(2)) < 1e-5);
    v_star_t = c_j2.head<2>();
    v_star_t.normalize();
    lam_star_t_cor = -lam_star.head<2>();
    lam_star_t_cor +=
        (-(mus[CAST_UL(i)] * mus[CAST_UL(i)] * lam_star(2) / G_j(2, 2)) *
         G_j.row(2).head<2>());
    lam_star_t_cor.normalize();
    // BOOST_CHECK(v_star_t.isApprox(lam_star_t_cor, 1e-4));
  }
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  settings.statistics_ = true;
  solver.solve(prob, x0, settings, 0., 1., 0.1, 1e-2, .5, 1.3, 0.99, 1e-6);
  VectorXs lam = solver.getSolution();
  VectorXs v = A * lam + b;
  double comp = prob.computeContactComplementarity(lam);
  CONTACTBENCH_UNUSED(comp);
  // BOOST_CHECK(std::abs(comp) < 1e-5);
  VectorXs lam_t_cor = VectorXs::Zero(2 * nc);
  VectorXs lam_t = VectorXs::Zero(2 * nc);
  VectorXs v_t = VectorXs::Zero(2 * nc);
  for (int i = 0; i < nc; i++) {
    lam_t_cor.segment<2>(2 * i) = A.row(3 * i + 2).segment<2>(3 * i);
    lam_t_cor.segment<2>(2 * i) *= (-(mus[CAST_UL(i)] * mus[CAST_UL(i)]) *
                                    lam(3 * i + 2) / A(3 * i + 2, 3 * i + 2));
    lam_t_cor.segment<2>(2 * i) += -lam.segment<2>(3 * i);
    lam_t_cor.segment<2>(2 * i).normalize();
    lam_t.segment<2>(2 * i) = -lam.segment<2>(3 * i);
    lam_t.segment<2>(2 * i).normalize();
    v_t.segment<2>(2 * i) = v.segment<2>(3 * i);
    v_t.segment<2>(2 * i).normalize();
  }
  // BOOST_CHECK(v_t.tail<2>().isApprox(lam_t_cor.tail<2>(), 1e-4));
}

BOOST_AUTO_TEST_CASE(Raisim_corrected_solver_getSolution) {
  int nc = 4;
  CONTACTBENCH_UNUSED(nc);
  MatrixXs A = MatrixXs::Zero(12, 12);
  A << 4.1224, 1.5, 1.56, 1.1224, 1.5, 1.56, 4.1224, -1.5, -1.56, 1.1224, -1.5,
      -1.56, 1.5, 4.1224, -1.56, -1.5, 4.1224, 1.56, 1.5, 1.1224, -1.56, -1.5,
      1.1224, 1.56, 1.56, -1.56, 4., 1.56, -1.56, 1., 1.56, -1.56, 1., 1.56,
      -1.56, -2., 1.1224, -1.5, 1.56, 4.1224, -1.5, 1.56, 1.1224, 1.5, -1.56,
      4.1224, 1.5, -1.56, 1.5, 4.1224, -1.56, -1.5, 4.1224, 1.56, 1.5, 1.1224,
      -1.56, -1.5, 1.1224, 1.56, 1.56, 1.56, 1., 1.56, 1.56, 4., 1.56, 1.56,
      -2., 1.56, 1.56, 1., 4.1224, 1.5, 1.56, 1.1224, 1.5, 1.56, 4.1224, -1.5,
      -1.56, 1.1224, -1.5, -1.56, -1.5, 1.1224, -1.56, 1.5, 1.1224, 1.56, -1.5,
      4.1224, -1.56, 1.5, 4.1224, 1.56, -1.56, -1.56, 1., -1.56, -1.56, -2.,
      -1.56, -1.56, 4., -1.56, -1.56, 1., 1.1224, -1.5, 1.56, 4.1224, -1.5,
      1.56, 1.1224, 1.5, -1.56, 4.1224, 1.5, -1.56, -1.5, 1.1224, -1.56, 1.5,
      1.1224, 1.56, -1.5, 4.1224, -1.56, 1.5, 4.1224, 1.56, -1.56, 1.56, -2.,
      -1.56, 1.56, 1., -1.56, 1.56, 1., -1.56, 1.56, 4.;
  VectorXs b = VectorXs::Zero(12);
  b << 0., 3., -0.00981, 0., 3., -0.00981, 0., 3., -0.00981, 0., 3., -0.00981;
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::RaisimCorrectedSolver<T> solver;
  solver.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(12);
  x0 << 0.0, -0.000098039, 0.0001089322, 0.0, -0.00431635, 0.004795953, 0.0,
      -0.0001843271, 0.00020480, 0.0, -0.0042301, 0.004700119;
  VectorXs v0 = A * x0 + b;
  prob.Del_->evaluateDel();
  solver.computeGinv(prob.Del_->G_);
  solver.computeC(prob.Del_->G_, prob.g_, x0);
  Vector3s lam_v0, c_j, c_j2, lam_star;
  Vector2s lam_star_t_cor, v_star_t;
  Matrix3s G_j, Ginv_j;
  int maxIter = 200;
  double th = 1e-8;
  double beta1 = 1e-2;
  double beta2 = 0.5;
  double beta3 = 1.3;
  CONTACTBENCH_UNUSED(th);
  CONTACTBENCH_UNUSED(beta2);
  CONTACTBENCH_UNUSED(beta1);
  CONTACTBENCH_UNUSED(beta3);
  cb::ContactSolverSettings<double> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-6;
  settings.rel_th_stop_ = 1e-8;
  solver.solve(prob, x0, settings, 0., 1., 0.1, 1e-2, .5, 1.3, 0.99, 1e-6);
  VectorXs lam = solver.getSolution();
  VectorXs v = A * lam + b;
  double comp = prob.computeContactComplementarity(lam);
  BOOST_CHECK(std::abs(comp) < 1e-5);
}

BOOST_AUTO_TEST_SUITE_END()
