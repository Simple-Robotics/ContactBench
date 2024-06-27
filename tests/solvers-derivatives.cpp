#include "contactbench/contact-problem.hpp"
#include "contactbench/friction-constraint.hpp"
#include "contactbench/solvers.hpp"
#include "vector"

#include <boost/test/tools/old/interface.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>

#include <iostream>
#include <proxsuite/serialization/archive.hpp>
#include <proxsuite/serialization/eigen.hpp>
#include <Eigen/Eigenvalues>

BOOST_AUTO_TEST_SUITE(SOLVERS_DERIVATIVES)

namespace cb = contactbench;
using T = double;
CONTACTBENCH_EIGEN_TYPEDEFS(T);

BOOST_AUTO_TEST_CASE(NCP_PGS_solver_vjp_fd) {
  int nc = 1;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  A << 5.47732006, -0.44641268, -0.85806775, -0.44641268, 4.73256402,
      -1.83316893, -0.85806775, -1.83316893, 2.16266878;
  VectorXs b = VectorXs::Zero(3 * nc);
  b << 4.58088011e-16, 5.88878006e-16, -9.61358750e-01;
  std::vector<T> mus = {0.95};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::NCPPGSSolver<T, cb::IceCreamCone> solver2;
  solver2.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 100;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver2.solve(prob, x0, settings);
  VectorXs lam2 = solver2.getSolution();
  // VectorXs lam2_lcp = VectorXs::Zero(6 * nc);
  // prob.computeLCPSolution(lam2, lam2_lcp);
  //
  //
  VectorXs dL_dlam = VectorXs::Zero(3 * nc);
  // solver2.vjp(prob, dL_dlam, 1e-12);
  double delta = 1e-5;
  VectorXs lam3 = VectorXs::Zero(3 * nc);
  VectorXs dlam_dmuj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dgj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dDeljk = VectorXs::Zero(3 * nc);
  VectorXs dlami_dmus = VectorXs::Zero(nc);
  VectorXs dlami_dg = VectorXs::Zero(3 * nc);
  MatrixXs dlami_dDel = MatrixXs::Zero(3 * nc, 3 * nc);
  cb::ContactProblem<T, cb::IceCreamCone> prob3;
  cb::NCPPGSSolver<T, cb::IceCreamCone> solver3;
  for (int i = 0; i < 3 * nc; i++) {
    // std::cout << "i:" << i << std::endl;
    dL_dlam.setZero();
    dL_dlam(i) = 1.;
    solver2.vjp_fd(prob, x0, dL_dlam, settings, delta);
    dlami_dmus = solver2.getdLdmus();
    dlami_dg = solver2.getdLdg();
    dlami_dDel = solver2.getdLdDel();
    for (int j = 0; j < 3 * nc; j++) {
      // std::cout << "j:" << j << std::endl;
      // test derivatives wrt mu
      if (j % 3 == 0) {
        mus[CAST_UL(j / 3)] += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::NCPPGSSolver<T, cb::IceCreamCone>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings);
        lam3 = solver3.getSolution();
        dlam_dmuj = (lam3 - lam2) / delta;
        BOOST_CHECK_SMALL(dlam_dmuj(i) - dlami_dmus(j / 3), 1e0);
        mus[CAST_UL(j / 3)] -= delta;
      }
      // test derivatives wrt g
      b(j) += delta;
      prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
      solver3 = cb::NCPPGSSolver<T, cb::IceCreamCone>();
      solver3.setProblem(prob3);
      solver3.solve(prob3, x0, settings);
      lam3 = solver3.getSolution();
      dlam_dgj = (lam3 - lam2) / delta;
      BOOST_CHECK_CLOSE(dlam_dgj(i), dlami_dg(j), 1e0);
      b(j) -= delta;
      // test derivatives wrt Del
      for (int k = 0; k < j + 1; k++) {
        // std::cout << "k:" << k << std::endl;
        // std::cout << "j:" << j << std::endl;
        A(j, k) += delta;
        A(k, j) += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::NCPPGSSolver<T, cb::IceCreamCone>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings);
        lam3 = solver3.getSolution();
        dlam_dDeljk = (lam3 - lam2) / (2 * delta);
        BOOST_CHECK_CLOSE(dlam_dDeljk(i), dlami_dDel(j, k), 1e0);
        A(j, k) -= delta;
        A(k, j) -= delta;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(LCP_QP_solver_vjp) {
  int nc = 1;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  A << 5.47732006, -0.44641268, -0.85806775, -0.44641268, 4.73256402,
      -1.83316893, -0.85806775, -1.83316893, 2.16266878;
  VectorXs b = VectorXs::Zero(3 * nc);
  b << 4.58088011e-16, 5.88878006e-16, -9.61358750e-01;
  std::vector<T> mus = {0.95};
  cb::ContactProblem<T, cb::PyramidCone> prob(A, b, mus);
  cb::LCPQPSolver<T> solver2;
  solver2.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 100;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver2.solve(prob, x0, settings);
  VectorXs lam2 = solver2.getSolution();
  VectorXs lam2_lcp = VectorXs::Zero(6 * nc);
  prob.computeLCPSolution(lam2, lam2_lcp);
  //
  //
  VectorXs dL_dlam = VectorXs::Zero(3 * nc);
  // solver2.vjp(prob, dL_dlam, 1e-12);
  double delta = 1e-5;
  VectorXs lam3 = VectorXs::Zero(3 * nc);
  VectorXs dlam_dmuj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dgj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dDeljk = VectorXs::Zero(3 * nc);
  VectorXs dlami_dmus = VectorXs::Zero(nc);
  VectorXs dlami_dg = VectorXs::Zero(3 * nc);
  MatrixXs dlami_dDel = MatrixXs::Zero(3 * nc, 3 * nc);
  cb::ContactProblem<T, cb::PyramidCone> prob3;
  cb::LCPQPSolver<T> solver3;
  for (int i = 0; i < 3 * nc; i++) {
    // std::cout << "i:" << i << std::endl;
    dL_dlam.setZero();
    dL_dlam(i) = 1.;
    solver2.vjp(prob, dL_dlam, settings);
    dlami_dmus = solver2.getdLdmus();
    dlami_dg = solver2.getdLdg();
    dlami_dDel = solver2.getdLdDel();
    for (int j = 0; j < 3 * nc; j++) {
      // std::cout << "j:" << j << std::endl;
      // test derivatives wrt mu
      if (j % 3 == 0) {
        mus[CAST_UL(j / 3)] += delta;
        prob3 = cb::ContactProblem<T, cb::PyramidCone>(A, b, mus);
        solver3 = cb::LCPQPSolver<T>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings);
        lam3 = solver3.getSolution();
        dlam_dmuj = (lam3 - lam2) / delta;
        BOOST_CHECK_SMALL(dlam_dmuj(i) - dlami_dmus(j / 3), 1e0);
        mus[CAST_UL(j / 3)] -= delta;
      }
      // test derivatives wrt g
      b(j) += delta;
      prob3 = cb::ContactProblem<T, cb::PyramidCone>(A, b, mus);
      solver3 = cb::LCPQPSolver<T>();
      solver3.setProblem(prob3);
      solver3.solve(prob3, x0, settings);
      lam3 = solver3.getSolution();
      dlam_dgj = (lam3 - lam2) / delta;
      BOOST_CHECK_CLOSE(dlam_dgj(i), dlami_dg(j), 1e0);
      b(j) -= delta;
      // test derivatives wrt Del
      for (int k = 0; k < j + 1; k++) {
        // std::cout << "k:" << k << std::endl;
        // std::cout << "j:" << j << std::endl;
        A(j, k) += delta;
        A(k, j) += delta;
        prob3 = cb::ContactProblem<T, cb::PyramidCone>(A, b, mus);
        solver3 = cb::LCPQPSolver<T>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings);
        lam3 = solver3.getSolution();
        dlam_dDeljk = (lam3 - lam2) / (2 * delta);
        BOOST_CHECK_CLOSE(dlam_dDeljk(i), dlami_dDel(j, k), 1e0);
        A(j, k) -= delta;
        A(k, j) -= delta;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(NCP_PGS_solver_polish) {
  // TODO
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(NCP_PGS_solver_vjp_approx1) {
  // 1 sticking contact (ball on floor)
  int nc = 1;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  A << 3.5, 0., 0., 0., 3.5, 0., 0., 0., 1.;
  VectorXs b = VectorXs::Zero(3 * nc);
  b << 0., 0., -0.00481;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::NCPPGSSolver<T, cb::IceCreamCone> solver2;
  solver2.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 100;
  bool polish = true, statistics = false;
  CONTACTBENCH_UNUSED(polish);
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver2.solve(prob, x0, settings, 0., 0.);
  VectorXs lam2 = solver2.getSolution();
  // return;
  //
  //
  VectorXs dL_dlam = VectorXs::Zero(3 * nc);
  double delta = 1e-5;
  VectorXs lam3 = VectorXs::Zero(3 * nc);
  VectorXs dlam_dmuj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dgj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dDeljk = VectorXs::Zero(3 * nc);
  VectorXs dlami_dmus = VectorXs::Zero(nc);
  VectorXs dlami_dg = VectorXs::Zero(3 * nc);
  MatrixXs dlami_dDel = MatrixXs::Zero(3 * nc, 3 * nc);
  cb::ContactProblem<T, cb::IceCreamCone> prob3;
  cb::NCPPGSSolver<T, cb::IceCreamCone> solver3;
  for (int i = 0; i < 3 * nc; i++) {
    // std::cout << "i:" << i << std::endl;
    dL_dlam.setZero();
    dL_dlam(i) = 1.;
    solver2.vjp_approx(prob, dL_dlam, settings);
    dlami_dmus = solver2.getdLdmus();
    dlami_dg = solver2.getdLdg();
    dlami_dDel = solver2.getdLdDel();
    for (int j = 0; j < 3 * nc; j++) {
      // std::cout << "j:" << j << std::endl;
      // test derivatives wrt mu
      if (j % 3 == 0) {
        mus[CAST_UL(j / 3)] += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::NCPPGSSolver<T, cb::IceCreamCone>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings, 0., 0.);
        lam3 = solver3.getSolution();
        dlam_dmuj = (lam3 - lam2) / delta;
        // BOOST_CHECK_CLOSE(dlam_dmuj(i), dlami_dmus(j / 3), 1e0);
        mus[CAST_UL(j / 3)] -= delta;
      }
      // test derivatives wrt g
      b(j) += delta;
      prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
      solver3 = cb::NCPPGSSolver<T, cb::IceCreamCone>();
      solver3.setProblem(prob3);
      solver3.solve(prob3, x0, settings, 0., 0.);
      lam3 = solver3.getSolution();
      dlam_dgj = (lam3 - lam2) / delta;
      // BOOST_CHECK_CLOSE(dlam_dgj(i), dlami_dg(j), 1e0);
      b(j) -= delta;
      // test derivatives wrt Del
      for (int k = 0; k < j + 1; k++) {
        // std::cout << "k:" << k << std::endl;
        A(j, k) += delta;
        A(k, j) += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::NCPPGSSolver<T, cb::IceCreamCone>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings, 0., 0.);
        lam3 = solver3.getSolution();
        dlam_dDeljk = (lam3 - lam2) / (2 * delta);
        // BOOST_CHECK_CLOSE(dlam_dDeljk(i), dlami_dDel(j, k), 1e0);
        A(j, k) -= delta;
        A(k, j) -= delta;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_vjp_approx1) {
  // return;
  // 1 sticking contact (ball on floor)
  int nc = 1;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  A << 3.5, 0., 0., 0., 3.5, 0., 0., 0., 1.;
  VectorXs b = VectorXs::Zero(3 * nc);
  b << 0., 0., -0.00481;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPPGSSolver<T> solver2;
  solver2.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 100;
  bool polish = true;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver2.solve(prob, x0, settings, 0., polish);
  VectorXs lam2 = solver2.getSolution();
  // return;
  //
  //
  VectorXs dL_dlam = VectorXs::Zero(3 * nc);
  double delta = 1e-5;
  VectorXs lam3 = VectorXs::Zero(3 * nc);
  VectorXs dlam_dmuj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dgj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dDeljk = VectorXs::Zero(3 * nc);
  VectorXs dlami_dmus = VectorXs::Zero(nc);
  VectorXs dlami_dg = VectorXs::Zero(3 * nc);
  MatrixXs dlami_dDel = MatrixXs::Zero(3 * nc, 3 * nc);
  cb::ContactProblem<T, cb::IceCreamCone> prob3;
  cb::CCPPGSSolver<T> solver3;
  for (int i = 0; i < 3 * nc; i++) {
    // std::cout << "i:" << i << std::endl;
    dL_dlam.setZero();
    dL_dlam(i) = 1.;
    solver2.vjp_approx(prob, dL_dlam, settings);
    dlami_dmus = solver2.getdLdmus();
    dlami_dg = solver2.getdLdg();
    dlami_dDel = solver2.getdLdDel();
    for (int j = 0; j < 3 * nc; j++) {
      // std::cout << "j:" << j << std::endl;
      // test derivatives wrt mu
      if (j % 3 == 0) {
        mus[CAST_UL(j / 3)] += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::CCPPGSSolver<T>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings, 0., polish);
        lam3 = solver3.getSolution();
        dlam_dmuj = (lam3 - lam2) / delta;
        BOOST_CHECK_CLOSE(dlam_dmuj(i), dlami_dmus(j / 3), 1e0);
        mus[CAST_UL(j / 3)] -= delta;
      }
      // test derivatives wrt g
      b(j) += delta;
      prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
      solver3 = cb::CCPPGSSolver<T>();
      solver3.setProblem(prob3);
      solver3.solve(prob3, x0, settings, 0., polish);
      lam3 = solver3.getSolution();
      dlam_dgj = (lam3 - lam2) / delta;
      BOOST_CHECK_CLOSE(dlam_dgj(i), dlami_dg(j), 1e0);
      b(j) -= delta;
      // test derivatives wrt Del
      for (int k = 0; k < j + 1; k++) {
        // std::cout << "k:" << k << std::endl;
        A(j, k) += delta;
        A(k, j) += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::CCPPGSSolver<T>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings, 0., polish);
        lam3 = solver3.getSolution();
        dlam_dDeljk = (lam3 - lam2) / (2 * delta);
        // BOOST_CHECK_CLOSE(dlam_dDeljk(i), dlami_dDel(j, k), 1e0);
        A(j, k) -= delta;
        A(k, j) -= delta;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_vjp_approx2) {
  return;
  // 1 sliding contact (ball dragged on floor)
  int nc = 1;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  A << 3.5, 0., 0., 0., 3.5, 0., 0., 0., 1.;
  VectorXs b = VectorXs::Zero(3 * nc);
  b << 0., 0.0132435, -0.00981;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPPGSSolver<T> solver2;
  solver2.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 100;
  bool polish = true;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver2.solve(prob, x0, settings, 0., polish);
  VectorXs lam2 = solver2.getSolution();
  // return;
  //
  //
  VectorXs dL_dlam = VectorXs::Zero(3 * nc);
  double delta = 1e-5;
  VectorXs lam3 = VectorXs::Zero(3 * nc);
  VectorXs dlam_dmuj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dgj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dDeljk = VectorXs::Zero(3 * nc);
  VectorXs dlami_dmus = VectorXs::Zero(nc);
  VectorXs dlami_dg = VectorXs::Zero(3 * nc);
  MatrixXs dlami_dDel = MatrixXs::Zero(3 * nc, 3 * nc);
  cb::ContactProblem<T, cb::IceCreamCone> prob3;
  cb::CCPPGSSolver<T> solver3;
  for (int i = 0; i < 3 * nc; i++) {
    // std::cout << "i:" << i << std::endl;
    dL_dlam.setZero();
    dL_dlam(i) = 1.;
    solver2.vjp_approx(prob, dL_dlam, settings);
    dlami_dmus = solver2.getdLdmus();
    dlami_dg = solver2.getdLdg();
    dlami_dDel = solver2.getdLdDel();
    for (int j = 0; j < 3 * nc; j++) {
      // std::cout << "j:" << j << std::endl;
      // test derivatives wrt mu
      if (j % 3 == 0) {
        mus[CAST_UL(j / 3)] += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::CCPPGSSolver<T>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings, 0., polish);
        lam3 = solver3.getSolution();
        dlam_dmuj = (lam3 - lam2) / delta;
        BOOST_CHECK_CLOSE(dlam_dmuj(i), dlami_dmus(j / 3), 1e0);
        mus[CAST_UL(j / 3)] -= delta;
      }
      // test derivatives wrt g
      b(j) += delta;
      prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
      solver3 = cb::CCPPGSSolver<T>();
      solver3.setProblem(prob3);
      solver3.solve(prob3, x0, settings, 0., polish);
      lam3 = solver3.getSolution();
      dlam_dgj = (lam3 - lam2) / delta;
      BOOST_CHECK_CLOSE(dlam_dgj(i), dlami_dg(j), 1e0);
      b(j) -= delta;
      // test derivatives wrt Del
      for (int k = 0; k < j + 1; k++) {
        // std::cout << "k:" << k << std::endl;
        A(j, k) += delta;
        A(k, j) += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::CCPPGSSolver<T>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings, 0., polish);
        lam3 = solver3.getSolution();
        dlam_dDeljk = (lam3 - lam2) / (2 * delta);
        BOOST_CHECK_CLOSE(dlam_dDeljk(i), dlami_dDel(j, k), 1e0);
        A(j, k) -= delta;
        A(k, j) -= delta;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_vjp_approx3) {
  return;
  // 1 breaking contact (lifted ball on floor)
  int nc = 1;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  A << 3.5, 0., 0., 0., 3.5, 0., 0., 0., 1.;
  VectorXs b = VectorXs::Zero(3 * nc);
  b << 0., 0., 0.00019;
  std::vector<T> mus = {0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPPGSSolver<T> solver2;
  solver2.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 100;
  bool polish = true;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver2.solve(prob, x0, settings, 0., polish);
  VectorXs lam2 = solver2.getSolution();
  // return;
  //
  //
  VectorXs dL_dlam = VectorXs::Zero(3 * nc);
  double delta = 1e-5;
  VectorXs lam3 = VectorXs::Zero(3 * nc);
  VectorXs dlam_dmuj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dgj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dDeljk = VectorXs::Zero(3 * nc);
  VectorXs dlami_dmus = VectorXs::Zero(nc);
  VectorXs dlami_dg = VectorXs::Zero(3 * nc);
  MatrixXs dlami_dDel = MatrixXs::Zero(3 * nc, 3 * nc);
  cb::ContactProblem<T, cb::IceCreamCone> prob3;
  cb::CCPPGSSolver<T> solver3;
  for (int i = 0; i < 3 * nc; i++) {
    // std::cout << "i:" << i << std::endl;
    dL_dlam.setZero();
    dL_dlam(i) = 1.;
    solver2.vjp_approx(prob, dL_dlam, settings);
    dlami_dmus = solver2.getdLdmus();
    dlami_dg = solver2.getdLdg();
    dlami_dDel = solver2.getdLdDel();
    for (int j = 0; j < 3 * nc; j++) {
      // std::cout << "j:" << j << std::endl;
      // test derivatives wrt mu
      if (j % 3 == 0) {
        mus[CAST_UL(j / 3)] += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::CCPPGSSolver<T>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings, 0., polish);
        lam3 = solver3.getSolution();
        dlam_dmuj = (lam3 - lam2) / delta;
        BOOST_CHECK_CLOSE(dlam_dmuj(i), dlami_dmus(j / 3), 1e0);
        mus[CAST_UL(j / 3)] -= delta;
      }
      // test derivatives wrt g
      b(j) += delta;
      prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
      solver3 = cb::CCPPGSSolver<T>();
      solver3.setProblem(prob3);
      solver3.solve(prob3, x0, settings, 0., polish);
      lam3 = solver3.getSolution();
      dlam_dgj = (lam3 - lam2) / delta;
      BOOST_CHECK_CLOSE(dlam_dgj(i), dlami_dg(j), 1e0);
      b(j) -= delta;
      // test derivatives wrt Del
      for (int k = 0; k < j + 1; k++) {
        // std::cout << "k:" << k << std::endl;
        A(j, k) += delta;
        A(k, j) += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::CCPPGSSolver<T>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings, 0., polish);
        lam3 = solver3.getSolution();
        dlam_dDeljk = (lam3 - lam2) / (2 * delta);
        BOOST_CHECK_CLOSE(dlam_dDeljk(i), dlami_dDel(j, k), 1e0);
        A(j, k) -= delta;
        A(k, j) -= delta;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_vjp_approx4) {
  return;
  // sticking contact
  // std::cout << " testing vjp approx " << std::endl;
  int nc = 1;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  A << 5.47732006, -0.44641268, -0.85806775, -0.44641268, 4.73256402,
      -1.83316893, -0.85806775, -1.83316893, 2.16266878;
  VectorXs b = VectorXs::Zero(3 * nc);
  b << 4.58088011e-16, 5.88878006e-16, -9.61358750e-01;
  std::vector<T> mus = {0.95};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPPGSSolver<T> solver2;
  solver2.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 100;
  bool polish = true;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver2.solve(prob, x0, settings, 0., polish);
  VectorXs lam2 = solver2.getSolution();
  // return;
  //
  //
  VectorXs dL_dlam = VectorXs::Zero(3 * nc);
  double delta = 1e-5;
  VectorXs lam3 = VectorXs::Zero(3 * nc);
  VectorXs dlam_dmuj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dgj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dDeljk = VectorXs::Zero(3 * nc);
  VectorXs dlami_dmus = VectorXs::Zero(nc);
  VectorXs dlami_dg = VectorXs::Zero(3 * nc);
  MatrixXs dlami_dDel = MatrixXs::Zero(3 * nc, 3 * nc);
  cb::ContactProblem<T, cb::IceCreamCone> prob3;
  cb::CCPPGSSolver<T> solver3;
  for (int i = 0; i < 3 * nc; i++) {
    // std::cout << "i:" << i << std::endl;
    dL_dlam.setZero();
    dL_dlam(i) = 1.;
    solver2.vjp_approx(prob, dL_dlam, settings);
    dlami_dmus = solver2.getdLdmus();
    dlami_dg = solver2.getdLdg();
    dlami_dDel = solver2.getdLdDel();
    for (int j = 0; j < 3 * nc; j++) {
      // std::cout << "j:" << j << std::endl;
      // test derivatives wrt mu
      if (j % 3 == 0) {
        mus[CAST_UL(j / 3)] += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::CCPPGSSolver<T>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings, 0., polish);
        lam3 = solver3.getSolution();
        dlam_dmuj = (lam3 - lam2) / delta;
        BOOST_CHECK_CLOSE(dlam_dmuj(i), dlami_dmus(j / 3), 1e0);
        mus[CAST_UL(j / 3)] -= delta;
      }
      // test derivatives wrt g
      b(j) += delta;
      prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
      solver3 = cb::CCPPGSSolver<T>();
      solver3.setProblem(prob3);
      solver3.solve(prob3, x0, settings, 0., polish);
      lam3 = solver3.getSolution();
      dlam_dgj = (lam3 - lam2) / delta;
      BOOST_CHECK_CLOSE(dlam_dgj(i), dlami_dg(j), 1e0);
      b(j) -= delta;
      // test derivatives wrt Del
      for (int k = 0; k < j + 1; k++) {
        // std::cout << "k:" << k << std::endl;
        A(j, k) += delta;
        A(k, j) += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::CCPPGSSolver<T>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings, 0., polish);
        lam3 = solver3.getSolution();
        dlam_dDeljk = (lam3 - lam2) / (2 * delta);
        BOOST_CHECK_CLOSE(dlam_dDeljk(i), dlami_dDel(j, k), 1e0);
        A(j, k) -= delta;
        A(k, j) -= delta;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_vjp_approx5) {
  return;
  // 2 sticking contacts and 1 breaking contact
  // std::cout << " testing vjp approx " << std::endl;
  int nc = 3;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
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
  VectorXs b = VectorXs::Zero(3 * nc);
  b << 0.0005535632995660456, -0.00021503837242162205, -0.2002703738176001,
      0.0003651400035045993, 0.00011681395199159006, -0.20929694476483335,
      0.00010131506722595163, -0.00016176693151320753, -0.009610457228502855;
  std::vector<T> mus = {0.9, 0.9, 0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  cb::CCPPGSSolver<T> solver2;
  solver2.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 100;
  bool polish = true;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver2.solve(prob, x0, settings, 0., polish);
  VectorXs lam2 = solver2.getSolution();
  // return;
  //
  //
  VectorXs dL_dlam = VectorXs::Zero(3 * nc);
  double delta = 1e-5;
  VectorXs lam3 = VectorXs::Zero(3 * nc);
  VectorXs dlam_dmuj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dgj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dDeljk = VectorXs::Zero(3 * nc);
  VectorXs dlami_dmus = VectorXs::Zero(nc);
  VectorXs dlami_dg = VectorXs::Zero(3 * nc);
  MatrixXs dlami_dDel = MatrixXs::Zero(3 * nc, 3 * nc);
  cb::ContactProblem<T, cb::IceCreamCone> prob3;
  cb::CCPPGSSolver<T> solver3;
  for (int i = 0; i < 3 * nc; i++) {
    // std::cout << "i:" << i << std::endl;
    dL_dlam.setZero();
    dL_dlam(i) = 1.;
    solver2.vjp_approx(prob, dL_dlam, settings);
    dlami_dmus = solver2.getdLdmus();
    dlami_dg = solver2.getdLdg();
    dlami_dDel = solver2.getdLdDel();
    for (int j = 0; j < 3 * nc; j++) {
      // std::cout << "j:" << j << std::endl;
      // test derivatives wrt mu
      if (j % 3 == 0) {
        mus[CAST_UL(j / 3)] += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::CCPPGSSolver<T>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings, 0., polish);
        lam3 = solver3.getSolution();
        dlam_dmuj = (lam3 - lam2) / delta;
        BOOST_CHECK_CLOSE(dlam_dmuj(i), dlami_dmus(j / 3), 1e0);
        mus[CAST_UL(j / 3)] -= delta;
      }
      //   continue;
      // test derivatives wrt g
      b(j) += delta;
      prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
      solver3 = cb::CCPPGSSolver<T>();
      solver3.setProblem(prob3);
      solver3.solve(prob3, x0, settings, 0., polish);
      lam3 = solver3.getSolution();
      dlam_dgj = (lam3 - lam2) / delta;
      BOOST_CHECK_CLOSE(dlam_dgj(i), dlami_dg(j), 1e-2);
      b(j) -= delta;
      // test derivatives wrt Del
      for (int k = 0; k < j + 1; k++) {
        // std::cout << "k:" << k << std::endl;
        A(j, k) += delta;
        A(k, j) += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::CCPPGSSolver<T>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings, 0., polish);
        lam3 = solver3.getSolution();
        dlam_dDeljk = (lam3 - lam2) / (2 * delta);
        BOOST_CHECK_CLOSE(dlam_dDeljk(i), dlami_dDel(j, k), 1e-2);
        A(j, k) -= delta;
        A(k, j) -= delta;
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_vjp_approx6) {
  return;
  // 4 sliding contacts
  // std::cout << " testing vjp approx 2" << std::endl;
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
  cb::CCPPGSSolver<T> solver2;
  solver2.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 100;
  bool polish = true;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver2.solve(prob, x0, settings, 0., polish);
  VectorXs lam2 = solver2.getSolution();
  //   std::cout << "lam: " << lam2 << std::endl;
  // return;
  //
  //
  VectorXs dL_dlam = VectorXs::Zero(3 * nc);
  double delta = 1e-5;
  VectorXs lam3 = VectorXs::Zero(3 * nc);
  VectorXs dlam_dmuj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dgj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dDeljk = VectorXs::Zero(3 * nc);
  VectorXs dlami_dmus = VectorXs::Zero(nc);
  VectorXs dlami_dg = VectorXs::Zero(3 * nc);
  MatrixXs dlami_dDel = MatrixXs::Zero(3 * nc, 3 * nc);
  cb::ContactProblem<T, cb::IceCreamCone> prob3;
  cb::CCPPGSSolver<T> solver3;
  for (int i = 0; i < 3 * nc; i++) {
    // std::cout << "i:" << i << std::endl;
    dL_dlam.setZero();
    dL_dlam(i) = 1.;
    solver2.vjp_approx(prob, dL_dlam, settings);
    dlami_dmus = solver2.getdLdmus();
    dlami_dg = solver2.getdLdg();
    dlami_dDel = solver2.getdLdDel();
    // return;
    for (int j = 0; j < 3 * nc; j++) {
      //   std::cout << "j:" << j << std::endl;
      // test derivatives wrt mu
      if (j % 3 == 0) {
        mus[CAST_UL(j / 3)] += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::CCPPGSSolver<T>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings, 0., polish);
        lam3 = solver3.getSolution();
        dlam_dmuj = (lam3 - lam2) / delta;
        BOOST_CHECK_CLOSE(dlam_dmuj(i), dlami_dmus(j / 3), 1e0);
        mus[CAST_UL(j / 3)] -= delta;
      }
      // return;
      //   continue;
      // test derivatives wrt g
      b(j) += delta;
      prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
      solver3 = cb::CCPPGSSolver<T>();
      solver3.setProblem(prob3);
      solver3.solve(prob3, x0, settings, 0., polish);
      lam3 = solver3.getSolution();
      dlam_dgj = (lam3 - lam2) / delta;
      BOOST_CHECK_CLOSE(dlam_dgj(i), dlami_dg(j), 1e0);
      b(j) -= delta;
      // test derivatives wrt Del
      for (int k = 0; k < j + 1; k++) {
        // std::cout << "k:" << k << std::endl;
        A(j, k) += delta;
        A(k, j) += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::CCPPGSSolver<T>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings, 0., polish);
        lam3 = solver3.getSolution();
        dlam_dDeljk = (lam3 - lam2) / (2 * delta);
        BOOST_CHECK_CLOSE(dlam_dDeljk(i), dlami_dDel(j, k), 1e0);
        A(j, k) -= delta;
        A(k, j) -= delta;
      }
      // return;
    }
  }
}

BOOST_AUTO_TEST_CASE(CCP_PGS_solver_vjp_approx7) {
  return;
  // 4 sliding contacts
  // std::cout << " testing vjp approx 2" << std::endl;
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
  cb::CCPPGSSolver<T> solver2;
  solver2.setProblem(prob);
  VectorXs x0 = VectorXs::Zero(3 * nc);
  int maxIter = 100;
  bool polish = true;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver2.solve(prob, x0, settings, 0., polish);
  VectorXs lam2 = solver2.getSolution();
  // return;
  //
  //
  VectorXs dL_dlam = VectorXs::Zero(3 * nc);
  double delta = 1e-5;
  VectorXs lam3 = VectorXs::Zero(3 * nc);
  VectorXs dlam_dmuj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dgj = VectorXs::Zero(3 * nc);
  VectorXs dlam_dDeljk = VectorXs::Zero(3 * nc);
  VectorXs dlami_dmus = VectorXs::Zero(nc);
  VectorXs dlami_dg = VectorXs::Zero(3 * nc);
  MatrixXs dlami_dDel = MatrixXs::Zero(3 * nc, 3 * nc);
  cb::ContactProblem<T, cb::IceCreamCone> prob3;
  cb::CCPPGSSolver<T> solver3;
  for (int i = 0; i < 3 * nc; i++) {
    std::cout << "i:" << i << std::endl;
    dL_dlam.setZero();
    dL_dlam(i) = 1.;
    solver2.vjp_approx(prob, dL_dlam, settings);
    dlami_dmus = solver2.getdLdmus();
    dlami_dg = solver2.getdLdg();
    dlami_dDel = solver2.getdLdDel();
    // return;
    for (int j = 0; j < 3 * nc; j++) {
      std::cout << "j:" << j << std::endl;
      // test derivatives wrt mu
      if (j % 3 == 0) {
        mus[CAST_UL(j / 3)] += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::CCPPGSSolver<T>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings, 0., polish);
        lam3 = solver3.getSolution();
        dlam_dmuj = (lam3 - lam2) / delta;
        BOOST_CHECK_CLOSE(dlam_dmuj(i), dlami_dmus(j / 3), 1e0);
        mus[CAST_UL(j / 3)] -= delta;
      }
      // return;
      continue;
      // test derivatives wrt g
      b(j) += delta;
      prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
      solver3 = cb::CCPPGSSolver<T>();
      solver3.setProblem(prob3);
      solver3.solve(prob3, x0, settings, 0., polish);
      lam3 = solver3.getSolution();
      dlam_dgj = (lam3 - lam2) / delta;
      BOOST_CHECK_CLOSE(dlam_dgj(i), dlami_dg(j), 1e0);
      b(j) -= delta;
      // test derivatives wrt Del
      for (int k = 0; k < j + 1; k++) {
        // std::cout << "k:" << k << std::endl;
        A(j, k) += delta;
        A(k, j) += delta;
        prob3 = cb::ContactProblem<T, cb::IceCreamCone>(A, b, mus);
        solver3 = cb::CCPPGSSolver<T>();
        solver3.setProblem(prob3);
        solver3.solve(prob3, x0, settings, 0., polish);
        lam3 = solver3.getSolution();
        dlam_dDeljk = (lam3 - lam2) / (2 * delta);
        BOOST_CHECK_CLOSE(dlam_dDeljk(i), dlami_dDel(j, k), 1e0);
        A(j, k) -= delta;
        A(k, j) -= delta;
      }
      // return;
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
