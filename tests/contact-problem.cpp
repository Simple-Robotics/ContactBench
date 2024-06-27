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

BOOST_AUTO_TEST_SUITE(PROBLEMS)

namespace cb = contactbench;
using T = double;
using Model = pinocchio::ModelTpl<T>;
using cb::isize;
using cb::usize;
static constexpr int Options = Eigen::ColMajor;
using Data = pinocchio::DataTpl<T>;
using JointIndex = pinocchio::JointIndex;
using SE3 = pinocchio::SE3Tpl<T>;
using RigidConstraintModel = pinocchio::RigidConstraintModelTpl<T, Options>;
using RigidConstraintData = pinocchio::RigidConstraintDataTpl<T, Options>;
using RigidConstraintModelVector =
    CONTACTBENCH_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintModel);
using RigidConstraintDataVector =
    CONTACTBENCH_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintData);
CONTACTBENCH_EIGEN_TYPEDEFS(T);

BOOST_AUTO_TEST_CASE(contact_problem_init) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};
  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(contact_problem_init_with_delassus) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};
  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);
  RigidConstraintModelVector contact_models;
  RigidConstraintDataVector contact_datas;
  for (int i = 0; i < nc; i++) {
    JointIndex id1 = 0;
    JointIndex id2 = 1;
    SE3 placement1 = SE3::Identity();
    SE3 placement2 = SE3::Identity();
    RigidConstraintModel contact_model_i(pinocchio::CONTACT_3D, model, id1,
                                         placement1, id2, placement2,
                                         pinocchio::LOCAL);
    RigidConstraintData contact_data_i(contact_model_i);
    contact_models.push_back(contact_model_i);
    contact_datas.push_back(contact_data_i);
  }
  auto Del = std::make_shared<cb::DelassusPinocchio<T>>(
      model, data, contact_models, contact_datas);
  int nc_test = Del->nc_;
  CONTACTBENCH_UNUSED(nc_test);
  cb::ContactProblem<T, cb::IceCreamCone> prob(Del, b, mus);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(primal_contact_problem_init_with_delassus) {
  // TODO
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};
  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);
  VectorXs dqf = VectorXs::Zero(model.nv);
  VectorXs vstar = VectorXs::Zero(3 * nc);
  MatrixXs M = MatrixXs::Zero(model.nv, model.nv);
  MatrixXs J = MatrixXs::Zero(3 * nc, model.nv);
  RigidConstraintModelVector contact_models;
  RigidConstraintDataVector contact_datas;
  for (int i = 0; i < nc; i++) {
    JointIndex id1 = 0;
    JointIndex id2 = 1;
    SE3 placement1 = SE3::Identity();
    SE3 placement2 = SE3::Identity();
    RigidConstraintModel contact_model_i(pinocchio::CONTACT_3D, model, id1,
                                         placement1, id2, placement2,
                                         pinocchio::LOCAL);
    RigidConstraintData contact_data_i(contact_model_i);
    contact_models.push_back(contact_model_i);
    contact_datas.push_back(contact_data_i);
  }
  auto Del = std::make_shared<cb::DelassusPinocchio<T>>(
      model, data, contact_models, contact_datas);
  int nc_test = Del->nc_;
  CONTACTBENCH_UNUSED(nc_test);
  cb::ContactProblem<T, cb::IceCreamCone> prob(M, J, dqf, vstar, mus);
  cb::ContactProblem<T, cb::IceCreamCone> prob2(Del, b, M, J, dqf, vstar, mus);
  Del->evaluateDel();
  cb::ContactProblem<T, cb::IceCreamCone> prob3(Del->G_, b, M, J, dqf, vstar,
                                                mus);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(contact_problem_contactComplementarity) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  T comp = prob.computeContactComplementarity(lam);
  BOOST_CHECK(!std::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_contactComplementarity2) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  VectorXs v = VectorXs::Zero(3 * nc);
  T comp = prob.computeContactComplementarity(lam, v);
  BOOST_CHECK(!std::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_linearComplementarity) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  prob.setLCP();
  prob.computeLCP();
  T comp = prob.computeLinearComplementarity(lam);
  BOOST_CHECK(!std::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_linearComplementarity2) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  VectorXs v = VectorXs::Zero(3 * nc);
  prob.setLCP();
  prob.computeLCP();
  T comp = prob.computeLinearComplementarity(lam, v);
  BOOST_CHECK(!std::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_signoriniComplementarity) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  T comp = prob.computeSignoriniComplementarity(lam);
  BOOST_CHECK(!std::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_signoriniComplementarity2) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  VectorXs v = VectorXs::Zero(3 * nc);
  T comp = prob.computeSignoriniComplementarity(lam, v);
  BOOST_CHECK(!std::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_conicComplementarity) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  T comp = prob.computeConicComplementarity(lam);
  BOOST_CHECK(!std::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_conicComplementarity2) {
  isize nc = 10;
  MatrixXs A = MatrixXs::Zero(3 * nc, 3 * nc);
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

  cb::ContactProblem<T, cb::IceCreamCone> prob(A, b, mus);
  VectorXs lam = VectorXs::Zero(3 * nc);
  VectorXs v = VectorXs::Zero(3 * nc);
  T comp = prob.computeConicComplementarity(lam, v);
  BOOST_CHECK(!std::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_perContactContactComplementarity) {
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
  BOOST_CHECK(!std::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_perContactSignoriniComplementarity) {
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
  BOOST_CHECK(!std::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_perContactConicComplementarity) {
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
  BOOST_CHECK(!std::isnan(comp));
}

BOOST_AUTO_TEST_CASE(contact_problem_setLCP) {
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

BOOST_AUTO_TEST_CASE(contact_problem_computeLCP) {
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
  double vt_norm = v.segment<2>(0).norm();
  BOOST_CHECK(std::abs(vt_norm - lam_lcp(5)) < 1e-4);
  double comp_lcp = v_lcp.dot(lam_lcp);
  BOOST_CHECK(std::abs(comp_lcp) < 1e-5);
}

BOOST_AUTO_TEST_CASE(contact_problem_computeLCP2) {
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
  solver.solve(prob, x0, settings, 1.);
  VectorXs lam = solver.getSolution();
  VectorXs v = A * lam + b;
  prob.setLCP();
  prob.computeLCP();
  VectorXs lam_lcp = VectorXs::Zero(6 * nc);
  prob.computeLCPSolution(lam, lam_lcp);
  VectorXs lam_b = prob.UD_ * lam_lcp;
  BOOST_CHECK((lam_b).isApprox(lam, 1e-3));
  VectorXs v_lcp = prob.A_ * lam_lcp + prob.b_;
  double vt_norm = v.segment<2>(0).norm();
  BOOST_CHECK(std::abs(vt_norm - lam_lcp(5)) < 1e-4);
  double comp_lcp = v_lcp.dot(lam_lcp);
  BOOST_CHECK(std::abs(comp_lcp) < 1e-5);
}

BOOST_AUTO_TEST_CASE(contact_problem_getTangentConstraintMatrix) {
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
  int maxIter = 100;
  cb::ContactSolverSettings<T> settings;
  settings.max_iter_ = maxIter;
  settings.th_stop_ = 1e-12;
  settings.rel_th_stop_ = 1e-12;
  solver.solve(prob, x0, settings);
  VectorXs lam = solver.getSolution();
  VectorXs c = prob.Del_->G_ * lam + prob.g_;
  VectorXs c_T_dir = VectorXs::Zero(2 * nc);
  VectorXs c_T_norm = VectorXs::Zero(nc);
  for (isize i = 0; i < prob.nc_; i++) {
    c_T_dir.segment<2>(2 * i) = c.segment<2>(3 * i);
    c_T_dir.segment<2>(2 * i).normalize();
    c_T_norm(i) = c.segment<2>(3 * i).norm();
    // std::cout << "lami :  " << lam.segment<3>(3*i) << std::endl;
    // std::cout << "lami on border:  " <<
    // prob.contact_constraints_[i].isOnBorder(lam.segment<3>(3*i)) <<
    // std::endl; std::cout << "tangent vel :  " << c_T_norm(i) << std::endl;
    // std::cout << "tangent vel dir :  " << c_T_dir.segment<2>(2*i) <<
    // std::endl;
  }
  prob.setLCCP();
  prob.computeTangentLCCP(solver.getSolution());
  MatrixXs C = prob.getLinearInConstraintMatrix();
  // std::cout << "C:  " << C << std::endl;
  VectorXs constraint = C * lam;
  // std::cout << "Constraint:  " << constraint << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()
