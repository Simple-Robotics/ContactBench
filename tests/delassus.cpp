#include "contactbench/delassus-wrapper.hpp"
#include "pinocchio/multibody/fwd.hpp"
#include "vector"

#include <iostream>
#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>
#include <pinocchio/parsers/sample-models.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/contact-cholesky.hpp>
#include <pinocchio/algorithm/contact-info.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/crba.hpp>

BOOST_AUTO_TEST_SUITE(DELASSUS_WRAPPER)

namespace cb = contactbench;
using isize = Eigen::Index;
using T = double;
static constexpr int Options = Eigen::ColMajor;
using Model = pinocchio::ModelTpl<T>;
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

BOOST_AUTO_TEST_CASE(delassus_pinocchio_init) {
  cb::DelassusPinocchio<T> Del;
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(delassus_dense_init) {
  isize nc = 3;
  MatrixXs G = MatrixXs::Ones(3 * nc, 3 * nc);
  cb::DelassusDense<T> Del(G);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(delassus_dense_computeChol) {
  isize nc = 3;
  MatrixXs G = MatrixXs::Ones(3 * nc, 3 * nc);
  cb::DelassusDense<T> Del(G);
  Del.computeChol(1e-9);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(delassus_dense_applyOnTheRight) {
  isize nc = 3;
  MatrixXs G = MatrixXs::Ones(3 * nc, 3 * nc);
  cb::DelassusDense<T> Del(G);
  VectorXs lam = VectorXs::Random(3 * nc);
  VectorXs lam_out = VectorXs::Random(3 * nc);
  Del.applyOnTheRight(lam, lam_out);
  VectorXs lam_out_test = VectorXs::Random(3 * nc);
  lam_out_test = G * lam;
  BOOST_CHECK(lam_out_test.isApprox(lam_out));
}

BOOST_AUTO_TEST_CASE(delassus_dense_applyPerContactOnTheRight) {
  isize nc = 3;
  MatrixXs G = MatrixXs::Ones(3 * nc, 3 * nc);
  cb::DelassusDense<T> Del(G);
  VectorXs lam = VectorXs::Random(3 * nc);
  VectorXs lam_out = VectorXs::Random(3);
  int i = 2;
  Del.applyPerContactOnTheRight(i, lam, lam_out);
  VectorXs lam_out_test = VectorXs::Random(3);
  lam_out_test = G.middleRows<3>(3 * i) * lam;
  BOOST_CHECK(lam_out_test.isApprox(lam_out));
}

BOOST_AUTO_TEST_CASE(delassus_dense_applyPerContactNormalOnTheRight) {
  isize nc = 3;
  MatrixXs G = MatrixXs::Ones(3 * nc, 3 * nc);
  cb::DelassusDense<T> Del(G);
  VectorXs lam = VectorXs::Random(3 * nc);
  double lam_out = 0.;
  int i = 2;
  Del.applyPerContactNormalOnTheRight(i, lam, lam_out);
  double lam_out_test = 1.;
  lam_out_test = G.row(3 * i + 2) * lam;
  double test = lam.sum(); CONTACTBENCH_UNUSED(test);
  BOOST_CHECK(lam_out_test == lam_out);
}

BOOST_AUTO_TEST_CASE(delassus_dense_applyPerContactTangentOnTheRight) {
  isize nc = 3;
  MatrixXs G = MatrixXs::Ones(3 * nc, 3 * nc);
  cb::DelassusDense<T> Del(G);
  VectorXs lam = VectorXs::Random(3 * nc);
  VectorXs lam_out = VectorXs::Random(2);
  int i = 2;
  Del.applyPerContactTangentOnTheRight(i, lam, lam_out);
  VectorXs lam_out_test = VectorXs::Random(2);
  lam_out_test = G.middleRows<2>(3 * i) * lam;
  BOOST_CHECK(lam_out_test.isApprox(lam_out));
}

BOOST_AUTO_TEST_CASE(delassus_dense_solve) {
  isize nc = 3;
  MatrixXs G = MatrixXs::Random(3 * nc, 3 * nc);
  G = G * G.transpose();
  cb::DelassusDense<T> Del(G);
  Del.computeChol(1e-9);
  VectorXs lam = VectorXs::Random(3 * nc);
  VectorXs lam_out = VectorXs::Random(3 * nc);
  Del.solve(lam, lam_out);
  VectorXs lam_out_test = VectorXs::Random(3 * nc);
  lam_out_test =
      (G + 1e-9 * MatrixXs::Identity(3 * nc, 3 * nc)).inverse() * lam;
  BOOST_CHECK(lam_out_test.isApprox(lam_out));
}

BOOST_AUTO_TEST_CASE(delassus_dense_solveInPlace) {
  isize nc = 3;
  MatrixXs G = MatrixXs::Random(3 * nc, 3 * nc);
  G = G * G.transpose();
  cb::DelassusDense<T> Del(G);
  Del.computeChol(1e-9);
  VectorXs lam = VectorXs::Random(3 * nc);
  VectorXs lam2 = lam;
  Del.solveInPlace(lam);
  VectorXs lam_out_test = VectorXs::Random(3 * nc);
  lam_out_test =
      (G + 1e-9 * MatrixXs::Identity(3 * nc, 3 * nc)).inverse() * lam2;
  BOOST_CHECK(lam_out_test.isApprox(lam));
}

BOOST_AUTO_TEST_CASE(delassus_pinocchio_computeChol) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};
  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);
  // we should do crba before cholesky computation
  VectorXs q = pinocchio::neutral(model);
  pinocchio::crba(model, data, q);
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
  cb::DelassusPinocchio<T> Del(model, data, contact_models, contact_datas);
  Del.computeChol(1e-9);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(delassus_pinocchio_evaluateDel) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};
  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);
  // we should do crba before cholesky computation
  VectorXs q = pinocchio::neutral(model);
  pinocchio::crba(model, data, q);
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
  cb::DelassusPinocchio<T> Del(model, data, contact_models, contact_datas);
  Del.computeChol(1e-9);
  Del.evaluateDel();
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(delassus_pinocchio_applyPerContactOnTheRight) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};
  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);
  // we should do crba before cholesky computation
  VectorXs q = pinocchio::neutral(model);
  pinocchio::crba(model, data, q);
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
  cb::DelassusPinocchio<T> Del(model, data, contact_models, contact_datas);
  Del.computeChol(1e-9);
  Del.evaluateDel();
  VectorXs lam = VectorXs::Ones(3 * nc);
  VectorXs lam_out = VectorXs::Ones(3);
  int i = 2;
  Del.applyPerContactOnTheRight(i, lam, lam_out);
  VectorXs lam_out_test = Del.G_.middleRows<3>(3 * i) * lam;
  BOOST_CHECK(lam_out.isApprox(lam_out_test));
}

BOOST_AUTO_TEST_CASE(delassus_pinocchio_applyPerContactNormalOnTheRight) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};
  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);
  // we should do crba before cholesky computation
  VectorXs q = pinocchio::neutral(model);
  pinocchio::crba(model, data, q);
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
  cb::DelassusPinocchio<T> Del(model, data, contact_models, contact_datas);
  Del.computeChol(1e-9);
  Del.evaluateDel();
  VectorXs lam = VectorXs::Ones(3 * nc);
  double lam_out = 0.;
  int i = 2;
  Del.applyPerContactNormalOnTheRight(i, lam, lam_out);
  double lam_out_test = Del.G_.row(3 * i + 2) * lam;
  BOOST_CHECK(lam_out == lam_out_test);
}

BOOST_AUTO_TEST_CASE(delassus_pinocchio_applyPerContactTangentOnTheRight) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};
  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);
  // we should do crba before cholesky computation
  VectorXs q = pinocchio::neutral(model);
  pinocchio::crba(model, data, q);
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
  cb::DelassusPinocchio<T> Del(model, data, contact_models, contact_datas);
  Del.computeChol(1e-9);
  Del.evaluateDel();
  VectorXs lam = VectorXs::Ones(3 * nc);
  VectorXs lam_out = VectorXs::Ones(2);
  int i = 2;
  Del.applyPerContactTangentOnTheRight(i, lam, lam_out);
  VectorXs lam_out_test = Del.G_.middleRows<2>(3 * i) * lam;
  BOOST_CHECK(lam_out.isApprox(lam_out_test));
}

BOOST_AUTO_TEST_CASE(delassus_pinocchio_applyOnTheRight) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};
  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);
  // we should do crba before cholesky computation
  VectorXs q = pinocchio::neutral(model);
  pinocchio::crba(model, data, q);
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
  cb::DelassusPinocchio<T> Del(model, data, contact_models, contact_datas);
  Del.computeChol(1e-9);
  Del.evaluateDel();
  VectorXs lam = VectorXs::Ones(3 * nc);
  VectorXs lam_out = VectorXs::Ones(3 * nc);
  Del.applyOnTheRight(lam, lam_out);
  VectorXs lam_out_test = Del.G_ * lam;
  BOOST_CHECK(lam_out.isApprox(lam_out_test));
}

BOOST_AUTO_TEST_CASE(delassus_pinocchio_solve) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};
  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);
  // we should do crba before cholesky computation
  VectorXs q = pinocchio::neutral(model);
  pinocchio::crba(model, data, q);
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
  cb::DelassusPinocchio<T> Del(model, data, contact_models, contact_datas);
  Del.computeChol(1e-9);
  VectorXs lam = VectorXs::Ones(3 * nc);
  VectorXs lam_out = VectorXs::Ones(3 * nc);
  Del.solve(lam, lam_out);
  Del.evaluateDel();
  VectorXs lam_out_test =
      (Del.G_ + 1e-9 * MatrixXs::Identity(3 * nc, 3 * nc)).inverse() * lam;
  BOOST_CHECK(lam_out.isApprox(lam_out_test));
}

BOOST_AUTO_TEST_CASE(delassus_pinocchio_solveInPlace) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};
  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);
  // we should do crba before cholesky computation
  VectorXs q = pinocchio::neutral(model);
  pinocchio::crba(model, data, q);
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
  cb::DelassusPinocchio<T> Del(model, data, contact_models, contact_datas);
  Del.computeChol(1e-9);
  VectorXs lam = VectorXs::Ones(3 * nc);
  VectorXs lam2 = VectorXs::Ones(3 * nc);
  Del.solveInPlace(lam);
  Del.evaluateDel();
  VectorXs lam_out_test =
      (Del.G_ + 1e-9 * MatrixXs::Identity(3 * nc, 3 * nc)).inverse() * lam2;
  BOOST_CHECK(lam.isApprox(lam_out_test));
}

BOOST_AUTO_TEST_SUITE_END()
