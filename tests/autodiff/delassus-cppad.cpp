#include "contactbench/delassus-wrapper.hpp"
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
#include <pinocchio/autodiff/cppad.hpp>

BOOST_AUTO_TEST_SUITE(DELASSUS_WRAPPER_CPPAD)

namespace cb = contactbench;
using isize = Eigen::Index;
using CppAD::AD;
// using CppAD::NearEqual;

typedef double Scalar;
typedef AD<Scalar> ADScalar;
using T = ADScalar;
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

// using Model = pinocchio::ModelTpl<T>;
// using Data = pinocchio::DataTpl<T>;
// using SE3 = pinocchio::SE3Tpl<T>;
// using RigidConstraintModel = pinocchio::RigidConstraintModelTpl<T, Options>;
// using RigidConstraintData = pinocchio::RigidConstraintDataTpl<T, Options>;
// using RigidConstraintModelVector =
//     CONTACTBENCH_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintModel);
// using RigidConstraintDataVector =
//     CONTACTBENCH_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidConstraintData);

CONTACTBENCH_EIGEN_TYPEDEFS(T);

BOOST_AUTO_TEST_CASE(delassus_pinocchio_init_cppad) {
  cb::DelassusPinocchio<T> Del;
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(delassus_dense_init_cppad) {
  isize nc = 3;
  MatrixXs G = MatrixXs::Ones(3 * nc, 3 * nc);
  cb::DelassusDense<T> Del(G);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(delassus_dense_computeChol_cppad) {
  isize nc = 3;
  MatrixXs G = MatrixXs::Ones(3 * nc, 3 * nc);
  cb::DelassusDense<T> Del(G);
  Del.computeChol(1e-9);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(delassus_dense_applyOnTheRight_cppad) {
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

BOOST_AUTO_TEST_CASE(delassus_dense_applyPerContactOnTheRight_cppad) {
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

BOOST_AUTO_TEST_CASE(delassus_dense_applyPerContactNormalOnTheRight_cppad) {
  isize nc = 3;
  MatrixXs G = MatrixXs::Ones(3 * nc, 3 * nc);
  cb::DelassusDense<T> Del(G);
  VectorXs lam = VectorXs::Random(3 * nc);
  T lam_out = 0.;
  int i = 2;
  Del.applyPerContactNormalOnTheRight(i, lam, lam_out);
  T lam_out_test = 1.;
  lam_out_test = (G.row(3 * i + 2) * lam).value();
  T test = lam.sum();
  BOOST_CHECK(lam_out_test == lam_out);
}

BOOST_AUTO_TEST_CASE(delassus_dense_applyPerContactTangentOnTheRight_cppad) {
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

BOOST_AUTO_TEST_CASE(delassus_dense_solve_cppad) {
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
  BOOST_CHECK(lam_out_test.isApprox(lam_out, 1e-8));
}

BOOST_AUTO_TEST_CASE(delassus_dense_solveInPlace_cppad) {
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
  BOOST_CHECK(lam_out_test.isApprox(lam, 1e-8));
}

BOOST_AUTO_TEST_CASE(delassus_pinocchio_computeChol_cppad) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};

  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);

  ADModel ad_model = model.cast<ADScalar>();
  ADData ad_data(ad_model);

  // we should do crba before cholesky computation
  ConfigVectorType q = pinocchio::neutral(model);
  ADConfigVectorType ad_q = q.cast<ADScalar>();
  pinocchio::crba(ad_model, ad_data, ad_q);

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

  cb::DelassusPinocchio<T> Del(ad_model, ad_data, ad_contact_models,
                               ad_contact_datas);
  Del.computeChol(1e-9);
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(delassus_pinocchio_evaluateDel_cppad) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};

  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);

  ADModel ad_model = model.cast<ADScalar>();
  ADData ad_data(ad_model);

  // we should do crba before cholesky computation
  ConfigVectorType q = pinocchio::neutral(model);
  ADConfigVectorType ad_q = q.cast<ADScalar>();
  pinocchio::crba(ad_model, ad_data, ad_q);

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

  cb::DelassusPinocchio<T> Del(ad_model, ad_data, ad_contact_models,
                               ad_contact_datas);
  Del.computeChol(1e-9);
  Del.evaluateDel();
  BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(delassus_pinocchio_applyPerContactOnTheRight_cppad) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};

  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);

  ADModel ad_model = model.cast<ADScalar>();
  ADData ad_data(ad_model);

  // we should do crba before cholesky computation
  ConfigVectorType q = pinocchio::neutral(model);
  ADConfigVectorType ad_q = q.cast<ADScalar>();
  pinocchio::crba(ad_model, ad_data, ad_q);

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
  cb::DelassusPinocchio<T> Del(ad_model, ad_data, ad_contact_models,
                               ad_contact_datas);
  Del.computeChol(1e-9);
  Del.evaluateDel();
  VectorXs lam = VectorXs::Ones(3 * nc);
  VectorXs lam_out = VectorXs::Ones(3);
  int i = 2;
  Del.applyPerContactOnTheRight(i, lam, lam_out);
  VectorXs lam_out_test = Del.G_.middleRows<3>(3 * i) * lam;
  BOOST_CHECK(lam_out.isApprox(lam_out_test));
}

BOOST_AUTO_TEST_CASE(delassus_pinocchio_applyPerContactNormalOnTheRight_cppad) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};
  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);

  ADModel ad_model = model.cast<ADScalar>();
  ADData ad_data(ad_model);

  // we should do crba before cholesky computation
  ConfigVectorType q = pinocchio::neutral(model);
  ADConfigVectorType ad_q = q.cast<ADScalar>();
  pinocchio::crba(ad_model, ad_data, ad_q);

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

  cb::DelassusPinocchio<T> Del(ad_model, ad_data, ad_contact_models,
                               ad_contact_datas);
  Del.computeChol(1e-9);
  Del.evaluateDel();
  VectorXs lam = VectorXs::Ones(3 * nc);
  T lam_out = 0.;
  int i = 2;
  Del.applyPerContactNormalOnTheRight(i, lam, lam_out);
  T lam_out_test = (Del.G_.row(3 * i + 2) * lam).value();
  BOOST_CHECK(lam_out == lam_out_test);
}

BOOST_AUTO_TEST_CASE(
    delassus_pinocchio_applyPerContactTangentOnTheRight_cppad) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};
  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);

  ADModel ad_model = model.cast<ADScalar>();
  ADData ad_data(ad_model);

  // we should do crba before cholesky computation
  ConfigVectorType q = pinocchio::neutral(model);
  ADConfigVectorType ad_q = q.cast<ADScalar>();
  pinocchio::crba(ad_model, ad_data, ad_q);

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

  cb::DelassusPinocchio<T> Del(ad_model, ad_data, ad_contact_models,
                               ad_contact_datas);
  Del.computeChol(1e-9);
  Del.evaluateDel();
  VectorXs lam = VectorXs::Ones(3 * nc);
  VectorXs lam_out = VectorXs::Ones(2);
  int i = 2;
  Del.applyPerContactTangentOnTheRight(i, lam, lam_out);
  VectorXs lam_out_test = Del.G_.middleRows<2>(3 * i) * lam;
  BOOST_CHECK(lam_out.isApprox(lam_out_test));
}

BOOST_AUTO_TEST_CASE(delassus_pinocchio_applyOnTheRight_cppad) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};
  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);

  ADModel ad_model = model.cast<ADScalar>();
  ADData ad_data(ad_model);

  // we should do crba before cholesky computation
  ConfigVectorType q = pinocchio::neutral(model);
  ADConfigVectorType ad_q = q.cast<ADScalar>();
  pinocchio::crba(ad_model, ad_data, ad_q);

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

  cb::DelassusPinocchio<T> Del(ad_model, ad_data, ad_contact_models,
                               ad_contact_datas);
  Del.computeChol(1e-9);
  Del.evaluateDel();
  VectorXs lam = VectorXs::Ones(3 * nc);
  VectorXs lam_out = VectorXs::Ones(3 * nc);
  Del.applyOnTheRight(lam, lam_out);
  VectorXs lam_out_test = Del.G_ * lam;
  BOOST_CHECK(lam_out.isApprox(lam_out_test));
}

BOOST_AUTO_TEST_CASE(delassus_pinocchio_solve_cppad) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};
  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);

  ADModel ad_model = model.cast<ADScalar>();
  ADData ad_data(ad_model);

  // we should do crba before cholesky computation
  ConfigVectorType q = pinocchio::neutral(model);
  ADConfigVectorType ad_q = q.cast<ADScalar>();
  pinocchio::crba(ad_model, ad_data, ad_q);

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

  cb::DelassusPinocchio<T> Del(ad_model, ad_data, ad_contact_models,
                               ad_contact_datas);
  Del.computeChol(1e-9);
  VectorXs lam = VectorXs::Ones(3 * nc);
  VectorXs lam_out = VectorXs::Ones(3 * nc);
  Del.solve(lam, lam_out);
  Del.evaluateDel();
  VectorXs lam_out_test =
      (Del.G_ + 1e-9 * MatrixXs::Identity(3 * nc, 3 * nc)).inverse() * lam;
  BOOST_CHECK(lam_out.isApprox(lam_out_test));
}

BOOST_AUTO_TEST_CASE(delassus_pinocchio_solveInPlace_cppad) {
  isize nc = 3;
  VectorXs b = VectorXs::Zero(3 * nc);
  std::vector<T> mus = {0.9, 0.8, 0.7};
  Model model;
  pinocchio::buildModels::manipulator(model);
  Data data(model);

  ADModel ad_model = model.cast<ADScalar>();
  ADData ad_data(ad_model);

  // we should do crba before cholesky computation
  ConfigVectorType q = pinocchio::neutral(model);
  ADConfigVectorType ad_q = q.cast<ADScalar>();
  pinocchio::crba(ad_model, ad_data, ad_q);

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

  cb::DelassusPinocchio<T> Del(ad_model, ad_data, ad_contact_models,
                               ad_contact_datas);
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
