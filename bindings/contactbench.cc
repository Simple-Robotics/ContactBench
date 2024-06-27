#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/contact-cholesky.hpp>
#include <pinocchio/algorithm/contact-info.hpp>
#include <eigenpy/eigenpy.hpp>
#include <eigenpy/std-vector.hpp>

#include "contactbench.hh"
#include "contactbench/bindings/context.hh"
#include "contactbench/friction-constraint.hpp"
#include "contactbench/delassus-wrapper.hpp"
#include "contactbench/contact-problem.hpp"
#include "contactbench/solvers.hpp"

#define DEF_RW_CLASS_ATTRIB(CLASS, ATTRIB)                                     \
  def_readwrite(#ATTRIB, &CLASS::ATTRIB)

#define DEF_R_CLASS_ATTRIB(CLASS, ATTRIB) def_readonly(#ATTRIB, &CLASS::ATTRIB)

namespace contactbench {
namespace context {

namespace cb = contactbench;

BOOST_PYTHON_MODULE(libpycontact) {

  boost::python::import("pinocchio");

  eigenpy::enableEigenPy();
  expose_contact_bench();
}

void expose_vector_types() {
  using eigenpy::StdVectorPythonVisitor;
  StdVectorPythonVisitor<std::vector<T>, true>::expose("StdVec_scalar");
}

namespace bp = boost::python;

template <typename ConeType>
struct ConePythonVisitor : bp::def_visitor<ConeType> {
  template <typename PyClass> void expose(PyClass &cl, std::string cl_name) {
    cl.def(
          "project",
          +[](ConeType &self, const Ref<Vector3d> x,
              const Ref<Vector3d> x_out) { self.project(x, x_out); },
          bp::args("self", "x", "x_out"))
        .def(
            "projectDual",
            +[](ConeType &self, const Ref<Vector3d> x,
                const Ref<Vector3d> x_out) { self.projectDual(x, x_out); },
            bp::args("self", "x", "x_out"))
        .def(
            "isInside",
            +[](ConeType &self, const Ref<Vector3d> x, double thresh) {
              return self.isInside(x, thresh);
            },
            bp::args("self", "x"))
        .def(
            "isInsideDual",
            +[](ConeType &self, const Ref<Vector3d> x, double thresh) {
              return self.isInsideDual(x, thresh);
            },
            bp::args("self", "x", "thresh"));
    std::string name = "StdVec_";
    name += cl_name;
    eigenpy::StdVectorPythonVisitor<ConeType>::expose(name);
  }
};

void expose_contact_bench() {
  using Eigen::Ref;
  // static constexpr int Options = Eigen::ColMajor;
  using Model = pin::ModelTpl<T>;
  using Data = pin::DataTpl<T>;
  // using RigidConstraintModel = pin::RigidConstraintModelTpl<T, Options>;
  // using RigidConstraintData = pin::RigidConstraintDataTpl<T, Options>;

  if (!eigenpy::register_symbolic_link_to_registered_type<
          FrictionConstraint>()) {
    bp::class_<FrictionConstraint>("FrictionConstraint",
                                   "A friction constraint.", bp::no_init)
        .def(bp::init<>(bp::arg("self"), "Default constructor"));
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<IceCreamCone>()) {
    bp::class_<IceCreamCone, bp::bases<FrictionConstraint>>(
        "IceCreamCone", "A quadratic cone constraint.", bp::no_init)
        .def(bp::init<double>(bp::args("self", "mu"), "Other constructor"))

        // Exposing attributes
        .DEF_RW_CLASS_ATTRIB(IceCreamCone, mu_)
        //   Exposing methods
        .def(
            "project",
            +[](const T &mu, const Ref<const Vector3s> &x,
                Ref<Vector3s> x_out) { IceCreamCone::project(mu, x, x_out); },
            bp::args("mu", "x", "x_out"))
        .def(
            "project",
            +[](IceCreamCone &self, const Ref<const Vector3s> &x,
                Ref<Vector3s> x_out) { self.project(x, x_out); },
            bp::args("self", "x", "x_out"))
        .def(
            "projectDual",
            +[](IceCreamCone &self, const Ref<const Vector3s> &x,
                Ref<Vector3s> x_out) { self.projectDual(x, x_out); },
            bp::args("self", "x", "x_out"))
        .def(
            "isInside",
            +[](const T &mu, const Ref<const Vector3s> &x, double thresh) {
              return IceCreamCone::isInside(mu, x, thresh);
            },
            bp::args("mu", "x", "thresh"))
        .def(
            "isInside",
            +[](IceCreamCone &self, const Ref<const Vector3s> &x,
                double thresh) { return self.isInside(x, thresh); },
            bp::args("self", "x", "thresh"))
        .def(
            "isInsideDual",
            +[](IceCreamCone &self, const Ref<const Vector3s> &x,
                double thresh) { return self.isInsideDual(x, thresh); },
            bp::args("self", "x", "thresh"))
        .def(
            "projectHorizontal",
            +[](IceCreamCone &self, const Ref<const Vector3s> &x,
                Ref<Vector3s> x_out) { self.projectHorizontal(x, x_out); },
            bp::args("self", "x", "x_out"))
        .def(
            "computeContactComplementarity",
            +[](IceCreamCone &self, const Ref<const Vector3s> &lam,
                const Ref<const Vector3s> &v) {
              return self.computeContactComplementarity(lam, v);
            },
            bp::args("self", "v", "lam"))
        .def(
            "computeConicComplementarity",
            +[](IceCreamCone &self, const Ref<Vector3s> &lam,
                const Ref<const Vector3s> &v) {
              return self.computeConicComplementarity(lam, v);
            },
            bp::args("self", "v", "lam"))
        .def(
            "computeSignoriniComplementarity",
            +[](IceCreamCone &self, const Ref<const Vector3s> &lam,
                const Ref<const Vector3s> &v) {
              return self.computeSignoriniComplementarity(lam, v);
            },
            bp::args("self", "v", "lam"))
        .def(
            "computeDeSaxceCorrection",
            +[](IceCreamCone &self, const Ref<const Vector3s> &v,
                Ref<Vector3s> v_out) {
              self.computeDeSaxceCorrection(v, v_out);
            },
            bp::args("self", "v", "v_out"));
    eigenpy::StdVectorPythonVisitor<std::vector<IceCreamCone>>::expose(
        "StdVec_IceCreamCone");
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<PyramidCone>()) {
    bp::class_<PyramidCone, bp::bases<FrictionConstraint>>(
        "PyramidCone", "A quadratic cone constraint.", bp::no_init)
        .def(bp::init<double>(bp::args("self", "mu"), "Other constructor"))

        // Exposing attributes
        .DEF_RW_CLASS_ATTRIB(PyramidCone, mu_)
        //   Exposing methods
        .def(
            "project",
            +[](PyramidCone &self, const Ref<const Vector3s> &x,
                Ref<Vector3s> x_out) { self.project(x, x_out); },
            bp::args("self", "x", "x_out"))
        .def(
            "projectDual",
            +[](PyramidCone &self, const Ref<Vector3s> &x,
                Ref<Vector3s> x_out) { self.projectDual(x, x_out); },
            bp::args("self", "x", "x_out"))
        .def(
            "isInside",
            +[](PyramidCone &self, const Ref<const Vector3s> &x,
                double thresh) { return self.isInside(x, thresh); },
            bp::args("self", "x", "thresh"))
        .def(
            "projectHorizontal",
            +[](PyramidCone &self, const Ref<const Vector3s> &x,
                Ref<Vector3s> x_out) { self.projectHorizontal(x, x_out); },
            bp::args("self", "x", "x_out"))
        .def(
            "computeContactComplementarity",
            +[](PyramidCone &self, const Ref<const Vector3s> &lam,
                const Ref<const Vector3s> &v) {
              return self.computeContactComplementarity(lam, v);
            },
            bp::args("self", "v", "lam"))
        .def(
            "computeConicComplementarity",
            +[](PyramidCone &self, const Ref<const Vector3s> &lam,
                const Ref<const Vector3s> &v) {
              return self.computeConicComplementarity(lam, v);
            },
            bp::args("self", "v", "lam"))
        .def(
            "computeSignoriniComplementarity",
            +[](PyramidCone &self, const Ref<const Vector3s> &lam,
                const Ref<const Vector3s> &v) {
              return self.computeSignoriniComplementarity(lam, v);
            },
            bp::args("self", "v", "lam"))
        .def(
            "computeCoordinatesInD",
            +[](PyramidCone &self, const Ref<const Vector3s> &x,
                Ref<Vector3s> x_out) { self.computeCoordinatesInD(x, x_out); },
            bp::args("self", "x", "x_out"));
    eigenpy::StdVectorPythonVisitor<std::vector<PyramidCone>>::expose(
        "StdVec_PyramidCone");
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<
          cb::DelassusBase<T>>()) {
    bp::class_<cb::DelassusBase<T>, boost::noncopyable>(
        "DelassusBase", "A wrapper for the Delassus' matrix.", bp::no_init);
    bp::register_ptr_to_python<std::shared_ptr<cb::DelassusBase<T>>>();
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<DelassusDense>()) {
    bp::class_<DelassusDense, bp::bases<cb::DelassusBase<T>>>(
        "DelassusDense", "A dense evaluation of the Delassus matrix.",
        bp::no_init)
        .def(bp::init<>(bp::arg("self"), "Default constructor"))

        // Exposing attributes
        .DEF_RW_CLASS_ATTRIB(DelassusDense, G_)
        .DEF_RW_CLASS_ATTRIB(DelassusDense, nc_)
        .DEF_RW_CLASS_ATTRIB(DelassusDense, R_reg_)
        //   Exposing methods
        .def("evaluateDel", &DelassusDense::evaluateDel, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def(
            "applyOnTheRight",
            +[](DelassusDense &self, const Ref<const VectorXs> &x,
                Ref<VectorXs> x_out) { self.applyOnTheRight(x, x_out); },
            bp::args("self", "x", "x_out"))
        .def(
            "applyPerContactOnTheRight",
            +[](DelassusDense &self, int i, const Ref<const VectorXs> &x,
                Ref<VectorXs> x_out) {
              self.applyPerContactOnTheRight(i, x, x_out);
            },
            bp::args("self", "i", "x", "x_out"))
        .def(
            "applyPerContactNormalOnTheRight",
            +[](DelassusDense &self, int i, const Ref<const VectorXs> &x,
                double x_out) {
              self.applyPerContactNormalOnTheRight(i, x, x_out);
            },
            bp::args("self", "i", "x", "x_out"))
        .def(
            "applyPerContactTangentOnTheRight",
            +[](DelassusDense &self, int i, const Ref<const VectorXs> &x,
                Ref<VectorXs> x_out) {
              self.applyPerContactTangentOnTheRight(i, x, x_out);
            },
            bp::args("self", "i", "x", "x_out"))
        .def(
            "computeChol", +[](DelassusDense &m, T mu) { m.computeChol(mu); },
            bp::args("self", "mu"),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "computeChol",
            +[](DelassusDense &m, const Ref<const VectorXs> &mus) {
              m.computeChol(mus);
            },
            bp::args("self", "mus"),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "updateChol", +[](DelassusDense &m, T mu) { m.updateChol(mu); },
            bp::args("self", "mu"),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "updateChol",
            +[](DelassusDense &m, const Ref<const VectorXs> &mus) {
              m.updateChol(mus);
            },
            bp::args("self", "mus"),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](DelassusDense &self, const Ref<const VectorXs> &x,
                Ref<VectorXs> x_out) { self.solve(x, x_out); },
            bp::args("self", "x", "x_out"));
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<
          DelassusPinocchio>()) {

    bp::class_<DelassusPinocchio, bp::bases<cb::DelassusBase<T>>>(
        "DelassusPinocchio", "Delassus from pinocchio contact model.",
        bp::no_init)
        .def(bp::init<>(bp::arg("self"), "Default constructor"))
        .def(bp::init<Model, Data,
                      const DelassusPinocchio::RigidConstraintModelVector &,
                      const DelassusPinocchio::RigidConstraintDataVector &>(
            bp::args("self", "model", "data", "contact_models",
                     "contact_datas"),
            "Other constructor"))

        // Exposing attributes
        .DEF_RW_CLASS_ATTRIB(DelassusPinocchio, G_)
        .DEF_RW_CLASS_ATTRIB(DelassusPinocchio, nc_)
        .DEF_RW_CLASS_ATTRIB(DelassusPinocchio, R_reg_)
        .DEF_RW_CLASS_ATTRIB(DelassusPinocchio, contact_chol_)
        //   Exposing methods
        .def("evaluateDel", &DelassusPinocchio::evaluateDel, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def(
            "applyOnTheRight",
            +[](DelassusPinocchio &self, const Ref<const VectorXs> &x,
                Ref<VectorXs> x_out) { self.applyOnTheRight(x, x_out); },
            bp::args("self", "x", "x_out"))
        .def(
            "applyPerContactOnTheRight",
            +[](DelassusPinocchio &self, int i, const Ref<const VectorXs> &x,
                Ref<VectorXs> x_out) {
              self.applyPerContactOnTheRight(i, x, x_out);
            },
            bp::args("self", "i", "x", "x_out"))
        .def(
            "applyPerContactNormalOnTheRight",
            +[](DelassusPinocchio &self, int i, const Ref<const VectorXs> &x,
                double x_out) {
              self.applyPerContactNormalOnTheRight(i, x, x_out);
            },
            bp::args("self", "i", "x", "x_out"))
        .def(
            "applyPerContactTangentOnTheRight",
            +[](DelassusPinocchio &self, int i, const Ref<const VectorXs> &x,
                Ref<VectorXs> x_out) {
              self.applyPerContactTangentOnTheRight(i, x, x_out);
            },
            bp::args("self", "i", "x", "x_out"))
        .def(
            "computeChol",
            +[](DelassusPinocchio &m, T mu) { m.computeChol(mu); },
            bp::args("self", "mu"),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "computeChol",
            +[](DelassusPinocchio &m, const Ref<const VectorXs> &mus) {
              m.computeChol(mus);
            },
            bp::args("self", "mus"),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "updateChol", +[](DelassusPinocchio &m, T mu) { m.updateChol(mu); },
            bp::args("self", "mu"),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "updateChol",
            +[](DelassusPinocchio &m, const Ref<const VectorXs> &mus) {
              m.updateChol(mus);
            },
            bp::args("self", "mus"),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](DelassusPinocchio &self, const Ref<const VectorXs> &x,
                Ref<VectorXs> x_out) { self.solve(x, x_out); },
            bp::args("self", "x", "x_out"));
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<
          ContactProblem<cb::IceCreamCone>>()) {
    T(ContactProblem<cb::IceCreamCone>::*computeContactComplementarity1)
    (const Ref<const VectorXs> &) =
        &ContactProblem<cb::IceCreamCone>::computeContactComplementarity;
    T(ContactProblem<cb::IceCreamCone>::*computeConicComplementarity1)
    (const Ref<const VectorXs> &) =
        &ContactProblem<cb::IceCreamCone>::computeConicComplementarity;
    T(ContactProblem<cb::IceCreamCone>::*computeSignoriniComplementarity1)
    (const Ref<const VectorXs> &) =
        &ContactProblem<cb::IceCreamCone>::computeSignoriniComplementarity;
    T(ContactProblem<cb::IceCreamCone>::*computeLinearComplementarity1)
    (const Ref<const VectorXs> &) =
        &ContactProblem<cb::IceCreamCone>::computeLinearComplementarity;
    T(ContactProblem<cb::IceCreamCone>::*computeContactComplementarity2)
    (const Ref<const VectorXs> &, const Ref<const VectorXs> &) const =
        &ContactProblem<cb::IceCreamCone>::computeContactComplementarity;
    T(ContactProblem<cb::IceCreamCone>::*computeConicComplementarity2)
    (const Ref<const VectorXs> &, const Ref<const VectorXs> &) const =
        &ContactProblem<cb::IceCreamCone>::computeConicComplementarity;
    T(ContactProblem<cb::IceCreamCone>::*computeSignoriniComplementarity2)
    (const Ref<const VectorXs> &, const Ref<const VectorXs> &) const =
        &ContactProblem<cb::IceCreamCone>::computeSignoriniComplementarity;
    T(ContactProblem<cb::IceCreamCone>::*computeLinearComplementarity2)
    (const Ref<const VectorXs> &, const Ref<const VectorXs> &) =
        &ContactProblem<cb::IceCreamCone>::computeLinearComplementarity;
    void (ContactProblem<cb::IceCreamCone>::*computeLCP1)(const T) =
        &ContactProblem<cb::IceCreamCone>::computeLCP;
    void (ContactProblem<cb::IceCreamCone>::*computeLCP2)(
        const Ref<const VectorXs> &) =
        &ContactProblem<cb::IceCreamCone>::computeLCP;
    void (ContactProblem<cb::IceCreamCone>::*computeTangentLCP1)(
        const Ref<const VectorXs> &, const T) =
        &ContactProblem<cb::IceCreamCone>::computeTangentLCP;
    void (ContactProblem<cb::IceCreamCone>::*computeTangentLCP2)(
        const Ref<const VectorXs> &, const Ref<const VectorXs> &) =
        &ContactProblem<cb::IceCreamCone>::computeTangentLCP;
    void (ContactProblem<cb::IceCreamCone>::*computeInscribedLCP1)(
        const Ref<const VectorXs> &, const T) =
        &ContactProblem<cb::IceCreamCone>::computeInscribedLCP;
    void (ContactProblem<cb::IceCreamCone>::*computeInscribedLCP2)(
        const Ref<const VectorXs> &, const Ref<const VectorXs> &) =
        &ContactProblem<cb::IceCreamCone>::computeInscribedLCP;
    bp::class_<ContactProblem<cb::IceCreamCone>>(
        "ContactProblem", "A contact problem.", boost::python::no_init)
        .def(bp::init<const Ref<const MatrixXs> &, const Ref<const VectorXs> &,
                      const std::vector<T> &, const T>(
            (bp::arg("self"), bp::arg("G"), bp::arg("g"), bp::arg("mus"),
             bp::arg("comp") = 0.),
            "Constructor of the dual problem using the dense evaluation of the "
            "Delassus' matrix."))
        .def(bp::init<const Ref<const MatrixXs> &, const Ref<const VectorXs> &,
                      const std::vector<T> &, const Ref<const VectorXs> &>(
            (bp::arg("self"), bp::arg("G"), bp::arg("g"), bp::arg("mus"),
             bp::arg("comp")),
            "Constructor of the dual problem using the dense evaluation of the "
            "Delassus' matrix."))
        .def(bp::init<std::shared_ptr<cb::DelassusBase<T>>,
                      const Ref<const VectorXs> &, const std::vector<T> &,
                      const T>(
            (bp::arg("self"), bp::arg("Del"), bp::arg("g"), bp::arg("mus"),
             bp::arg("comp") = 0.),
            "Constructor of the dual problem using the cholesky factorization "
            "of the Delassus "
            "matrix."))
        .def(bp::init<std::shared_ptr<cb::DelassusBase<T>>,
                      const Ref<const VectorXs> &, const std::vector<T> &,
                      const Ref<const VectorXs> &>(
            (bp::arg("self"), bp::arg("Del"), bp::arg("g"), bp::arg("mus"),
             bp::arg("comp")),
            "Constructor of the dual problem using the cholesky factorization "
            "of the Delassus "
            "matrix."))
        .def(bp::init<const Ref<const MatrixXs> &, const Ref<const MatrixXs> &,
                      const Ref<const VectorXs> &, const Ref<const VectorXs> &,
                      const std::vector<T> &, const T>(
            (bp::arg("self"), bp::arg("M"), bp::arg("J"), bp::arg("dqf"),
             bp::arg("vstar"), bp::arg("mus"), bp::arg("comp") = 0.),
            "Constructor of the primal contact problem on joint velocities."))
        .def(bp::init<const Ref<const MatrixXs> &, const Ref<const MatrixXs> &,
                      const Ref<const VectorXs> &, const Ref<const VectorXs> &,
                      const std::vector<T> &, const Ref<const VectorXs> &>(
            (bp::arg("self"), bp::arg("M"), bp::arg("J"), bp::arg("dqf"),
             bp::arg("vstar"), bp::arg("mus"), bp::arg("comp")),
            "Constructor of the primal contact problem on joint velocities."))
        .def(bp::init<const Ref<const MatrixXs> &, const Ref<const VectorXs> &,
                      const Ref<const MatrixXs> &, const Ref<const MatrixXs> &,
                      const Ref<const VectorXs> &, const Ref<const VectorXs> &,
                      const std::vector<T> &, const T>(
            (bp::arg("self"), bp::arg("G"), bp::arg("g"), bp::arg("M"),
             bp::arg("J"), bp::arg("dqf"), bp::arg("vstar"), bp::arg("mus"),
             bp::arg("comp") = 0.),
            "Constructor of the primal-dual problem using the dense evaluation "
            "of the Delassus' matrix."))
        .def(bp::init<const Ref<const MatrixXs> &, const Ref<const VectorXs> &,
                      const Ref<const MatrixXs> &, const Ref<const MatrixXs> &,
                      const Ref<const VectorXs> &, const Ref<const VectorXs> &,
                      const std::vector<T> &, const Ref<const VectorXs> &>(
            (bp::arg("self"), bp::arg("G"), bp::arg("g"), bp::arg("M"),
             bp::arg("J"), bp::arg("dqf"), bp::arg("vstar"), bp::arg("mus"),
             bp::arg("comp")),
            "Constructor of the primal-dual problem using the dense evaluation "
            "of the Delassus' matrix."))
        .def(bp::init<std::shared_ptr<cb::DelassusBase<T>>,
                      const Ref<const VectorXs> &, const Ref<const MatrixXs> &,
                      const Ref<const MatrixXs> &, const Ref<const VectorXs> &,
                      const Ref<const VectorXs> &, const std::vector<T> &,
                      const T>(
            (bp::arg("self"), bp::arg("Del"), bp::arg("g"), bp::arg("M"),
             bp::arg("J"), bp::arg("dqf"), bp::arg("vstar"), bp::arg("mus"),
             bp::arg("comp") = 0.),
            "Constructor  of the primal-dual problem using the cholesky "
            "factorization of the Delassus "
            "matrix."))
        .def(bp::init<std::shared_ptr<cb::DelassusBase<T>>,
                      const Ref<const VectorXs> &, const Ref<const MatrixXs> &,
                      const Ref<const MatrixXs> &, const Ref<const VectorXs> &,
                      const Ref<const VectorXs> &, const std::vector<T> &,
                      const Ref<const VectorXs> &>(
            (bp::arg("self"), bp::arg("Del"), bp::arg("g"), bp::arg("M"),
             bp::arg("J"), bp::arg("dqf"), bp::arg("vstar"), bp::arg("mus"),
             bp::arg("comp")),
            "Constructor  of the primal-dual problem using the cholesky "
            "factorization of the Delassus "
            "matrix."))

        // Exposing attributes
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::IceCreamCone>, Del_)
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::IceCreamCone>, g_)
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::IceCreamCone>, M_)
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::IceCreamCone>, J_)
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::IceCreamCone>, R_comp_)
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::IceCreamCone>, dqf_)
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::IceCreamCone>, vstar_)
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::IceCreamCone>, nc_)
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::IceCreamCone>,
                             contact_constraints_)
        //   Exposing methods
        .def(
            "project",
            +[](const std::vector<T> &mus, const Ref<VectorXs> &lam,
                Ref<VectorXs> lam_out) {
              ContactProblem<cb::IceCreamCone>::project(mus, lam, lam_out);
            },
            bp::args("mus", "lam", "lam_out"))
        .def(
            "project",
            +[](ContactProblem<cb::IceCreamCone> &self,
                const Ref<VectorXs> &lam,
                Ref<VectorXs> lam_out) { self.project(lam, lam_out); },
            bp::args("self", "lam", "lam_out"))
        .def(
            "projectDual",
            +[](ContactProblem<cb::IceCreamCone> &self, const Ref<VectorXs> v,
                Ref<VectorXs> v_out) { self.projectDual(v, v_out); },
            bp::args("self", "v", "v_out"))
        .def("isInside", &ContactProblem<cb::IceCreamCone>::isInside,
             bp::args("self", "lam", "thresh"),
             bp::return_value_policy<bp::return_by_value>())
        .def("isInsideDual", &ContactProblem<cb::IceCreamCone>::isInsideDual,
             bp::args("self", "v", "thresh"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computePerContactContactComplementarity",
             &ContactProblem<
                 cb::IceCreamCone>::computePerContactContactComplementarity,
             bp::args("self", "i", "lam", "v"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeContactComplementarity", computeContactComplementarity1,
             bp::args("self", "lam"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeContactComplementarity", computeContactComplementarity2,
             bp::args("self", "lam", "v"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computePerContactConicComplementarity",
             &ContactProblem<
                 cb::IceCreamCone>::computePerContactConicComplementarity,
             bp::args("self", "i", "lam", "v"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeConicComplementarity", computeConicComplementarity1,
             bp::args("self", "lam"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeConicComplementarity", computeConicComplementarity2,
             bp::args("self", "lam", "v"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computePerContactSignoriniComplementarity",
             &ContactProblem<
                 cb::IceCreamCone>::computePerContactSignoriniComplementarity,
             bp::args("self", "i", "lam", "v"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeSignoriniComplementarity",
             computeSignoriniComplementarity1, bp::args("self", "lam"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeSignoriniComplementarity",
             computeSignoriniComplementarity2, bp::args("self", "lam", "v"),
             bp::return_value_policy<bp::return_by_value>())
        .def(
            "computeDeSaxceCorrection",
            +[](ContactProblem<cb::IceCreamCone> &self, const Ref<VectorXs> v,
                Ref<VectorXs> v_out) {
              self.computeDeSaxceCorrection(v, v_out);
            },
            bp::args("self", "v", "v_out"))
        .def("computeLinearComplementarity", computeLinearComplementarity1,
             bp::args("self", "lam"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeLinearComplementarity", computeLinearComplementarity2,
             bp::args("self", "lam", "v"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeLCP", computeLCP1, bp::args("self"),
             bp::args("eps_reg") = 0.)
        .def("computeLCP", computeLCP2, bp::args("self"), bp::args("R_reg"))
        .def("computeTangentLCP", computeTangentLCP1,
             bp::args("self", "lam", "eps_reg"))
        .def("computeTangentLCP", computeTangentLCP2,
             bp::args("self", "lam", "R_reg"))
        .def("computeInscribedLCP", computeInscribedLCP1,
             bp::args("self", "lam", "eps_reg"))
        .def("computeInscribedLCP", computeInscribedLCP2,
             bp::args("self", "lam", "R_reg"));
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<
          ContactProblem<cb::PyramidCone>>()) {
    T(ContactProblem<cb::PyramidCone>::*computeContactComplementarity1)
    (const Ref<const VectorXs> &) =
        &ContactProblem<cb::PyramidCone>::computeContactComplementarity;
    T(ContactProblem<cb::PyramidCone>::*computeConicComplementarity1)
    (const Ref<const VectorXs> &) =
        &ContactProblem<cb::PyramidCone>::computeConicComplementarity;
    T(ContactProblem<cb::PyramidCone>::*computeSignoriniComplementarity1)
    (const Ref<const VectorXs> &) =
        &ContactProblem<cb::PyramidCone>::computeSignoriniComplementarity;
    void (ContactProblem<cb::PyramidCone>::*computeLCPSolution1)(
        const Ref<const VectorXs> &, VectorRef) =
        &ContactProblem<cb::PyramidCone>::computeLCPSolution;
    T(ContactProblem<cb::PyramidCone>::*computeContactComplementarity2)
    (const Ref<const VectorXs> &, const Ref<const VectorXs> &) const =
        &ContactProblem<cb::PyramidCone>::computeContactComplementarity;
    T(ContactProblem<cb::PyramidCone>::*computeConicComplementarity2)
    (const Ref<const VectorXs> &, const Ref<const VectorXs> &) const =
        &ContactProblem<cb::PyramidCone>::computeConicComplementarity;
    T(ContactProblem<cb::PyramidCone>::*computeSignoriniComplementarity2)
    (const Ref<const VectorXs> &, const Ref<const VectorXs> &) const =
        &ContactProblem<cb::PyramidCone>::computeSignoriniComplementarity;
    void (ContactProblem<cb::PyramidCone>::*computeLCPSolution2)(
        const Ref<const VectorXs> &, const Ref<const VectorXs> &, VectorRef)
        const = &ContactProblem<cb::PyramidCone>::computeLCPSolution;
    void (ContactProblem<cb::PyramidCone>::*computeLCP1)(const T) =
        &ContactProblem<cb::PyramidCone>::computeLCP;
    void (ContactProblem<cb::PyramidCone>::*computeLCP2)(
        const Ref<const VectorXs> &) =
        &ContactProblem<cb::PyramidCone>::computeLCP;
    bp::class_<ContactProblem<cb::PyramidCone>>(
        "LinearContactProblem", "A contact problem.", boost::python::no_init)
        .def(bp::init<const Ref<const MatrixXs> &, const Ref<const VectorXs> &,
                      const std::vector<T> &, const T>(
            (bp::arg("self"), bp::arg("G"), bp::arg("g"), bp::arg("mus"),
             bp::arg("comp") = 0.),
            "Other constructor"))
        .def(bp::init<const Ref<const MatrixXs> &, const Ref<const VectorXs> &,
                      const std::vector<T> &, const Ref<const VectorXs> &>(
            (bp::arg("self"), bp::arg("G"), bp::arg("g"), bp::arg("mus"),
             bp::arg("comp")),
            "Other constructor"))
        .def(bp::init<std::shared_ptr<cb::DelassusBase<T>>,
                      const Ref<const VectorXs> &, const std::vector<T> &,
                      const T>(
            (bp::arg("self"), bp::arg("Del"), bp::arg("g"), bp::arg("mus"),
             bp::arg("comp") = 0.),
            "Constructor using the cholesky factorization of the Delassus "
            "matrix."))
        .def(bp::init<std::shared_ptr<cb::DelassusBase<T>>,
                      const Ref<const VectorXs> &, const std::vector<T> &,
                      const Ref<const VectorXs> &>(
            (bp::arg("self"), bp::arg("Del"), bp::arg("g"), bp::arg("mus"),
             bp::arg("comp")),
            "Constructor using the cholesky factorization of the Delassus "
            "matrix."))

        // Exposing attributes
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::PyramidCone>, Del_)
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::PyramidCone>, R_comp_)
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::PyramidCone>, g_)
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::PyramidCone>, nc_)
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::PyramidCone>,
                             contact_constraints_)
        // Exposing LCP attributes
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::PyramidCone>, A_)
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::PyramidCone>, b_)
        .DEF_RW_CLASS_ATTRIB(ContactProblem<cb::PyramidCone>, UD_)
        //   Exposing methods
        .def("isInside", &ContactProblem<cb::PyramidCone>::isInside,
             bp::args("self", "lam", "thresh"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computePerContactContactComplementarity",
             &ContactProblem<
                 cb::PyramidCone>::computePerContactContactComplementarity,
             bp::args("self", "i", "lam", "v"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeContactComplementarity", computeContactComplementarity1,
             bp::args("self", "lam"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeContactComplementarity", computeContactComplementarity2,
             bp::args("self", "lam", "v"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computePerContactConicComplementarity",
             &ContactProblem<
                 cb::PyramidCone>::computePerContactConicComplementarity,
             bp::args("self", "i", "lam", "v"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeConicComplementarity", computeConicComplementarity1,
             bp::args("self", "lam"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeConicComplementarity", computeConicComplementarity2,
             bp::args("self", "lam", "v"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computePerContactSignoriniComplementarity",
             &ContactProblem<
                 cb::PyramidCone>::computePerContactSignoriniComplementarity,
             bp::args("self", "i", "lam", "v"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeSignoriniComplementarity",
             computeSignoriniComplementarity1, bp::args("self", "lam"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeSignoriniComplementarity",
             computeSignoriniComplementarity2, bp::args("self", "lam", "v"),
             bp::return_value_policy<bp::return_by_value>())
        .def("setLCP", &ContactProblem<cb::PyramidCone>::setLCP,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        // .def("computeLCP", &ContactProblem<cb::PyramidCone>::computeLCP,
        //      bp::args("self"))
        .def("computeLCP", computeLCP1, bp::args("self"),
             bp::args("eps_reg") = 0.)
        .def("computeLCP", computeLCP2, bp::args("self"), bp::args("R_reg"))
        // .def("computeTangentLCP",
        //      &ContactProblem<cb::PyramidCone>::computeTangentLCP,
        //      bp::args("self", "lam"))
        .def("computeLCPSolution", computeLCPSolution1,
             bp::args("self", "lam", "lam_lcp"));
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<Statistics>()) {
    bp::class_<Statistics>(
        "Statistics", "Statistics collected when solving the contact problem.",
        boost::python::no_init)
        .def(bp::init<>(bp::args("self"), "Default constructor"))
        // Exposing attributes
        .DEF_R_CLASS_ATTRIB(Statistics, stop_)
        .DEF_R_CLASS_ATTRIB(Statistics, rel_stop_)
        .DEF_R_CLASS_ATTRIB(Statistics, comp_)
        .DEF_R_CLASS_ATTRIB(Statistics, prim_feas_)
        .DEF_R_CLASS_ATTRIB(Statistics, dual_feas_)
        .DEF_R_CLASS_ATTRIB(Statistics, sig_comp_)
        .DEF_R_CLASS_ATTRIB(Statistics, ncp_comp_)
        // Exposing methods
        .def("addStop", &Statistics::addStop, bp::args("self", "stop"),
             bp::return_value_policy<bp::return_by_value>())
        .def("addRelStop", &Statistics::addRelStop,
             bp::args("self", "rel_stop"),
             bp::return_value_policy<bp::return_by_value>())
        .def("addComp", &Statistics::addComp, bp::args("self", "comp"),
             bp::return_value_policy<bp::return_by_value>())
        .def("addSigComp", &Statistics::addSigComp,
             bp::args("self", "sig_comp"),
             bp::return_value_policy<bp::return_by_value>())
        .def("addNcpComp", &Statistics::addNcpComp,
             bp::args("self", "ncp_comp"),
             bp::return_value_policy<bp::return_by_value>())
        .def("addPrimFeas", &Statistics::addPrimFeas,
             bp::args("self", "prim_feas"),
             bp::return_value_policy<bp::return_by_value>())
        .def("addDualFeas", &Statistics::addDualFeas,
             bp::args("self", "dual_feas"),
             bp::return_value_policy<bp::return_by_value>())
        .def("reset", &Statistics::reset, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>());
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<
          ContactSolverSettings>()) {
    bp::class_<ContactSolverSettings, boost::noncopyable>(
        "ContactSolverSettings", "Solver settings.", boost::python::no_init)
        .def(bp::init<>(bp::args("self"), "Default constructor"))
        // Exposing attributes
        .DEF_RW_CLASS_ATTRIB(ContactSolverSettings, stop_)
        .DEF_RW_CLASS_ATTRIB(ContactSolverSettings, rel_stop_)
        .DEF_RW_CLASS_ATTRIB(ContactSolverSettings, th_stop_)
        .DEF_RW_CLASS_ATTRIB(ContactSolverSettings, rel_th_stop_)
        .DEF_RW_CLASS_ATTRIB(ContactSolverSettings, max_iter_)
        .DEF_RW_CLASS_ATTRIB(ContactSolverSettings, timings_)
        .DEF_RW_CLASS_ATTRIB(ContactSolverSettings, statistics_);
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<
          NCPPGSSolver<cb::IceCreamCone>>()) {
    bp::class_<NCPPGSSolver<cb::IceCreamCone>, boost::noncopyable>(
        "NCPPGSSolver", "A pgs solver solving the NCP problem.",
        boost::python::no_init)
        .def(bp::init<>(bp::args("self"), "Default constructor"))
        // Exposing attributes
        .DEF_R_CLASS_ATTRIB(NCPPGSSolver<cb::IceCreamCone>, n_iter_)
        .DEF_R_CLASS_ATTRIB(NCPPGSSolver<cb::IceCreamCone>, max_iter_)
        .DEF_R_CLASS_ATTRIB(NCPPGSSolver<cb::IceCreamCone>, th_stop_)
        .DEF_R_CLASS_ATTRIB(NCPPGSSolver<cb::IceCreamCone>, stop_)
        .DEF_R_CLASS_ATTRIB(NCPPGSSolver<cb::IceCreamCone>, rel_th_stop_)
        .DEF_R_CLASS_ATTRIB(NCPPGSSolver<cb::IceCreamCone>, rel_stop_)
        .DEF_R_CLASS_ATTRIB(NCPPGSSolver<cb::IceCreamCone>, ncp_comp_)
        .DEF_R_CLASS_ATTRIB(NCPPGSSolver<cb::IceCreamCone>, stats_)
        // Exposing methods
        .def("setProblem", &NCPPGSSolver<cb::IceCreamCone>::setProblem,
             bp::args("self", "problem"),
             bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](NCPPGSSolver<cb::IceCreamCone> &m,
                const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<const VectorXs> &lam0,
                ContactSolverSettings &settings, T over_relax, T eps_reg) {
              return m.solve(prob, lam0, settings, over_relax, eps_reg);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("settings"), bp::arg("over_relax") = 1.0,
             bp::arg("eps_reg") = 0.0),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](NCPPGSSolver<cb::IceCreamCone> &m,
                const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<const VectorXs> &lam0,
                ContactSolverSettings &settings,
                const Ref<const VectorXs> &R_reg, T over_relax) {
              return m.solve(prob, lam0, settings, R_reg, over_relax);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("settings"), bp::arg("R_reg"),
             bp::arg("over_relax") = 1.0),
            bp::return_value_policy<bp::return_by_value>())
        .def("getSolution", &NCPPGSSolver<cb::IceCreamCone>::getSolution,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getDualSolution",
             &NCPPGSSolver<cb::IceCreamCone>::getDualSolution, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("resetStats", &NCPPGSSolver<cb::IceCreamCone>::resetStats,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getCPUTimes", &NCPPGSSolver<cb::IceCreamCone>::getCPUTimes,
             bp::args("self"))
        .def("vjp_approx", &NCPPGSSolver<cb::IceCreamCone>::vjp_approx,
             (bp::arg("self"), bp::arg("problem"), bp::arg("dL_dlam"),
              bp::arg("settings"), bp::arg("rho") = 1e-6,
              bp::arg("eps_reg") = 0.))
        .def("vjp_fd", &NCPPGSSolver<cb::IceCreamCone>::vjp_fd,
             (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
               bp::arg("dL_dlam"), bp::arg("settings"), bp::arg("delta") = 1e-8))
        .def("getdLdmus", &NCPPGSSolver<cb::IceCreamCone>::getdLdmus,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getdLdDel", &NCPPGSSolver<cb::IceCreamCone>::getdLdDel,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getdLdg", &NCPPGSSolver<cb::IceCreamCone>::getdLdg,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
#ifdef DIFFCONTACT_WITH_CPPAD
        .def("vjp_cppad", &NCPPGSSolver<cb::IceCreamCone>::vjp_cppad,
             (bp::arg("self"), bp::arg("problem"), bp::arg("dL_dlam"),
              bp::arg("settings")))
        .def("jvp_cppad", &NCPPGSSolver<cb::IceCreamCone>::jvp_cppad,
             (bp::arg("self"), bp::arg("problem"), bp::arg("dG_dtheta"),
              bp::arg("dg_dtheta"), bp::arg("dmus_dtheta"), bp::arg("settings"),
              bp::arg("eps_reg") = 0.))
#endif
        ;
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<
          NCPPGSSolver<cb::PyramidCone>>()) {
    bp::class_<NCPPGSSolver<cb::PyramidCone>, boost::noncopyable>(
        "LCPPGSSolver", "A pgs solver solving the LCP problem.",
        boost::python::no_init)
        .def(bp::init<>(bp::args("self"), "Default constructor"))
        // Exposing attributes
        .DEF_R_CLASS_ATTRIB(NCPPGSSolver<cb::PyramidCone>, n_iter_)
        .DEF_R_CLASS_ATTRIB(NCPPGSSolver<cb::PyramidCone>, max_iter_)
        .DEF_R_CLASS_ATTRIB(NCPPGSSolver<cb::PyramidCone>, th_stop_)
        .DEF_R_CLASS_ATTRIB(NCPPGSSolver<cb::PyramidCone>, stop_)
        .DEF_R_CLASS_ATTRIB(NCPPGSSolver<cb::PyramidCone>, rel_th_stop_)
        .DEF_R_CLASS_ATTRIB(NCPPGSSolver<cb::PyramidCone>, rel_stop_)
        .DEF_R_CLASS_ATTRIB(NCPPGSSolver<cb::PyramidCone>, ncp_comp_)
        .DEF_R_CLASS_ATTRIB(NCPPGSSolver<cb::PyramidCone>, stats_)
        // Exposing methods
        .def("setProblem", &NCPPGSSolver<cb::PyramidCone>::setProblem,
             bp::args("self", "problem"),
             bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](NCPPGSSolver<cb::PyramidCone> &m,
                const ContactProblem<cb::PyramidCone> &prob,
                const Ref<const VectorXs> &lam0,
                ContactSolverSettings &settings, T over_relax, T eps_reg) {
              return m.solve(prob, lam0, settings, over_relax, eps_reg);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("settings"), bp::arg("over_relax") = 1.0,
             bp::arg("eps_reg") = 0.),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](NCPPGSSolver<cb::PyramidCone> &m,
                const ContactProblem<cb::PyramidCone> &prob,
                const Ref<const VectorXs> &lam0,
                ContactSolverSettings &settings,
                const Ref<const VectorXs> &R_reg, T over_relax) {
              return m.solve(prob, lam0, settings, R_reg, over_relax);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("settings"), bp::arg("R_reg"),
             bp::arg("over_relax") = 1.0),
            bp::return_value_policy<bp::return_by_value>())
        .def("getSolution", &NCPPGSSolver<cb::PyramidCone>::getSolution,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getDualSolution", &NCPPGSSolver<cb::PyramidCone>::getDualSolution,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("resetStats", &NCPPGSSolver<cb::PyramidCone>::resetStats,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getCPUTimes", &NCPPGSSolver<cb::PyramidCone>::getCPUTimes,
             bp::args("self"))
        .def("vjp_fd", &NCPPGSSolver<cb::PyramidCone>::vjp_fd,
             (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
               bp::arg("dL_dlam"), bp::arg("settings"), bp::arg("delta") = 1e-8))
        .def("getdLdmus", &NCPPGSSolver<cb::PyramidCone>::getdLdmus,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getdLdDel", &NCPPGSSolver<cb::PyramidCone>::getdLdDel,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getdLdg", &NCPPGSSolver<cb::PyramidCone>::getdLdg,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
#ifdef DIFFCONTACT_WITH_CPPAD
        .def("vjp_cppad", &NCPPGSSolver<cb::IceCreamCone>::vjp_cppad,
             (bp::arg("self"), bp::arg("problem"), bp::arg("dL_dlam"),
              bp::arg("settings")))
        .def("jvp_cppad", &NCPPGSSolver<cb::IceCreamCone>::jvp_cppad,
             (bp::arg("self"), bp::arg("problem"), bp::arg("dG_dtheta"),
              bp::arg("dg_dtheta"), bp::arg("dmus_dtheta"), bp::arg("settings"),
              bp::arg("eps_reg") = 0.))
#endif
        ;
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<LCPQPSolver>()) {
    bp::class_<LCPQPSolver, boost::noncopyable>(
        "LCPQPSolver", "A QP solver solving the LCP problem.",
        boost::python::no_init)
        .def(bp::init<>(bp::args("self"), "Default constructor"))
        // Exposing attributes
        .DEF_R_CLASS_ATTRIB(LCPQPSolver, n_iter_)
        .DEF_R_CLASS_ATTRIB(LCPQPSolver, max_iter_)
        .DEF_R_CLASS_ATTRIB(LCPQPSolver, th_stop_)
        .DEF_R_CLASS_ATTRIB(LCPQPSolver, stop_)
        .DEF_R_CLASS_ATTRIB(LCPQPSolver, prim_feas_)
        .DEF_R_CLASS_ATTRIB(LCPQPSolver, dual_feas_)
        .DEF_R_CLASS_ATTRIB(LCPQPSolver, rel_th_stop_)
        .DEF_R_CLASS_ATTRIB(LCPQPSolver, rel_stop_)
        .DEF_R_CLASS_ATTRIB(LCPQPSolver, comp_)
        .DEF_R_CLASS_ATTRIB(LCPQPSolver, stats_)
        // Exposing methods
        .def("setProblem", &LCPQPSolver::setProblem,
             bp::args("self", "problem"),
             bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](LCPQPSolver &m, ContactProblem<cb::PyramidCone> &prob,
                const Ref<const VectorXs> &lam0,
                ContactSolverSettings &settings, T rho, T eps_reg) {
              return m.solve(prob, lam0, settings, rho, eps_reg);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("settings"), bp::arg("rho") = 1.0,
             bp::arg("eps_reg") = 0.0),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](LCPQPSolver &m, ContactProblem<cb::PyramidCone> &prob,
                const Ref<const VectorXs> &lam0,
                ContactSolverSettings &settings,
                const Ref<const VectorXs> &R_reg,
                T rho) { return m.solve(prob, lam0, settings, R_reg, rho); },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("settings"), bp::arg("R_reg"), bp::arg("rho") = 1.0),
            bp::return_value_policy<bp::return_by_value>())
        .def("getSolution", &LCPQPSolver::getSolution, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getDualSolution", &LCPQPSolver::getDualSolution, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("resetStats", &LCPQPSolver::resetStats, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getCPUTimes", &LCPQPSolver::getCPUTimes, bp::args("self"))
        .def("vjp", &LCPQPSolver::vjp,
             (bp::arg("self"), bp::arg("problem"), bp::arg("dL_dlam"),
              bp::arg("settings")))
        .def("vjp_fd", &LCPQPSolver::vjp_fd,
             (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"), bp::arg("dL_dlam"),
              bp::arg("settings"), bp::arg("delta") = 1e-8))
        .def("getdLdmus", &LCPQPSolver::getdLdmus, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getdLdDel", &LCPQPSolver::getdLdDel, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getdLdg", &LCPQPSolver::getdLdg, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>());
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<
          NCPStagProjSolver<cb::IceCreamCone>>()) {
    bp::class_<NCPStagProjSolver<cb::IceCreamCone>, boost::noncopyable>(
        "NCPStagProjSolver",
        "A staggered projection solver solving the NCP problem.",
        boost::python::no_init)
        .def(bp::init<>(bp::args("self"), "Default constructor"))
        // Exposing attributes
        .DEF_R_CLASS_ATTRIB(NCPStagProjSolver<cb::IceCreamCone>, n_iter_)
        .DEF_R_CLASS_ATTRIB(NCPStagProjSolver<cb::IceCreamCone>, max_iter_)
        .DEF_R_CLASS_ATTRIB(NCPStagProjSolver<cb::IceCreamCone>, th_stop_)
        .DEF_R_CLASS_ATTRIB(NCPStagProjSolver<cb::IceCreamCone>, stop_)
        .DEF_R_CLASS_ATTRIB(NCPStagProjSolver<cb::IceCreamCone>, rel_th_stop_)
        .DEF_R_CLASS_ATTRIB(NCPStagProjSolver<cb::IceCreamCone>, rel_stop_)
        .DEF_R_CLASS_ATTRIB(NCPStagProjSolver<cb::IceCreamCone>, comp_)
        .DEF_R_CLASS_ATTRIB(NCPStagProjSolver<cb::IceCreamCone>, stats_)
        // Exposing methods
        .def("setProblem", &NCPStagProjSolver<cb::IceCreamCone>::setProblem,
             bp::args("self", "problem"),
             bp::return_value_policy<bp::return_by_value>())
        .def("solve", &NCPStagProjSolver<cb::IceCreamCone>::solve,
             (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
              bp::arg("settings"), bp::arg("max_inner_iter") = 100,
              bp::arg("rho") = 1e-6, bp::arg("over_relax") = 1.0),
             bp::return_value_policy<bp::return_by_value>())
        .def("getSolution", &NCPStagProjSolver<cb::IceCreamCone>::getSolution,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("resetStats", &NCPStagProjSolver<cb::IceCreamCone>::resetStats,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getCPUTimes", &NCPStagProjSolver<cb::IceCreamCone>::getCPUTimes,
             bp::args("self"))
        .def("vjp_fd", &NCPStagProjSolver<cb::IceCreamCone>::vjp_fd,
             (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
              bp::arg("settings"), bp::arg("dL_dlam"), bp::arg("delta") = 1e-8))
        .def("getdLdmus", &NCPStagProjSolver<cb::IceCreamCone>::getdLdmus,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getdLdDel", &NCPStagProjSolver<cb::IceCreamCone>::getdLdDel,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getdLdg", &NCPStagProjSolver<cb::IceCreamCone>::getdLdg,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>());
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<
          NCPStagProjSolver<cb::PyramidCone>>()) {
    bp::class_<NCPStagProjSolver<cb::PyramidCone>, boost::noncopyable>(
        "LCPStagProjSolver",
        "A staggered projection solver solving the LCP problem.",
        boost::python::no_init)
        .def(bp::init<>(bp::args("self"), "Default constructor"))
        // Exposing attributes
        .DEF_R_CLASS_ATTRIB(NCPStagProjSolver<cb::PyramidCone>, n_iter_)
        .DEF_R_CLASS_ATTRIB(NCPStagProjSolver<cb::PyramidCone>, max_iter_)
        .DEF_R_CLASS_ATTRIB(NCPStagProjSolver<cb::PyramidCone>, th_stop_)
        .DEF_R_CLASS_ATTRIB(NCPStagProjSolver<cb::PyramidCone>, stop_)
        .DEF_R_CLASS_ATTRIB(NCPStagProjSolver<cb::PyramidCone>, rel_th_stop_)
        .DEF_R_CLASS_ATTRIB(NCPStagProjSolver<cb::PyramidCone>, rel_stop_)
        .DEF_R_CLASS_ATTRIB(NCPStagProjSolver<cb::PyramidCone>, comp_)
        .DEF_R_CLASS_ATTRIB(NCPStagProjSolver<cb::PyramidCone>, stats_)
        // Exposing methods
        .def("setProblem", &NCPStagProjSolver<cb::PyramidCone>::setProblem,
             bp::args("self", "problem"),
             bp::return_value_policy<bp::return_by_value>())
        .def("solve", &NCPStagProjSolver<cb::PyramidCone>::solve,
             (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
              bp::arg("settings"), bp::arg("max_inner_iter") = 100,
              bp::arg("rho") = 1e-6, bp::arg("over_relax") = 1.0),
             bp::return_value_policy<bp::return_by_value>())
        .def("getSolution", &NCPStagProjSolver<cb::PyramidCone>::getSolution,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("resetStats", &NCPStagProjSolver<cb::PyramidCone>::resetStats,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getCPUTimes", &NCPPGSSolver<cb::PyramidCone>::getCPUTimes,
             bp::args("self"))
        .def("vjp_fd", &NCPStagProjSolver<cb::PyramidCone>::vjp_fd,
             (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"), bp::arg("dL_dlam"),
              bp::arg("settings"), bp::arg("delta") = 1e-8))
        .def("getdLdmus", &NCPStagProjSolver<cb::PyramidCone>::getdLdmus,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getdLdDel", &NCPStagProjSolver<cb::PyramidCone>::getdLdDel,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getdLdg", &NCPStagProjSolver<cb::PyramidCone>::getdLdg,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>());
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<CCPPGSSolver>()) {
    bp::class_<CCPPGSSolver, boost::noncopyable>(
        "CCPPGSSolver", "A pgs solver solving the CCP problem.",
        boost::python::no_init)
        .def(bp::init<>(bp::args("self"), "Default constructor"))
        // Exposing attributes
        .DEF_R_CLASS_ATTRIB(CCPPGSSolver, n_iter_)
        .DEF_R_CLASS_ATTRIB(CCPPGSSolver, max_iter_)
        .DEF_R_CLASS_ATTRIB(CCPPGSSolver, th_stop_)
        .DEF_R_CLASS_ATTRIB(CCPPGSSolver, stop_)
        .DEF_R_CLASS_ATTRIB(CCPPGSSolver, rel_th_stop_)
        .DEF_R_CLASS_ATTRIB(CCPPGSSolver, rel_stop_)
        .DEF_R_CLASS_ATTRIB(CCPPGSSolver, prim_feas_)
        .DEF_R_CLASS_ATTRIB(CCPPGSSolver, dual_feas_)
        .DEF_R_CLASS_ATTRIB(CCPPGSSolver, dual_feas_reg_)
        .DEF_R_CLASS_ATTRIB(CCPPGSSolver, comp_)
        .DEF_R_CLASS_ATTRIB(CCPPGSSolver, comp_reg_)
        .DEF_R_CLASS_ATTRIB(CCPPGSSolver, sig_comp_)
        .DEF_R_CLASS_ATTRIB(CCPPGSSolver, sig_comp_reg_)
        .DEF_R_CLASS_ATTRIB(CCPPGSSolver, ncp_comp_)
        .DEF_R_CLASS_ATTRIB(CCPPGSSolver, ncp_comp_reg_)
        .DEF_R_CLASS_ATTRIB(CCPPGSSolver, stats_)
        // Exposing methods
        .def("setProblem", &CCPPGSSolver::setProblem,
             bp::args("self", "problem"),
             bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](CCPPGSSolver &m, ContactProblem<cb::IceCreamCone> &prob,
                const Ref<const VectorXs> &lam0,
                ContactSolverSettings &settings, T eps_reg, bool polish) {
              return m.solve(prob, lam0, settings, eps_reg, polish);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("settings"), bp::arg("eps_reg") = 0.,
             bp::arg("polish") = false),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](CCPPGSSolver &m, ContactProblem<cb::IceCreamCone> &prob,
                const Ref<const VectorXs> &lam0,
                ContactSolverSettings &settings,
                const Ref<const VectorXs> &R_reg, bool polish) {
              return m.solve(prob, lam0, settings, R_reg, polish);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("settings"), bp::arg("R_reg"), bp::arg("polish") = false),
            bp::return_value_policy<bp::return_by_value>())
        .def("getSolution", &CCPPGSSolver::getSolution, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getDualSolution", &CCPPGSSolver::getDualSolution,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("resetStats", &CCPPGSSolver::resetStats, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getCPUTimes", &CCPPGSSolver::getCPUTimes, bp::args("self"))
        .def("vjp_approx", &CCPPGSSolver::vjp_approx,
             (bp::arg("self"), bp::arg("problem"), bp::arg("dL_dlam"),
              bp::arg("settings"), bp::arg("eps_reg") = 0.))
        .def("vjp_fd", &CCPPGSSolver::vjp_fd,
             (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"), bp::arg("dL_dlam"),
              bp::arg("settings"), bp::arg("delta") = 1e-8))
        .def("getdLdmus", &CCPPGSSolver::getdLdmus, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getdLdDel", &CCPPGSSolver::getdLdDel, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getdLdg", &CCPPGSSolver::getdLdg, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getdlamdtheta", &CCPPGSSolver::getdlamdtheta, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
#ifdef DIFFCONTACT_WITH_CPPAD
        .def("vjp_cppad", &NCPPGSSolver<cb::IceCreamCone>::vjp_cppad,
             (bp::arg("self"), bp::arg("problem"), bp::arg("dL_dlam"),
              bp::arg("settings")))
        .def("jvp_cppad", &NCPPGSSolver<cb::IceCreamCone>::jvp_cppad,
             (bp::arg("self"), bp::arg("problem"), bp::arg("dG_dtheta"),
              bp::arg("dg_dtheta"), bp::arg("dmus_dtheta"), bp::arg("settings"),
              bp::arg("eps_reg") = 0.))
#endif
        ;
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<CCPADMMSolver>()) {
    bp::class_<CCPADMMSolver, boost::noncopyable>(
        "CCPADMMSolver", "An admm solver solving the CCP problem.",
        boost::python::no_init)
        .def(bp::init<>(bp::args("self"), "Default constructor"))
        // Exposing attributes
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, n_iter_)
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, max_iter_)
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, th_stop_)
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, stop_)
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, rel_th_stop_)
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, rel_stop_)
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, rho_)
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, eigval_max_)
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, eigval_min_)
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, prim_feas_)
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, dual_feas_)
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, dual_feas_reg_)
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, comp_)
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, comp_reg_)
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, ncp_comp_)
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, ncp_comp_reg_)
        .DEF_R_CLASS_ATTRIB(CCPADMMSolver, stats_)
        // Exposing methods
        .def("setProblem", &CCPADMMSolver::setProblem,
             bp::args("self", "problem"),
             bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](CCPADMMSolver &m, const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<const VectorXs> &lam0,
                ContactSolverSettings &settings, T rho, T over_relax,
                T eps_reg) {
              return m.solve(prob, lam0, settings, rho, over_relax, eps_reg);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("settings"), bp::arg("rho") = 1e-6,
             bp::arg("over_relax") = 1., bp::arg("eps_reg") = 0.),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](CCPADMMSolver &m, const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<const VectorXs> &lam0,
                const Ref<const VectorXs> &gamma0,
                ContactSolverSettings &settings, T rho, T over_relax,
                T eps_reg) {
              return m.solve(prob, lam0, gamma0, settings, rho, over_relax,
                             eps_reg);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("gamma0"), bp::arg("settings"), bp::arg("rho") = 1e-6,
             bp::arg("over_relax") = 1., bp::arg("eps_reg") = 0.),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](CCPADMMSolver &m, const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<const VectorXs> &lam0,
                const Ref<const VectorXs> &gamma0, T rho_admm,
                ContactSolverSettings &settings, T rho, T over_relax,
                T eps_reg) {
              return m.solve(prob, lam0, gamma0, rho_admm, settings, rho,
                             over_relax, eps_reg);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("gamma0"), bp::arg("rho_admm"), bp::arg("settings"),
             bp::arg("rho") = 1e-6, bp::arg("over_relax") = 1.,
             bp::arg("eps_reg") = 0.),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](CCPADMMSolver &m, const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<const VectorXs> &lam0,
                const Ref<const VectorXs> &gamma0, T rho_admm, T max_eigval,
                ContactSolverSettings &settings, T rho, T over_relax,
                T eps_reg) {
              return m.solve(prob, lam0, gamma0, rho_admm, max_eigval, settings,
                             rho, over_relax, eps_reg);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("gamma0"), bp::arg("rho_admm"), bp::arg("max_eigval"),
             bp::arg("settings"), bp::arg("rho") = 1e-6,
             bp::arg("over_relax") = 1., bp::arg("eps_reg") = 0.),
            bp::return_value_policy<bp::return_by_value>())
        .def("getSolution", &CCPADMMSolver::getSolution, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getDualSolution", &CCPADMMSolver::getDualSolution,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("resetStats", &CCPADMMSolver::resetStats, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getCPUTimes", &CCPADMMSolver::getCPUTimes, bp::args("self"))
        .def("vjp_fd", &CCPADMMSolver::vjp_fd,
             (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"), bp::arg("dL_dlam"),
              bp::arg("settings"), bp::arg("delta") = 1e-8))
        .def("getdLdmus", &CCPADMMSolver::getdLdmus, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getdLdDel", &CCPADMMSolver::getdLdDel, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getdLdg", &CCPADMMSolver::getdLdg, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getdlamdtheta", &CCPADMMSolver::getdlamdtheta, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>());
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<
          CCPADMMPrimalSolver>()) {
    bp::class_<CCPADMMPrimalSolver, boost::noncopyable>(
        "CCPADMMPrimalSolver", "An admm solver solving the primal CCP problem.",
        boost::python::no_init)
        .def(bp::init<>(bp::args("self"), "Default constructor"))
        // Exposing attributes
        .DEF_R_CLASS_ATTRIB(CCPADMMPrimalSolver, n_iter_)
        .DEF_R_CLASS_ATTRIB(CCPADMMPrimalSolver, max_iter_)
        .DEF_R_CLASS_ATTRIB(CCPADMMPrimalSolver, th_stop_)
        .DEF_R_CLASS_ATTRIB(CCPADMMPrimalSolver, stop_)
        .DEF_R_CLASS_ATTRIB(CCPADMMPrimalSolver, rel_th_stop_)
        .DEF_R_CLASS_ATTRIB(CCPADMMPrimalSolver, rel_stop_)
        .DEF_R_CLASS_ATTRIB(CCPADMMPrimalSolver, rho_)
        .DEF_R_CLASS_ATTRIB(CCPADMMPrimalSolver, eigval_max_)
        .DEF_R_CLASS_ATTRIB(CCPADMMPrimalSolver, eigval_min_)
        .DEF_R_CLASS_ATTRIB(CCPADMMPrimalSolver, prim_feas_)
        .DEF_R_CLASS_ATTRIB(CCPADMMPrimalSolver, dual_feas_)
        .DEF_R_CLASS_ATTRIB(CCPADMMPrimalSolver, dual_feas_reg_)
        .DEF_R_CLASS_ATTRIB(CCPADMMPrimalSolver, comp_)
        .DEF_R_CLASS_ATTRIB(CCPADMMPrimalSolver, comp_reg_)
        .DEF_R_CLASS_ATTRIB(CCPADMMPrimalSolver, stats_)
        // Exposing methods
        .def("setProblem", &CCPADMMPrimalSolver::setProblem,
             bp::args("self", "problem"),
             bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](CCPADMMPrimalSolver &m,
                const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<const VectorXs> &lam0,
                ContactSolverSettings &settings,
                const Ref<const VectorXs> &R_reg, T rho, T over_relax) {
              return m.solve(prob, lam0, settings, R_reg, rho, over_relax);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("settings"), bp::arg("R_reg"), bp::arg("rho") = 1e-6,
             bp::arg("over_relax") = 1.),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](CCPADMMPrimalSolver &m,
                const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<const VectorXs> &lam0,
                ContactSolverSettings &settings, T rho, T over_relax,
                T eps_reg) {
              return m.solve(prob, lam0, settings, rho, over_relax, eps_reg);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("dryyings"), bp::arg("rho") = 1e-6,
             bp::arg("over_relax") = 1., bp::arg("eps_reg") = 1e-6),
            bp::return_value_policy<bp::return_by_value>())
        .def("getSolution", &CCPADMMPrimalSolver::getSolution, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getDualSolution", &CCPADMMPrimalSolver::getDualSolution,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getCPUTimes", &CCPADMMPrimalSolver::getCPUTimes, bp::args("self"))
        .def("vjp_fd", &CCPADMMPrimalSolver::vjp_fd,
             (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"), bp::arg("dL_dlam"),
              bp::arg("settings"),
              bp::arg("delta") = 1e-8));
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<
          CCPNewtonPrimalSolver>()) {
    bp::class_<CCPNewtonPrimalSolver, boost::noncopyable>(
        "CCPNewtonPrimalSolver",
        "A Newton solver solving the primal CCP problem.",
        boost::python::no_init)
        .def(bp::init<>(bp::args("self"), "Default constructor"))
        // Exposing attributes
        .DEF_R_CLASS_ATTRIB(CCPNewtonPrimalSolver, n_iter_)
        .DEF_R_CLASS_ATTRIB(CCPNewtonPrimalSolver, max_iter_)
        .DEF_R_CLASS_ATTRIB(CCPNewtonPrimalSolver, th_stop_)
        .DEF_R_CLASS_ATTRIB(CCPNewtonPrimalSolver, stop_)
        .DEF_R_CLASS_ATTRIB(CCPNewtonPrimalSolver, sig_comp_)
        .DEF_R_CLASS_ATTRIB(CCPNewtonPrimalSolver, sig_comp_reg_)
        .DEF_R_CLASS_ATTRIB(CCPNewtonPrimalSolver, comp_)
        .DEF_R_CLASS_ATTRIB(CCPNewtonPrimalSolver, comp_reg_)
        .DEF_R_CLASS_ATTRIB(CCPNewtonPrimalSolver, ncp_comp_)
        .DEF_R_CLASS_ATTRIB(CCPNewtonPrimalSolver, ncp_comp_reg_)
        .DEF_R_CLASS_ATTRIB(CCPNewtonPrimalSolver, rel_th_stop_)
        .DEF_R_CLASS_ATTRIB(CCPNewtonPrimalSolver, rel_stop_)
        .DEF_R_CLASS_ATTRIB(CCPNewtonPrimalSolver, stats_)
        // // Exposing methods
        .def("setProblem", &CCPNewtonPrimalSolver::setProblem,
             bp::args("self", "problem"),
             bp::return_value_policy<bp::return_by_value>())
        .def(
            "setCompliance",
            +[](CCPNewtonPrimalSolver &self,
                const ContactProblem<cb::IceCreamCone> &prob,
                T eps_reg) { self.setCompliance(prob, eps_reg); },
            bp::args("self", "problem", "eps_reg"))
        .def(
            "setCompliance",
            +[](CCPNewtonPrimalSolver &self, const Ref<const VectorXs> &R_reg) {
              self.setCompliance(R_reg);
            },
            bp::args("self", "R_reg"))
        .def(
            "complianceMap",
            +[](CCPNewtonPrimalSolver &self,
                const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<VectorXs> R, const Ref<VectorXs> y,
                const Ref<VectorXs> lam_out) {
              self.complianceMap(prob, R, y, lam_out);
            },
            bp::args("self", "prob", "R", "y", "lam_out"))
        .def(
            "projKR",
            +[](CCPNewtonPrimalSolver &self,
                const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<VectorXs> R, const Ref<VectorXs> y,
                const Ref<VectorXs> projy_out) {
              self.projKR(prob, R, y, projy_out);
            },
            bp::args("self", "prob", "R", "y", "projy_out"))
        .def(
            "unconstrainedCost",
            +[](CCPNewtonPrimalSolver &self,
                const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<VectorXs> R, const Ref<VectorXs> dq) {
              self.unconstrainedCost(prob, R, dq);
            },
            bp::args("self", "prob", "R", "dq"))
        .def(
            "computeRegularizationGrad",
            +[](CCPNewtonPrimalSolver &self,
                const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<VectorXs> R, const Ref<VectorXs> dq,
                const Ref<VectorXs> grad) {
              self.computeRegularizationGrad(prob, R, dq, grad);
            },
            bp::args("self", "prob", "R", "dq", "grad"))
        .def(
            "computeGrad",
            +[](CCPNewtonPrimalSolver &self,
                const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<VectorXs> R, const Ref<VectorXs> dq,
                const Ref<VectorXs> grad) {
              self.computeGrad(prob, R, dq, grad);
            },
            bp::args("self", "prob", "R", "dq", "grad"))
        .def(
            "computeHessReg",
            +[](CCPNewtonPrimalSolver &self,
                const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<VectorXs> R, const Ref<VectorXs> y,
                const Ref<VectorXs> y_tilde, const Ref<MatrixXs> H) {
              self.computeHessReg(prob, R, y, y_tilde, H);
            },
            bp::args("self", "prob", "R", "y", "y_tilde", "H"))
        .def(
            "computeHess",
            +[](CCPNewtonPrimalSolver &self,
                const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<VectorXs> R, const Ref<VectorXs> dq,
                const Ref<MatrixXs> H) { self.computeHess(prob, R, dq, H); },
            bp::args("self", "prob", "R", "dq", "H"))
        .def(
            "solve",
            +[](CCPNewtonPrimalSolver &self,
                const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<const VectorXs> &lam0,
                ContactSolverSettings &settings,
                const Ref<const VectorXs> &R_reg) {
              return self.solve(prob, lam0, settings, R_reg);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("settings"), bp::arg("R_reg")),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](CCPNewtonPrimalSolver &self,
                const ContactProblem<cb::IceCreamCone> &prob,
                const Ref<const VectorXs> &lam0,
                ContactSolverSettings &settings, T eps_reg) {
              return self.solve(prob, lam0, settings, eps_reg);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("settings"), bp::arg("eps_reg") = 1e-6),
            bp::return_value_policy<bp::return_by_value>())
        .def("getCompliance", &CCPNewtonPrimalSolver::getCompliance,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getSolution", &CCPNewtonPrimalSolver::getSolution,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getDualSolution", &CCPNewtonPrimalSolver::getDualSolution,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getCPUTimes", &CCPNewtonPrimalSolver::getCPUTimes,
             bp::args("self"))
        .def("vjp_fd", &CCPNewtonPrimalSolver::vjp_fd,
             (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"), bp::arg("dL_dlam"),
              bp::arg("settings"),
              bp::arg("delta") = 1e-8));
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<RaisimSolver>()) {
    //     auto computeLamV0_wrapper = [](auto x) { return do_something(x); };
    bp::class_<RaisimSolver, boost::noncopyable>(
        "RaisimSolver", "Raisim's contact solver.", boost::python::no_init)
        .def(bp::init<>(bp::args("self"), "Default constructor"))
        // Exposing attributes
        .DEF_R_CLASS_ATTRIB(RaisimSolver, n_iter_)
        .DEF_R_CLASS_ATTRIB(RaisimSolver, max_iter_)
        .DEF_R_CLASS_ATTRIB(RaisimSolver, alpha_)
        .DEF_R_CLASS_ATTRIB(RaisimSolver, alpha_min_)
        .DEF_R_CLASS_ATTRIB(RaisimSolver, th_stop_)
        .DEF_R_CLASS_ATTRIB(RaisimSolver, stop_)
        .DEF_R_CLASS_ATTRIB(RaisimSolver, rel_th_stop_)
        .DEF_R_CLASS_ATTRIB(RaisimSolver, rel_stop_)
        .DEF_R_CLASS_ATTRIB(RaisimSolver, comp_)
        .DEF_R_CLASS_ATTRIB(RaisimSolver, stats_)

        // Exposing methods
        .def("computeGinv", &RaisimSolver::computeGinv, bp::args("self", "G"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getGinv", &RaisimSolver::getGinv, bp::args("self", "i"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeGlam", &RaisimSolver::computeGlam,
             bp::args("self", "G", "lam"),
             bp::return_value_policy<bp::return_by_value>())
        .def(
            "updateGlam",
            +[](RaisimSolver &self, int i, const MatrixXd &G,
                const Ref<const Vector3s> &lami) {
              self.updateGlam(i, G, lami);
            },
            bp::args("self", "i", "G", "lami"),
            bp::return_value_policy<bp::return_by_value>())
        .def("getGlam", &RaisimSolver::getGlam, bp::args("self", "i", "j"),
             bp::return_value_policy<bp::return_by_value>())
        .def("setGlam", &RaisimSolver::setGlam,
             bp::args("self", "i", "j", "Glamij"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeC", &RaisimSolver::computeC,
             bp::args("self", "G", "g", "lam"),
             bp::return_value_policy<bp::return_by_value>())
        .def(
            "updateC",
            +[](RaisimSolver &self, int i, const MatrixXd &G, const VectorXd &b,
                const Ref<const Vector3s> &lami) {
              self.updateC(i, G, b, lami);
            },
            bp::args("self", "i", "G", "g", "lami"),
            bp::return_value_policy<bp::return_by_value>())
        .def("getC", &RaisimSolver::getC, bp::args("self", "i"),
             bp::return_value_policy<bp::return_by_value>())
        .def("setProblem", &RaisimSolver::setProblem,
             bp::args("self", "problem"),
             bp::return_value_policy<bp::return_by_value>())
        .def(
            "computeLamV0",
            +[](RaisimSolver &self, const Ref<const Matrix3s> &Ginvi,
                const Ref<const Vector3s> &ci, Ref<Vector3s> lam_out) {
              self.computeLamV0(Ginvi, ci, lam_out);
            },
            bp::args("self", "Ginvi", "ci", "lam_out"))
        .def(
            "solve",
            +[](RaisimSolver &m, ContactProblem<cb::IceCreamCone> &prob,
                const Ref<const VectorXs> &lam0,
                ContactSolverSettings &settings,
                const Ref<const VectorXs> &R_reg, T alpha, T alpha_min, T beta1,
                T beta2, T beta3, T gamma, T th) {
              return m.solve(prob, lam0, settings, R_reg, alpha, alpha_min,
                             beta1, beta2, beta3, gamma, th);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("settings"), bp::arg("R_reg"), bp::arg("alpha") = 1.,
             bp::arg("alpha_min") = 0.0001, bp::arg("beta1") = 1e-2,
             bp::arg("beta2") = 0.5, bp::arg("beta3") = 1.3,
             bp::arg("gamma") = 0.9, bp::arg("th") = 1e-3),
            bp::return_value_policy<bp::return_by_value>())
        .def(
            "solve",
            +[](RaisimSolver &m, ContactProblem<cb::IceCreamCone> &prob,
                const Ref<const VectorXs> &lam0,
                ContactSolverSettings &settings, T eps_reg, T alpha,
                T alpha_min, T beta1, T beta2, T beta3, T gamma, T th) {
              return m.solve(prob, lam0, settings, eps_reg, alpha, alpha_min,
                             beta1, beta2, beta3, gamma, th);
            },
            (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
             bp::arg("settings"), bp::arg("eps_reg") = 0.,
             bp::arg("alpha") = 1., bp::arg("alpha_min") = 0.0001,
             bp::arg("beta1") = 1e-2, bp::arg("beta2") = 0.5,
             bp::arg("beta3") = 1.3, bp::arg("gamma") = 0.9,
             bp::arg("th") = 1e-3),
            bp::return_value_policy<bp::return_by_value>())
        .def("getSolution", &RaisimSolver::getSolution, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getDualSolution", &RaisimSolver::getDualSolution,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("resetStats", &RaisimSolver::resetStats, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getCPUTimes", &RaisimSolver::getCPUTimes, bp::args("self"))
        .def("vjp_fd", &RaisimSolver::vjp_fd,
             (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"), bp::arg("dL_dlam"),
              bp::arg("settings"), bp::arg("delta") = 1e-8))
        .def("getdLdmus", &RaisimSolver::getdLdmus, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getdLdDel", &RaisimSolver::getdLdDel, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getdLdg", &RaisimSolver::getdLdg, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>());
  }

  if (!eigenpy::register_symbolic_link_to_registered_type<
          RaisimCorrectedSolver>()) {
    bp::class_<RaisimCorrectedSolver, boost::noncopyable>(
        "RaisimCorrectedSolver", "Raisim's contact solver.",
        boost::python::no_init)
        .def(bp::init<>(bp::args("self"), "Default constructor"))
        // Exposing attributes
        .DEF_R_CLASS_ATTRIB(RaisimCorrectedSolver, n_iter_)
        .DEF_R_CLASS_ATTRIB(RaisimCorrectedSolver, max_iter_)
        .DEF_R_CLASS_ATTRIB(RaisimCorrectedSolver, alpha_)
        .DEF_R_CLASS_ATTRIB(RaisimCorrectedSolver, alpha_min_)
        .DEF_R_CLASS_ATTRIB(RaisimCorrectedSolver, th_stop_)
        .DEF_R_CLASS_ATTRIB(RaisimCorrectedSolver, stop_)
        .DEF_R_CLASS_ATTRIB(RaisimCorrectedSolver, rel_th_stop_)
        .DEF_R_CLASS_ATTRIB(RaisimCorrectedSolver, rel_stop_)
        .DEF_R_CLASS_ATTRIB(RaisimCorrectedSolver, comp_)
        .DEF_R_CLASS_ATTRIB(RaisimCorrectedSolver, stats_)

        // Exposing methods
        .def("computeGinv", &RaisimCorrectedSolver::computeGinv,
             bp::args("self", "G"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getGinv", &RaisimCorrectedSolver::getGinv, bp::args("self", "i"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeGlam", &RaisimCorrectedSolver::computeGlam,
             bp::args("self", "G", "lam"),
             bp::return_value_policy<bp::return_by_value>())
        .def(
            "updateGlam",
            +[](RaisimCorrectedSolver &self, int i, const MatrixXd &G,
                const Ref<const Vector3s> &lami) {
              self.updateGlam(i, G, lami);
            },
            bp::args("self", "i", "G", "lami"),
            bp::return_value_policy<bp::return_by_value>())
        .def("getGlam", &RaisimCorrectedSolver::getGlam,
             bp::args("self", "i", "j"),
             bp::return_value_policy<bp::return_by_value>())
        .def("setGlam", &RaisimCorrectedSolver::setGlam,
             bp::args("self", "i", "j", "Glamij"),
             bp::return_value_policy<bp::return_by_value>())
        .def("computeC", &RaisimCorrectedSolver::computeC,
             bp::args("self", "G", "g", "lam"),
             bp::return_value_policy<bp::return_by_value>())
        .def(
            "computeCorrectedC",
            +[](RaisimCorrectedSolver &self, const Ref<const Vector3s> &ci,
                const Ref<const Vector3s> &vi,
                double mui) { self.computeCorrectedC(ci, vi, mui); },
            bp::args("self", "ci", "mui"))
        .def(
            "updateC",
            +[](RaisimCorrectedSolver &self, int i, const MatrixXd &G,
                const VectorXd &b, const Ref<const Vector3s> &lami) {
              self.updateC(i, G, b, lami);
            },
            bp::args("self", "i", "G", "g", "lami"),
            bp::return_value_policy<bp::return_by_value>())
        .def("getC", &RaisimCorrectedSolver::getC, bp::args("self", "i"),
             bp::return_value_policy<bp::return_by_value>())
        .def("setProblem", &RaisimCorrectedSolver::setProblem,
             bp::args("self", "problem"),
             bp::return_value_policy<bp::return_by_value>())
        .def(
            "computeLamV0",
            +[](RaisimCorrectedSolver &self, const Ref<const Matrix3s> &Ginvi,
                const Ref<const Vector3s> &ci, Ref<Vector3s> lam_out) {
              self.computeLamV0(Ginvi, ci, lam_out);
            },
            bp::args("self", "Ginvi", "ci", "lam_out"))
        .def(
            "computeCorrectedLamV0",
            +[](RaisimCorrectedSolver &self, const Ref<const Matrix3s> &Ginvi,
                const Ref<const Vector3s> ci, const Ref<const Vector3s> &vi,
                const double mui, Ref<Vector3s> lam_out) {
              self.computeCorrectedLamV0(Ginvi, ci, vi, mui, lam_out);
            },
            bp::args("self", "Ginvi", "ci", "mui", "lam_out"))
        .def("solve", &RaisimCorrectedSolver::solve,
             (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"),
              bp::arg("settings"),
              //   bp::arg("max_iter"), bp::arg("th_stop") = 1e-6,
              //   bp::arg("rel_th_stop") = 1e-8,
              bp::arg("eps_reg") = 0.0, bp::arg("alpha") = 1.,
              bp::arg("alpha_min") = 0.0001, bp::arg("beta1") = 1e-2,
              bp::arg("beta2") = 0.5, bp::arg("beta3") = 1.3,
              bp::arg("gamma") = 0.9, bp::arg("th") = 1e-3),
             //   bp::arg("statistics") = false,
             //   bp::arg("timings") = false),
             bp::return_value_policy<bp::return_by_value>())
        .def("getSolution", &RaisimCorrectedSolver::getSolution,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("getDualSolution", &RaisimCorrectedSolver::getDualSolution,
             bp::args("self"), bp::return_value_policy<bp::return_by_value>())
        .def("resetStats", &RaisimCorrectedSolver::resetStats, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getCPUTimes", &RaisimCorrectedSolver::getCPUTimes,
             bp::args("self"))
        .def("vjp_fd", &RaisimCorrectedSolver::vjp_fd,
             (bp::arg("self"), bp::arg("problem"), bp::arg("lam0"), bp::arg("dL_dlam"),
              bp::arg("settings"), bp::arg("delta") = 1e-8))
        .def("getdLdmus", &RaisimCorrectedSolver::getdLdmus, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getdLdDel", &RaisimCorrectedSolver::getdLdDel, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>())
        .def("getdLdg", &RaisimCorrectedSolver::getdLdg, bp::args("self"),
             bp::return_value_policy<bp::return_by_value>());
  }
}

} // namespace context
} // namespace contactbench
