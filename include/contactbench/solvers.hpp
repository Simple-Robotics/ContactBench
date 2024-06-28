#ifndef CONTACT_BENCH_SOLVERS_H
#define CONTACT_BENCH_SOLVERS_H

// #include <contact-bench/workspace.hpp>
#include "contactbench/contact-problem.hpp"
#include "contactbench/friction-constraint.hpp"
#include <hpp/fcl/timings.h>
#include <pinocchio/math/eigenvalues.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <limits>
#include <cmath>
#include <optional>

#ifdef DIFFCONTACT_WITH_CPPAD
#include <pinocchio/autodiff/cppad.hpp>
#endif

namespace contactbench {

template <typename T> struct Statistics {

public:
  std::vector<T> stop_;
  std::vector<T> rel_stop_;
  std::vector<T> comp_;
  std::vector<T> prim_feas_;
  std::vector<T> dual_feas_;
  std::vector<T> sig_comp_;
  std::vector<T> ncp_comp_;
  std::vector<T> cost_;

  Statistics(){};

  void addStop(T stop);
  void addRelStop(T rel_stop);
  void addComp(T comp);
  void addPrimFeas(T prim_feas);
  void addDualFeas(T dual_feas);
  void addSigComp(T sig_comp);
  void addNcpComp(T ncp_comp);
  void addCost(T cost);
  void reset();
};

template <typename T> struct ContactSolverSettings {

public:
  T stop_;
  T rel_stop_;
  T th_stop_ = 1e-6;
  T rel_th_stop_ = 1e-6;
  int max_iter_ = 100;
  bool timings_ = false;
  bool statistics_ = false;

  ContactSolverSettings(){};

  template <typename NewScalar> ContactSolverSettings<NewScalar> cast() const {
    ContactSolverSettings<NewScalar> newSettings;
    newSettings.stop_ = static_cast<NewScalar>(stop_);
    newSettings.rel_stop_ = static_cast<NewScalar>(rel_stop_);
    newSettings.th_stop_ = static_cast<NewScalar>(th_stop_);
    newSettings.rel_th_stop_ = static_cast<NewScalar>(rel_th_stop_);
    newSettings.max_iter_ = max_iter_;
    newSettings.timings_ = timings_;
    newSettings.statistics_ = statistics_;
    return newSettings;
  }
};

template <typename T, template <typename> class ConstraintTpl>
struct BaseSolver {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);
  using ConstraintType = ConstraintTpl<T>;

public:
  int nc_;

  BaseSolver() = default;
  virtual ~BaseSolver() = default;

  void setProblem(const ContactProblem<T, ConstraintTpl> &prob);

  T stoppingCriteria(const ContactProblem<T, ConstraintTpl> &prob,
                     const Ref<const VectorXs> &lam,
                     const Ref<const VectorXs> &v);

  T relativeStoppingCriteria(const Ref<const VectorXs> &lam,
                             const Ref<const VectorXs> &lam_pred);

  bool
  solve(CONTACTBENCH_MAYBE_UNUSED const ContactProblem<T, ConstraintTpl> &prob,
        CONTACTBENCH_MAYBE_UNUSED const Ref<const VectorXs> &lam0,
        CONTACTBENCH_MAYBE_UNUSED ContactSolverSettings<T> &settings) {
    CONTACTBENCH_RUNTIME_ERROR("Unimplemented.");
  }

  typedef hpp::fcl::CPUTimes CPUTimes;
  typedef hpp::fcl::Timer Timer;

  CPUTimes getCPUTimes() const { return timer_.elapsed(); }

protected:
  Timer timer_;
  VectorXs R_reg_;
};

template <typename T, template <typename> class ConstraintTpl>
struct DualBaseSolver : BaseSolver<T, ConstraintTpl> {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);
  using ConstraintType = ConstraintTpl<T>;

public:
  typedef BaseSolver<T, ConstraintTpl> Base;
  using Base::nc_;
  DualBaseSolver() = default;
  virtual ~DualBaseSolver() = default;

  const Ref<const VectorXs> getSolution() const;

  DualBaseSolver &operator=(DualBaseSolver &&) = default;

  // Derivatives

  void setJvpProblem(const ContactProblem<T, ConstraintTpl> &prob,
                     const Ref<const MatrixXs> &dg_dtheta);

  // TODO vjp_fd
  void jvp_fd(ContactProblem<T, ConstraintTpl> &prob,
              const Ref<const VectorXs> &lam0,
              const Ref<const MatrixXs> &dG_dtheta,
              const Ref<const MatrixXs> &dg_dtheta,
              const Ref<const MatrixXs> &dmu_dtheta,
              ContactSolverSettings<T> &settings, const T delta = 1e-8);

  const Ref<const MatrixXs> getdlamdtheta() const;

  void setVjpProblem(const ContactProblem<T, ConstraintTpl> &prob);

  void vjp_fd(ContactProblem<T, ConstraintTpl> &prob,
              const Ref<const VectorXs> &lam0,
              const Ref<const VectorXs> &dL_dlam,
              ContactSolverSettings<T> &settings, const T delta = 1e-8);

  const Ref<const VectorXs> getdLdmus() const;
  const Ref<const MatrixXs> getdLdDel() const;
  const Ref<const VectorXs> getdLdg() const;

  // CPPAD derivatives

#ifdef DIFFCONTACT_WITH_CPPAD
  void jvp_cppad(ContactProblem<T, ConstraintTpl> &prob,
                 const Ref<const MatrixXs> &dG_dtheta,
                 const Ref<const MatrixXs> &dg_dtheta,
                 const Ref<const MatrixXs> &dmu_dtheta,
                 ContactSolverSettings<T> &settings, const T eps_reg = 0.);
  void vjp_cppad(ContactProblem<T, ConstraintTpl> &prob,
                 const Ref<const VectorXs> &dL_dlam,
                 ContactSolverSettings<T> &settings);
#endif

protected:
  virtual bool _solve(ContactProblem<T, ConstraintTpl> &prob,
                      const Ref<const VectorXs> &lam0,
                      ContactSolverSettings<T> &settings) = 0;
  VectorXs lam_, lam_fd_;
  using Base::R_reg_;
  // Derivatives
  int ntheta_;
  VectorXs dL_dmus_, dL_dg_;
  MatrixXs dL_dDel_, dlam_dtheta_;
  using Base::timer_;
};

template <typename T, template <typename> class ConstraintTpl>
struct NCPPGSSolver : public DualBaseSolver<T, ConstraintTpl> {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);

public:
  typedef DualBaseSolver<T, ConstraintTpl> Base;
  using Base::nc_;
  int max_iter_, n_iter_;
  T step_eps_;
  T th_stop_, stop_, rel_th_stop_, rel_stop_, ncp_comp_, dual_feas_, sig_comp_;

  NCPPGSSolver();

  typedef hpp::fcl::CPUTimes CPUTimes;
  typedef hpp::fcl::Timer Timer;

  // Timer timer;

  // CPUTimes getCPUTimes() const { return timer.elapsed(); }

  void setProblem(const ContactProblem<T, ConstraintTpl> &prob);

  T stoppingCriteria(const ContactProblem<T, ConstraintTpl> &prob,
                     const Ref<const VectorXs> &lam,
                     const Ref<const VectorXs> &v);

  T relativeStoppingCriteria(const Ref<const VectorXs> &lam,
                             const Ref<const VectorXs> &lam_pred);

  void addStatistics(T stop, T rel_stop, T ncp_comp, T sig_comp, T dual_feas);

  bool solve(const ContactProblem<T, ConstraintTpl> &prob,
             const Ref<const VectorXs> &lam0,
             ContactSolverSettings<T> &settings, T over_relax = 1.,
             T eps_reg = 0.);

  bool solve(const ContactProblem<T, ConstraintTpl> &prob,
             const Ref<const VectorXs> &lam0,
             ContactSolverSettings<T> &settings,
             const Ref<const VectorXs> &R_reg, T over_relax = 1.);

  void setLCP(const ContactProblem<T, ConstraintTpl> &prob);

  T computeSmallestEigenValue(const Ref<const MatrixXs> &H,
                              const double epsilon = 1e-10,
                              const int max_iter = 10);

  bool _polish(ContactProblem<T, ConstraintTpl> &prob,
               const Ref<const VectorXs> &lam,
               ContactSolverSettings<T> &settings, T rho = 1e-6,
               T eps_reg = 0.);

  using Base::getSolution;

  const Ref<const VectorXs> getDualSolution() const;

  // Derivatives

  // JVP

  using Base::setJvpProblem;

  using Base::getdlamdtheta;

  // VJP

  using Base::setVjpProblem;

  void setApproxVjpProblem(const ContactProblem<T, ConstraintTpl> &prob);

  void vjp_approx(ContactProblem<T, ConstraintTpl> &prob,
                  const Ref<const VectorXs> &dL_dlam,
                  ContactSolverSettings<T> &settings, T rho = 1e-6,
                  const T eps_reg = 0.);

  using Base::getdLdDel, Base::getdLdg, Base::getdLdmus;

  void resetStats();

  // Statistics
  Statistics<T> stats_;

protected:
  using Base::lam_, Base::R_reg_;
  VectorXs lam_pred_, dx_;
  bool lam_is_polished_ = false;
  VectorXs v_, v_cor_, v_proj_, v_reg_;
  proxsuite::proxqp::dense::isize dim_, neq_, nin_;
  T rho, l_min_, prim_feas_;
  std::unique_ptr<proxsuite::proxqp::dense::QP<T>> qp_;
  MatrixXs H_, C_;
  VectorXs l_, x_lcp_, y_lcp_, z_lcp_;
  VectorXs vpow_, Hvpow_, err_vpow_;
  using Base::ntheta_, Base::dL_dDel_, Base::dL_dg_, Base::dL_dmus_,
      Base::dlam_dtheta_;
  VectorXs dL_db_, dL_dxyz_;
  MatrixXs dL_dA_;

  bool _solve(ContactProblem<T, ConstraintTpl> &prob,
              const Ref<const VectorXs> &lam0,
              ContactSolverSettings<T> &settings);
  using Base::timer_;
};

template <typename T>
struct LCPBaseSolver : public DualBaseSolver<T, PyramidCone> {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);

public:
  typedef DualBaseSolver<T, PyramidCone> Base;
  using Base::nc_;
  int max_iter_, n_iter_;
  T th_stop_, rel_th_stop_, stop_, rel_stop_, l_min_, l_max_, comp_, prim_feas_,
      dual_feas_;

  LCPBaseSolver() = default;
  virtual ~LCPBaseSolver() = default;

  using Base::getSolution;

  // Statistics
  Statistics<T> stats_;

  T computeLargestEigenValue(const Ref<const MatrixXs> &H,
                             const T epsilon = 1e-10, const int max_iter = 10);

  T computeSmallestEigenValue(const Ref<const MatrixXs> &H,
                              const T epsilon = 1e-10, const int max_iter = 10);

  bool _solve_qp(ContactProblem<T, PyramidCone> &prob,
                 const Ref<const VectorXs> &lam0,
                 ContactSolverSettings<T> &settings,
                 const Ref<const VectorXs> &R_reg, T rho = 1e-6);

  bool _solve_qp(ContactProblem<T, PyramidCone> &prob,
                 const Ref<const VectorXs> &lam0,
                 ContactSolverSettings<T> &settings, T rho = 1e-6,
                 T eps_reg = 0.);

  const MatrixXs getQPH() const;
  const VectorXs getQPg() const;
  const MatrixXs getQPC() const;
  const VectorXs getQPl() const;

  // JVP

  // TODO

  using Base::getdlamdtheta;

  // VJP

  void setVjpProblem(const ContactProblem<T, PyramidCone> &prob);

  void vjp(ContactProblem<T, PyramidCone> &prob,
           const Ref<const VectorXs> &dL_dlam,
           ContactSolverSettings<T> &settings);

  using Base::getdLdDel, Base::getdLdg, Base::getdLdmus;

#ifdef DIFFCONTACT_WITH_CPPAD
  using Base::jvp_cppad, Base::vjp_cppad;
#endif

  LCPBaseSolver &operator=(LCPBaseSolver &&) = default;

protected:
  bool lam_is_polished_ = false;
  VectorXs vpow_, Hvpow_, err_vpow_;
  using Base::lam_;
  using Base::ntheta_;
  VectorXs v_reg_, R_reg_;
  std::unique_ptr<proxsuite::proxqp::dense::QP<T>> qp_;
  proxsuite::proxqp::dense::isize dim_, neq_, nin_;
  MatrixXs C_, H_;
  VectorXs l_, x_lcp_, y_lcp_, z_lcp_, dL_dxyz_, dL_dlam_lcp_;
  using Base::dL_dDel_, Base::dL_dg_, Base::dL_dmus_, Base::dlam_dtheta_;
  MatrixXs dL_dDel_t_, dL_dC_;
  MatrixXs dlam_;
  VectorXs dL_db_;
  MatrixXs dL_dA_, db_dtheta_, dA_dtheta_lam_;
  using Base::timer_;
};

template <typename T> struct LCPQPSolver : public LCPBaseSolver<T> {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);

public:
  typedef LCPBaseSolver<T> Base;
  using Base::max_iter_, Base::n_iter_;
  using Base::nc_;
  using Base::th_stop_, Base::rel_th_stop_, Base::stop_, Base::rel_stop_,
      Base::comp_, Base::prim_feas_, Base::dual_feas_, Base::l_min_,
      Base::l_max_;

  LCPQPSolver();

  void setProblem(ContactProblem<T, PyramidCone> &prob);

  using Base::_solve_qp;

  bool solve(ContactProblem<T, PyramidCone> &prob,
             const Ref<const VectorXs> &lam0,
             ContactSolverSettings<T> &settings, T rho = 1e-6, T eps_reg = 0.);

  bool solve(ContactProblem<T, PyramidCone> &prob,
             const Ref<const VectorXs> &lam0,
             ContactSolverSettings<T> &settings,
             const Ref<const VectorXs> &R_reg, T rho = 1e-6);

  using Base::getSolution;

  const Ref<const VectorXs> getDualSolution() const;

  void resetStats();

  // Statistics
  using Base::stats_;

protected:
  using Base::C_, Base::H_;
  using Base::dim_, Base::neq_, Base::nin_;
  using Base::l_, Base::x_lcp_, Base::y_lcp_, Base::z_lcp_, Base::dL_dxyz_;
  using Base::lam_;
  using Base::lam_is_polished_;
  using Base::qp_;
  VectorXs v_;
  using Base::dL_db_, Base::dL_dA_;
  using Base::dL_dDel_, Base::dL_dg_, Base::dL_dmus_;
  using Base::vpow_, Base::Hvpow_, Base::err_vpow_;

  bool _solve(ContactProblem<T, PyramidCone> &prob,
              const Ref<const VectorXs> &lam0,
              ContactSolverSettings<T> &settings);
  using Base::timer_;
};

template <template <typename> class ConstraintTpl>
struct NCPStagProjSolver : public DualBaseSolver<double, ConstraintTpl> {

public:
  typedef DualBaseSolver<double, ConstraintTpl> Base;
  using Base::nc_;
  int max_iter_, n_iter_;
  double th_stop_, stop_, rel_th_stop_, rel_stop_, comp_, dual_feas_;
  double stop_n_, rel_stop_n_, stop_t_, rel_stop_t_, comp_n_;
  double prim_feas_n_, dual_feas_n_, prim_feas_t_, dual_feas_t_;
  double rho_n_, rho_t_, eigval_max_n_, eigval_max_t_, eigval_min_;

  NCPStagProjSolver();

  void setProblem(const ContactProblem<double, ConstraintTpl> &prob);

  void
  evaluateNormalDelassus(const ContactProblem<double, ConstraintTpl> &prob);

  void
  evaluateTangentDelassus(const ContactProblem<double, ConstraintTpl> &prob);

  double stoppingCriteria(const ContactProblem<double, ConstraintTpl> &prob,
                          const Eigen::VectorXd &lam, const Eigen::VectorXd &v);

  double relativeStoppingCriteria(const Eigen::VectorXd &lam,
                                  const Eigen::VectorXd &lam_pred);

  void addStatistics(double stop, double rel_stop, double comp, double sig_comp,
                     double dual_feas);

  double computeLargestEigenValueNormal(const MatrixXd &Gn,
                                        const double epsilon = 1e-10,
                                        const int max_iter = 10);

  double computeLargestEigenValueTangent(const MatrixXd &Gt,
                                         const double epsilon = 1e-10,
                                         const int max_iter = 10);

  void computeGntild(const MatrixXd &Gn, double rho);

  void computeGttild(const MatrixXd &Gt, double rho);

  void computeGninv(const MatrixXd &Gn);

  void computeGtinv(const MatrixXd &Gt);

  template <typename VecIn, typename VecIn2, typename VecIn3, typename VecIn4>
  double stoppingCriteriaNormal(const Eigen::MatrixBase<VecIn> &lam1,
                                const Eigen::MatrixBase<VecIn2> &lam2,
                                const Eigen::MatrixBase<VecIn3> &gamma,
                                const Eigen::MatrixBase<VecIn4> &v);

  template <typename VecIn, typename VecIn2>
  double
  relativeStoppingCriteriaNormal(const Eigen::MatrixBase<VecIn> &lam,
                                 const Eigen::MatrixBase<VecIn2> &lam_pred);

  template <typename VecIn, typename VecIn2, typename VecIn3, typename VecIn4>
  double stoppingCriteriaTangent(const Eigen::MatrixBase<VecIn> &lam1,
                                 const Eigen::MatrixBase<VecIn2> &lam2,
                                 const Eigen::MatrixBase<VecIn3> &gamma,
                                 const Eigen::MatrixBase<VecIn4> &v);

  template <typename VecIn, typename VecIn2>
  double
  relativeStoppingCriteriaTangent(const Eigen::MatrixBase<VecIn> &lam,
                                  const Eigen::MatrixBase<VecIn2> &lam_pred);

  bool updateRhoNormal(double res_prim, double res_dual);

  bool updateRhoTangent(double res_prim, double res_dual);

  template <typename MatIn, typename VecIn, typename VecIn2, typename VecOut>
  void solveNormal(const Eigen::MatrixBase<MatIn> &Gn,
                   const Eigen::MatrixBase<VecIn> &gn,
                   Eigen::LLT<Ref<MatrixXd>> &llt_n,
                   const Eigen::MatrixBase<VecIn2> &x0,
                   const Eigen::MatrixBase<VecOut> &lam_out, int maxIter,
                   double th_stop = 1e-6, double rel_th_stop = 1e-6,
                   double rho = 1e-6, double over_relax = 1.);

  template <typename MatIn, typename VecIn, typename VecIn2, typename VecIn3,
            typename VecOut>
  void solveTangent(const Eigen::MatrixBase<MatIn> &Gt,
                    const Eigen::MatrixBase<VecIn> &gt,
                    Eigen::LLT<Ref<Eigen::MatrixXd>> &llt_t,
                    const ContactProblem<double, ConstraintTpl> &prob,
                    const Eigen::MatrixBase<VecIn2> &lam_n,
                    const Eigen::MatrixBase<VecIn3> &x0,
                    const Eigen::MatrixBase<VecOut> &lam_out, int maxIter,
                    double th_stop = 1e-6, double rel_th_stop = 1e-6,
                    double rho = 1e-6, double over_relax = 1.);

  bool solve(const ContactProblem<double, ConstraintTpl> &prob,
             const VectorXd &lam0, ContactSolverSettings<double> &settings,
             int maxInnerIter = 100, double rho = 1e-6, double over_relax = 1.);

  using Base::getSolution;
  const VectorXd &getDualNormal() const;
  const VectorXd &getDualTangent() const;

  void resetStats();

  // Statistics
  Statistics<double> stats_;

protected:
  using Base::lam_;
  Eigen::VectorXd lam_pred_, dlam_;
  Eigen::VectorXd v_, v_n_, v_t_, v_cor_, v_proj_;
  std::vector<int> ind_n_, ind_t_;
  Eigen::VectorXd lam_n1_, lam_t1_, lam_n1_pred_, lam_n2_pred_, lam_t1_pred_,
      lam_t2_pred_, dlam_n_, dlam_t_;
  Eigen::VectorXd gamma_n_, gamma_t_;
  Eigen::MatrixXd G_n_, G_t_, G_nt_, G_tn_, G_n_tild_, G_t_tild_;
  Eigen::VectorXd g_n_, g_t_;
  Eigen::MatrixXd G_n_inv_, G_t_inv_, G_n_llt_, G_t_llt_;
  Eigen::VectorXd Gnvnpow_, Gtvtpow_, vnpow_, vtpow_, err_vnpow_, err_vtpow_;

  bool _solve(ContactProblem<double, ConstraintTpl> &prob,
              const Ref<const VectorXd> &lam0,
              ContactSolverSettings<double> &settings);
  using Base::timer_;
};

template <typename T>
struct CCPBaseSolver : public DualBaseSolver<T, IceCreamCone> {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);

public:
  typedef DualBaseSolver<T, IceCreamCone> Base;
  using Base::nc_;
  int max_iter_, n_iter_;

  CCPBaseSolver() = default;
  virtual ~CCPBaseSolver() = default;

  using Base::getSolution;

  void setLCCP(const ContactProblem<T, IceCreamCone> &prob);

  bool _polish(ContactProblem<T, IceCreamCone> &prob,
               const Ref<const VectorXs> &lam,
               ContactSolverSettings<T> &settings, T eps_reg = 0.);

  bool _polish(ContactProblem<T, IceCreamCone> &prob,
               const Ref<const VectorXs> &lam,
               ContactSolverSettings<T> &settings,
               const Ref<const VectorXs> &R_reg);

  using Base::getdlamdtheta;

  void setApproxVjpProblem(const ContactProblem<T, IceCreamCone> &prob);

  void vjp_approx(ContactProblem<T, IceCreamCone> &prob,
                  const Ref<const VectorXs> &dL_dlam,
                  ContactSolverSettings<T> &settings, const T eps_reg = 0.);

  using Base::getdLdDel, Base::getdLdg, Base::getdLdmus;

#ifdef DIFFCONTACT_WITH_CPPAD
  using Base::jvp_cppad, Base::vjp_cppad;
#endif

  CCPBaseSolver &operator=(CCPBaseSolver &&) = default;

protected:
  bool lam_is_polished_ = false;
  using Base::lam_;
  using Base::ntheta_;
  VectorXs v_reg_, R_reg_;
  std::unique_ptr<proxsuite::proxqp::dense::QP<T>> qp_;
  proxsuite::proxqp::dense::isize dim_, nin_, neq_;
  MatrixXs C_;
  VectorXs l_, y_qp_, z_qp_, u_in_, u_eq_;
  using Base::dL_dDel_, Base::dL_dg_, Base::dL_dmus_, Base::dlam_dtheta_;
  VectorXs dL_dxyz_;
  MatrixXs dL_dDel_t_, dL_dC_;
  MatrixXs dlam_;
  using Base::timer_;
};

template <typename T> struct CCPPGSSolver : public CCPBaseSolver<T> {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);

public:
  typedef CCPBaseSolver<T> Base;
  using Base::nc_, Base::max_iter_, Base::n_iter_;
  T prim_feas_, dual_feas_, dual_feas_reg_, comp_, comp_reg_, ncp_comp_,
      ncp_comp_reg_, sig_comp_, sig_comp_reg_;
  T th_stop_, stop_, rel_th_stop_, rel_stop_;
  T step_eps_;

  // typedef hpp::fcl::CPUTimes CPUTimes;
  // typedef hpp::fcl::Timer Timer;

  // Timer timer;

  // CPUTimes getCPUTimes() const { return timer.elapsed(); }

  CCPPGSSolver() {
    timer_.stop();
    step_eps_ = 1e-12;
  }

  void setProblem(const ContactProblem<T, IceCreamCone> &prob);

  T relativeStoppingCriteria(const Ref<const VectorXs> &lam,
                             const Ref<const VectorXs> &lam_pred);

  void computeStatistics(const ContactProblem<T, IceCreamCone> &prob,
                         const Ref<const VectorXs> &lam,
                         const Ref<const VectorXs> &v,
                         const Ref<const VectorXs> &v_reg);

  T stoppingCriteria(const ContactProblem<T, IceCreamCone> &prob,
                     const Ref<const VectorXs> &lam,
                     const Ref<const VectorXs> &v_reg);

  void addStatistics(T stop, T rel_stop, T comp, T prim_feas, T dual_feas,
                     T sig_comp, T ncp_comp);

  bool solve(ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0,
             ContactSolverSettings<T> &settings, T eps_reg = 0.,
             bool polish = false);

  bool solve(ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0,
             ContactSolverSettings<T> &settings,
             const Ref<const VectorXs> &R_reg, bool polish = false);

  using Base::getSolution;

  const Ref<const VectorXs> getDualSolution() const;

  void resetStats();

  // Statistics
  Statistics<T> stats_;

protected:
  VectorXs dx_, lam_pred_;
  VectorXs v_, v_proj_;
  using Base::lam_, Base::v_reg_, Base::ntheta_, Base::qp_, Base::C_, Base::l_,
      Base::y_qp_, Base::z_qp_, Base::u_in_, Base::u_eq_, Base::dL_dmus_,
      Base::dL_dg_, Base::dL_dxyz_, Base::dL_dDel_, Base::dL_dC_,
      Base::dlam_dtheta_, Base::dlam_, Base::lam_is_polished_, Base::R_reg_;

  bool _solve(ContactProblem<T, IceCreamCone> &prob,
              const Ref<const VectorXs> &lam0,
              ContactSolverSettings<T> &settings);
  using Base::timer_;
};

template <typename T> struct CCPADMMSolver : CCPBaseSolver<T> {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);

public:

  typedef contactbench::PowerIterationWrapper<VectorXs> PowerIterationAlgo;
  typedef CCPBaseSolver<T> Base;
  using Base::nc_, Base::max_iter_, Base::n_iter_;
  T rho_, eigval_max_, eigval_min_;
  T prim_feas_, dual_feas_, dual_feas_reg_, dual_feas_reg_approx_, comp_,
      comp_reg_, comp_reg_approx_, sig_comp_, sig_comp_reg_, ncp_comp_,
      ncp_comp_reg_;
  T th_stop_, stop_, rel_th_stop_, rel_stop_;

  CCPADMMSolver();

  template <typename MatIn>
  T computeLargestEigenValue(const Eigen::MatrixBase<MatIn> &G,
                             const T epsilon = 1e-10, const int max_iter = 10);

  void setProblem(const ContactProblem<T, IceCreamCone> &prob);

  bool updateRho(T res_prim, T res_dual, T &rho_out);

  void computeStatistics(const ContactProblem<T, IceCreamCone> &prob,
                         const Ref<const VectorXs> &lam1,
                         const Ref<const VectorXs> &lam2,
                         const Ref<const VectorXs> &gamma,
                         const Ref<const VectorXs> &v,
                         const Ref<const VectorXs> &v_reg);

  T stoppingCriteria(const ContactProblem<T, IceCreamCone> &prob,
                     const Ref<const VectorXs> &lam1,
                     const Ref<const VectorXs> &lam1_pred,
                     const Ref<const VectorXs> &lam2,
                     const Ref<const VectorXs> &lam2_pred,
                     const Ref<const VectorXs> &gamma, const T rho_,
                     const T rho);

  T relativeStoppingCriteria(const Ref<const VectorXs> &lam,
                             const Ref<const VectorXs> &lam_pred);

  void addStatistics(T stop, T rel_stop, T comp, T prim_feas, T dual_feas,
                     T sig_comp, T ncp_comp);

  void computeChol(const ContactProblem<T, IceCreamCone> &prob, T rho = 1e-6,
                   bool statistics = false);

  void evaluateEigVals(const ContactProblem<T, IceCreamCone> &prob,
                       T eps_reg = 0., T rho = 1e-6);

  void evaluateEigVals(const ContactProblem<T, IceCreamCone> &prob,
                       const Ref<const VectorXs> &R_reg, T rho = 1e-6);

  void evaluateEigValMin(const ContactProblem<T, IceCreamCone> &prob,
                         T eps_reg = 0., T rho = 1e-6);

  void evaluateEigValMin(const ContactProblem<T, IceCreamCone> &prob,
                         const Ref<const VectorXs> &R_reg, T rho = 1e-6);

  bool solve(const ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0, const Ref<const VectorXs> &gamma0,
             ContactSolverSettings<T> &settings, T rho = 1e-6,
             T over_relax = 1., T eps_reg = 0.);

  bool solve(const ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0, const Ref<const VectorXs> &gamma0,
             T rho_admm, ContactSolverSettings<T> &settings, T rho = 1e-6,
             T over_relax = 1., T eps_reg = 0.);

  bool solve(const ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0, const Ref<const VectorXs> &gamma0,
             T rho_admm, T max_eigval, ContactSolverSettings<T> &settings,
             T rho = 1e-6, T over_relax = 1., T eps_reg = 0.);

  bool solve(const ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0,
             ContactSolverSettings<T> &settings, T rho = 1e-6,
             T over_relax = 1., T eps_reg = 0.);

  bool solve(const ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0, const Ref<const VectorXs> &gamma0,
             ContactSolverSettings<T> &settings,
             const Ref<const VectorXs> &R_reg, T rho = 1e-6, T over_relax = 1.);

  bool solve(const ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0, const Ref<const VectorXs> &gamma0,
             T rho_admm, ContactSolverSettings<T> &settings,
             const Ref<const VectorXs> &R_reg, T rho = 1e-6, T over_relax = 1.);

  bool solve(const ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0, const Ref<const VectorXs> &gamma0,
             T rho_admm, T max_eigval, ContactSolverSettings<T> &settings,
             const Ref<const VectorXs> &R_reg, T rho = 1e-6, T over_relax = 1.);

  bool solve(const ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0,
             ContactSolverSettings<T> &settings,
             const Ref<const VectorXs> &R_reg, T rho = 1e-6, T over_relax = 1.);

  using Base::getSolution;

  const Ref<const VectorXs> getDualSolution() const;

  void resetStats();

  // Statistics
  Statistics<T> stats_;

protected:
  PowerIterationAlgo power_iteration_algo_;
  MatrixXs Ginv_, Gtild_;
  MatrixXs G_llt_;
  VectorXs vpow_, Gvpow_, err_vpow_;
  VectorXs R_reg_, rhos_;
  VectorXs lam_pred_, lam2_, lam2_pred_, lam_or_, gamma_, deltalam_, v_,
      v_proj_;
  // quantities used for gradients computation
  using Base::lam_, Base::v_reg_, Base::ntheta_, Base::qp_, Base::C_, Base::l_,
      Base::y_qp_, Base::z_qp_, Base::u_in_, Base::u_eq_, Base::dL_dmus_,
      Base::dL_dg_, Base::dL_dxyz_, Base::dL_dDel_, Base::dL_dC_,
      Base::dlam_dtheta_, Base::dlam_, Base::lam_is_polished_;

  bool _solve_impl(const ContactProblem<T, IceCreamCone> &prob,
                   const Ref<const VectorXs> &lam0,
                   const Ref<const VectorXs> &gamma0,
                   ContactSolverSettings<T> &settings,
                   const Ref<const VectorXs> &R_reg, T rho_admm = 1e-6,
                   T rho = 1e-6, T over_relax = 1.);

  bool _solve_impl(const ContactProblem<T, IceCreamCone> &prob,
                   const Ref<const VectorXs> &lam0,
                   const Ref<const VectorXs> &gamma0,
                   ContactSolverSettings<T> &settings, T rho_admm = 1e-6,
                   T rho = 1e-6, T over_relax = 1., T eps_reg = 0.);

  bool _solve(ContactProblem<T, IceCreamCone> &prob,
              const Ref<const VectorXs> &lam0,
              ContactSolverSettings<T> &settings);
  using Base::timer_;
};

template <typename T> struct CCPADMMPrimalSolver : CCPADMMSolver<T> {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);

public:
  typedef CCPBaseSolver<T> Base;
  using Base::nc_;
  int nv_;
  int max_iter_, n_iter_;
  T rho_, eigval_max_, eigval_min_;
  T prim_feas_, dual_feas_, dual_feas_reg_, dual_feas_reg_approx_, comp_,
      comp_reg_, comp_reg_approx_, ncp_comp_, ncp_comp_reg_, sig_comp_,
      sig_comp_reg_;
  T th_stop_, stop_, rel_th_stop_, rel_stop_;

  CCPADMMPrimalSolver();

  void setProblem(const ContactProblem<T, IceCreamCone> &prob);

  bool updateRho(T res_prim, T res_dual, T &rho_out);

  void addStatistics(T stop, T rel_stop, T comp, T prim_feas, T dual_feas,
                     T sig_comp, T ncp_comp);

  void computeStatistics(const ContactProblem<T, IceCreamCone> &prob,
                         const Ref<const VectorXs> &x,
                         const Ref<const VectorXs> &z,
                         const Ref<const VectorXs> &y,
                         const Ref<const VectorXs> &v,
                         const Ref<const VectorXs> &v_reg);

  T stoppingCriteria(const ContactProblem<T, IceCreamCone> &prob,
                     const Ref<const VectorXs> &x,
                     const Ref<const VectorXs> &x_pred,
                     const Ref<const VectorXs> &z,
                     const Ref<const VectorXs> &z_pred,
                     const Ref<const VectorXs> &y, const T rho_, const T rho);

  void setCompliance(const ContactProblem<T, IceCreamCone> &prob,
                     T eps_reg = 1e-6);

  void setCompliance(const Ref<const VectorXs> &R_reg);

  template <typename MatIn>
  void evaluateEigVals(const Eigen::MatrixBase<MatIn> &P, T rho = 1e-6,
                       T eps = 1e-6, int max_iter = 20);

  bool solve(const ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0,
             ContactSolverSettings<T> &settings, T rho = 1e-6,
             T over_relax = 1., T eps_reg = 1e-6);

  bool solve(const ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0,
             ContactSolverSettings<T> &settings,
             const Ref<const VectorXs> &R_reg, T rho = 1e-6, T over_relax = 1.);

  const Ref<const VectorXs> getSolution() const;

  const Ref<const VectorXs> getDualSolution();

  // Statistics
  Statistics<T> stats_;

protected:
  MatrixXs P_, Pinv_, Ptild_, A_, ATA_;
  MatrixXs P_llt_;
  Eigen::LLT<MatrixXs> llt_;
  VectorXs vpow_, Mvpow_, err_vpow_;
  VectorXs R_reg_, rhos_;
  VectorXs x_, x_pred_, x_v_, dx_, z_, z_pred_, dz_, z_or_, y_, v_, v_reg_,
      v_proj_, q_, x0_, lam_;

  bool _solve_impl(const ContactProblem<T, IceCreamCone> &prob,
                   const Ref<const VectorXs> &x0, const Ref<const VectorXs> &y0,
                   ContactSolverSettings<T> &settings,
                   const Ref<const VectorXs> &R_reg, T rho_admm = 1e-6,
                   T rho = 1e-6, T over_relax = 1.);

  bool _solve_impl(const ContactProblem<T, IceCreamCone> &prob,
                   const Ref<const VectorXs> &x0, const Ref<const VectorXs> &y0,
                   ContactSolverSettings<T> &settings, T rho_admm = 1e-6,
                   T rho = 1e-6, T over_relax = 1., T eps_reg = 1e-6);

  bool _solve(ContactProblem<T, IceCreamCone> &prob,
              const Ref<const VectorXs> &lam0,
              ContactSolverSettings<T> &settings);
  using Base::timer_;
};

template <typename T> struct CCPNewtonPrimalSolver : CCPBaseSolver<T> {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);

public:
  typedef CCPBaseSolver<T> Base;
  using Base::nc_;
  int nv_;
  int max_iter_, n_iter_;
  T th_stop_, stop_, rel_th_stop_, rel_stop_, comp_, comp_reg_, sig_comp_,
      sig_comp_reg_, ncp_comp_, ncp_comp_reg_, prim_feas_, dual_feas_,
      dual_feas_reg_, alpha_, cost_, cost_try_;

  CCPNewtonPrimalSolver();

  void setProblem(const ContactProblem<T, IceCreamCone> &prob);

  void setCompliance(const ContactProblem<T, IceCreamCone> &prob,
                     T eps_reg = 1e-6);

  void setCompliance(const Ref<const VectorXs> &R_reg);

  T stoppingCriteria(const ContactProblem<T, IceCreamCone> &prob,
                     const Ref<const VectorXs> &x,
                     const Ref<const VectorXs> &v);

  T relativeStoppingCriteria(const Ref<const VectorXs> &x,
                             const Ref<const VectorXs> &x_pred);

  void computeStatistics(const ContactProblem<T, IceCreamCone> &prob,
                         const Ref<const VectorXs> &R_reg,
                         const Ref<const VectorXs> &dq,
                         const Ref<const VectorXs> &lam);

  void addStatistics(T stop, T rel_stop, T comp, T sig_comp, T ncp_comp,
                     T prim_feas, T dual_feas, T cost);

  template <typename VecIn, typename VecIn2, typename VecOut>
  void complianceMap(const ContactProblem<T, IceCreamCone> &prob,
                     const Eigen::MatrixBase<VecIn> &R,
                     const Eigen::MatrixBase<VecIn2> &y,
                     const Eigen::MatrixBase<VecOut> &y_out);

  template <typename VecIn, typename VecIn2, typename VecOut>
  void projKR(const ContactProblem<T, IceCreamCone> &prob,
              const Eigen::MatrixBase<VecIn> &R,
              const Eigen::MatrixBase<VecIn2> &y,
              const Eigen::MatrixBase<VecOut> &projy);

  template <typename VecIn, typename VecIn2, typename VecIn3, typename VecOut>
  void projKR(const ContactProblem<T, IceCreamCone> &prob,
              const Eigen::MatrixBase<VecIn> &R,
              const Eigen::MatrixBase<VecIn2> &R_sqrt,
              const Eigen::MatrixBase<VecIn3> &y,
              const Eigen::MatrixBase<VecOut> &projy);

  template <typename VecIn, typename VecIn2>
  T regularizationCost(const ContactProblem<T, IceCreamCone> &prob,
                       const Eigen::MatrixBase<VecIn> &R,
                       const Eigen::MatrixBase<VecIn2> &dq);

  template <typename VecIn, typename VecIn2, typename VecIn3>
  T regularizationCost(const ContactProblem<T, IceCreamCone> &prob,
                       const Eigen::MatrixBase<VecIn> &R,
                       const Eigen::MatrixBase<VecIn2> &dq,
                       const Eigen::MatrixBase<VecIn3> &y);

  template <typename VecIn, typename VecIn2, typename VecIn3, typename VecIn4>
  T regularizationCost(const ContactProblem<T, IceCreamCone> &prob,
                       const Eigen::MatrixBase<VecIn> &R,
                       const Eigen::MatrixBase<VecIn2> &dq,
                       const Eigen::MatrixBase<VecIn3> &y,
                       const Eigen::MatrixBase<VecIn4> &lam);

  template <typename VecIn, typename VecIn2>
  T unconstrainedCost(const ContactProblem<T, IceCreamCone> &prob,
                      const Eigen::MatrixBase<VecIn> &R,
                      const Eigen::MatrixBase<VecIn2> &dq);

  template <typename VecIn, typename VecIn2, typename VecIn3>
  T unconstrainedCost(const ContactProblem<T, IceCreamCone> &prob,
                      const Eigen::MatrixBase<VecIn> &R,
                      const Eigen::MatrixBase<VecIn2> &dq,
                      const Eigen::MatrixBase<VecIn3> &y);

  template <typename VecIn, typename VecIn2, typename VecIn3, typename VecIn4>
  T unconstrainedCost(const ContactProblem<T, IceCreamCone> &prob,
                      const Eigen::MatrixBase<VecIn> &R,
                      const Eigen::MatrixBase<VecIn2> &dq,
                      const Eigen::MatrixBase<VecIn3> &y,
                      const Eigen::MatrixBase<VecIn4> &lam);

  template <typename VecIn, typename VecIn2, typename VecOut>
  void computeRegularizationGrad(const ContactProblem<T, IceCreamCone> &prob,
                                 const Eigen::MatrixBase<VecIn> &R,
                                 const Eigen::MatrixBase<VecIn2> &dq,
                                 const Eigen::MatrixBase<VecOut> &grad);

  template <typename VecIn, typename VecIn2, typename VecIn3, typename VecOut>
  void computeRegularizationGrad(const ContactProblem<T, IceCreamCone> &prob,
                                 const Eigen::MatrixBase<VecIn> &R,
                                 const Eigen::MatrixBase<VecIn2> &dq,
                                 const Eigen::MatrixBase<VecIn3> &y,
                                 const Eigen::MatrixBase<VecOut> &grad);

  template <typename VecIn, typename VecIn2, typename VecIn3, typename VecIn4,
            typename VecOut>
  void computeRegularizationGrad(const ContactProblem<T, IceCreamCone> &prob,
                                 const Eigen::MatrixBase<VecIn> &R,
                                 const Eigen::MatrixBase<VecIn2> &dq,
                                 const Eigen::MatrixBase<VecIn3> &y,
                                 const Eigen::MatrixBase<VecIn4> &lam,
                                 const Eigen::MatrixBase<VecOut> &grad);

  template <typename VecIn, typename VecIn2, typename VecOut>
  void computeGrad(const ContactProblem<T, IceCreamCone> &prob,
                   const Eigen::MatrixBase<VecIn> &R,
                   const Eigen::MatrixBase<VecIn2> &dq,
                   const Eigen::MatrixBase<VecOut> &grad);

  template <typename VecIn, typename VecIn2, typename VecIn3, typename VecOut>
  void computeGrad(const ContactProblem<T, IceCreamCone> &prob,
                   const Eigen::MatrixBase<VecIn> &R,
                   const Eigen::MatrixBase<VecIn2> &dq,
                   const Eigen::MatrixBase<VecIn3> &y,
                   const Eigen::MatrixBase<VecOut> &grad);

  template <typename VecIn, typename VecIn2, typename VecIn3, typename VecIn4,
            typename VecOut>
  void computeGrad(const ContactProblem<T, IceCreamCone> &prob,
                   const Eigen::MatrixBase<VecIn> &R,
                   const Eigen::MatrixBase<VecIn2> &dq,
                   const Eigen::MatrixBase<VecIn3> &y,
                   const Eigen::MatrixBase<VecIn4> &lam,
                   const Eigen::MatrixBase<VecOut> &grad);

  template <typename VecIn, typename VecIn2, typename VecIn3, typename MatOut>
  void computeHessReg(const ContactProblem<T, IceCreamCone> &prob,
                      const Eigen::MatrixBase<VecIn> &R,
                      const Eigen::MatrixBase<VecIn2> &y,
                      const Eigen::MatrixBase<VecIn3> &y_tilde,
                      const Eigen::MatrixBase<MatOut> &H);

  template <typename VecIn, typename VecIn2, typename MatOut>
  void computeHess(const ContactProblem<T, IceCreamCone> &prob,
                   const Eigen::MatrixBase<VecIn> &R,
                   const Eigen::MatrixBase<VecIn2> &dq,
                   const Eigen::MatrixBase<MatOut> &H);

  template <typename VecIn, typename VecIn2, typename VecIn3, typename MatOut>
  void computeHess(const ContactProblem<T, IceCreamCone> &prob,
                   const Eigen::MatrixBase<VecIn> &R,
                   const Eigen::MatrixBase<VecIn2> &dq,
                   const Eigen::MatrixBase<VecIn3> &y,
                   const Eigen::MatrixBase<MatOut> &H);

  template <typename VecIn, typename VecIn2, typename VecIn3, typename VecIn4,
            typename MatOut>
  void computeHess(const ContactProblem<T, IceCreamCone> &prob,
                   const Eigen::MatrixBase<VecIn> &R,
                   const Eigen::MatrixBase<VecIn2> &dq,
                   const Eigen::MatrixBase<VecIn3> &y,
                   const Eigen::MatrixBase<VecIn4> &y_tilde,
                   const Eigen::MatrixBase<MatOut> &H);

  template <typename VecIn, typename MatIn, typename VecOut>
  void computeDescentDirection(const Eigen::MatrixBase<VecIn> &grad,
                               const Eigen::MatrixBase<MatIn> &H,
                               const Eigen::MatrixBase<VecOut> &ddq);

  bool solve(const ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0,
             ContactSolverSettings<T> &settings,
             const Ref<const VectorXs> &R_reg);

  bool solve(const ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0,
             ContactSolverSettings<T> &settings, T eps_reg = 1e-6);

  const Ref<const VectorXs> getCompliance() const;

  const Ref<const VectorXs> getSolution() const;

  const Ref<const VectorXs> getDualSolution() const;

  void resetStats();

  // Statistics
  Statistics<T> stats_;

protected:
  std::vector<T> mus_tilde_, mus_hat_;
  VectorXs l_, dL_dxyz_;
  VectorXs dq_, dq_pred_, dq_try_, ddq_, ddq2_, ddq3_, lam_, R_reg_, R_sqrt_,
      y_, y_tilde_, dvstar_;
  MatrixXs H_, H_yy_;
  VectorXs grad_;
  Vector2s t_dir_;
  Matrix2s P_t_, P_t_perp_;
  VectorXs v_, v_cor_, v_proj_, M_diag_sqrt_;
  using Base::dL_dDel_, Base::dL_dmus_, Base::dL_dg_;
  VectorXs dL_db_;
  MatrixXs dL_dA_;
  MatrixXs H_llt_;
  Eigen::LLT<MatrixXs> llt_;

  bool _solve_impl(const ContactProblem<T, IceCreamCone> &prob,
                   const Ref<const VectorXs> &lam0,
                   ContactSolverSettings<T> &settings,
                   const Ref<const VectorXs> &R_reg);

  bool _solve_impl(const ContactProblem<T, IceCreamCone> &prob,
                   const Ref<const VectorXs> &lam0,
                   ContactSolverSettings<T> &settings, T eps_reg = 1e-6);

  bool _solve(ContactProblem<T, IceCreamCone> &prob,
              const Ref<const VectorXs> &lam0,
              ContactSolverSettings<T> &settings);
  using Base::timer_;
};

template <typename T> struct RaisimSolver : DualBaseSolver<T, IceCreamCone> {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);

public:
  typedef DualBaseSolver<T, IceCreamCone> Base;
  using Base::nc_;
  // convergence criterion
  T th_stop_, stop_, rel_th_stop_, rel_stop_;
  T comp_, prim_feas_, dual_feas_, sig_comp_, ncp_comp_;
  int n_iter_, max_iter_, n_iter_mod_, n_iter_bis_;
  // hyper-params
  T beta1_, beta2_, beta3_, gamma_;
  T alpha_, alpha_min_;
  T th_;

  RaisimSolver() { timer_.stop(); };

  void computeGinv(const Ref<const MatrixXs> &G);

  const Ref<const Matrix3s> getGinv(int i) const;

  void computeGlam(const Ref<const MatrixXs> &G,
                   const Ref<const VectorXs> &lam);

  template <typename VecIn>
  void updateGlam(int i, const Ref<const MatrixXs> &G,
                  const Eigen::MatrixBase<VecIn> &lami);

  const Ref<const Vector3s> getGlam(int i, int j) const;

  void setGlam(int i, int j, Vector3s Glamij);

  void computeC(const Ref<const MatrixXs> &G, const Ref<const VectorXs> &g,
                const Ref<const VectorXs> &lam);

  template <typename VecIn>
  void updateC(int i, const Ref<const MatrixXs> &G,
               const Ref<const VectorXs> &g,
               const Eigen::MatrixBase<VecIn> &lami);

  const Ref<const Vector3s> getC(int i);

  void setProblem(const ContactProblem<T, IceCreamCone> &prob);

  template <typename MatIn, typename VecIn, typename VecOut>
  void computeLamV0(const Eigen::MatrixBase<MatIn> &Ginvi,
                    const Eigen::MatrixBase<VecIn> &ci,
                    const Eigen::MatrixBase<VecOut> &lam_out);

  // const Vector3s &getLamV0();

  template <typename MatIn>
  void computeH1Grad(const Eigen::MatrixBase<MatIn> &G, Ref<Vector3s> grad_out);

  template <typename VecIn>
  void computeH2Grad(const T mu, const Eigen::MatrixBase<VecIn> &lam,
                     Ref<Vector3s> grad_out);

  template <typename MatIn, typename VecIn>
  void computeEta(const Eigen::MatrixBase<MatIn> &G, const T mu,
                  const Eigen::MatrixBase<VecIn> &lam, Ref<Vector3s> eta_out);

  template <typename MatIn>
  void computeEta(const Eigen::MatrixBase<MatIn> &G,
                  const Ref<const Vector3s> &gradH2, Ref<Vector3s> eta_out);

  template <typename MatIn>
  void computeGbar(const Eigen::MatrixBase<MatIn> &G,
                   Eigen::Matrix<T, 3, 2> &Gbar);

  template <typename MatIn, typename VecIn>
  void computeCbar(const Eigen::MatrixBase<MatIn> &G,
                   const Eigen::MatrixBase<VecIn> &c, Ref<Vector3s> cbar);

  template <typename MatIn, typename VecIn, typename VecIn2>
  T computeGradTheta(const Eigen::MatrixBase<MatIn> &G,
                     const Eigen::MatrixBase<VecIn> &c, const T mu,
                     const Eigen::MatrixBase<VecIn2> &lam);

  template <typename MatIn, typename VecIn, typename VecIn2>
  T computeGradTheta(const Eigen::MatrixBase<MatIn> &G,
                     const Eigen::MatrixBase<VecIn> &c,
                     const Eigen::MatrixBase<VecIn2> &lam,
                     const Ref<const Vector3s> &gradH2);

  T computeLamZ(const T mu, const T r);

  template <typename MatIn, typename VecIn>
  T computeR(const Eigen::MatrixBase<MatIn> &G,
             const Eigen::MatrixBase<VecIn> &c, const T mu, const T theta);

  template <typename VecOut>
  void computeLam(const T r, const T theta, const T lamZ,
                  Eigen::MatrixBase<VecOut> &lam_out);

  template <typename VecIn> T computeTheta(const Eigen::MatrixBase<VecIn> &lam);

  Vector3s initialStep();

  template <typename MatIn, typename MatIn2, typename VecIn>
  void bisectionStep(const Eigen::MatrixBase<MatIn> &G,
                     const Eigen::MatrixBase<MatIn2> &Ginv,
                     const Eigen::MatrixBase<VecIn> &c, const T mu,
                     const Ref<const Vector3s> &lam_v0, Ref<Vector3s> lam_out,
                     int max_iter = 100, T th = 1e-6, T beta1 = 1e-2,
                     T beta2 = 0.5, T beta3 = 1.3);

  T stoppingCriteria(const ContactProblem<T, IceCreamCone> &prob,
                     const Ref<const VectorXs> &lam,
                     const Ref<const VectorXs> &v);

  T relativeStoppingCriteria(const Ref<const VectorXs> &lam,
                             const Ref<const VectorXs> &lam_pred);

  void computeStatistics(const ContactProblem<T, IceCreamCone> &prob,
                         const Ref<const VectorXs> &lam,
                         const Ref<const VectorXs> &v);

  void addStatistics(T stop, T rel_stop, T comp, T sig_comp, T ncp_comp,
                     T prim_feas, T dual_feas);

  bool solve(const ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0,
             ContactSolverSettings<T> &settings, T eps_reg = 0., T alpha = 1.,
             T alpha_min = 0.1, T beta1 = 1e-2, T beta2 = 0.5, T beta3 = 1.3,
             T gamma = 0.99, T th = 1e-6);

  bool solve(const ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0,
             ContactSolverSettings<T> &settings,
             const Ref<const VectorXs> &R_reg, T alpha = 1., T alpha_min = 0.1,
             T beta1 = 1e-2, T beta2 = 0.5, T beta3 = 1.3, T gamma = 0.99,
             T th = 1e-6);

  // const Ref<const VectorXs> getSolution() const;

  const Ref<const VectorXs> getDualSolution() const;

  void resetStats();

  // Statistics
  Statistics<T> stats_;

protected:
  // problem quantities
  MatrixXs c_;
  MatrixXs Ginv_;
  MatrixXs Glam_;
  using Base::lam_, Base::lam_fd_;
  VectorXs lam_pred_, lam_proj_, dlam_, v_, v_cor_, v_proj_;

  bool _solve(ContactProblem<T, IceCreamCone> &prob,
              const Ref<const VectorXs> &lam0,
              ContactSolverSettings<T> &settings);
  using Base::timer_;
};

template <typename T> struct RaisimCorrectedSolver : RaisimSolver<T> {
  CONTACTBENCH_EIGEN_TYPEDEFS(T);

public:
  typedef RaisimSolver<T> Base;
  using Base::nc_, Base::max_iter_, Base::n_iter_, Base::n_iter_bis_,
      Base::n_iter_mod_, Base::stop_, Base::th_stop_, Base::rel_stop_,
      Base::rel_th_stop_, Base::th_, Base::alpha_, Base::alpha_min_,
      Base::beta1_, Base::beta2_, Base::beta3_, Base::gamma_, Base::stats_,
      Base::comp_, Base::sig_comp_, Base::ncp_comp_, Base::prim_feas_,
      Base::dual_feas_;
  RaisimCorrectedSolver() { timer_.stop(); };

  template <typename VecIn, typename VecIn2>
  Vector3s computeCorrectedC(const MatrixBase<VecIn> &ci,
                             const MatrixBase<VecIn2> &vi, const T mu);

  template <typename MatIn, typename VecIn, typename VecIn2, typename VecOut>
  void computeCorrectedLamV0(const Eigen::MatrixBase<MatIn> &Ginvi,
                             const Eigen::MatrixBase<VecIn> &ci,
                             const Eigen::MatrixBase<VecIn2> &vi, const T mu,
                             const Eigen::MatrixBase<VecOut> &lam_out);

  template <typename MatIn, typename MatIn2, typename VecIn, typename VecIn2>
  void bisectionStepCorrected(const Eigen::MatrixBase<MatIn> &G,
                              const Eigen::MatrixBase<MatIn2> &Ginv,
                              const Eigen::MatrixBase<VecIn> &c,
                              const Eigen::MatrixBase<VecIn2> &v, const T mu,
                              const Ref<const Vector3s> &lam_v0_cor,
                              Ref<Vector3s> lam_out, int max_iter = 100,
                              T th = 1e-6, T beta1 = 1e-2, T beta2 = 0.5,
                              T beta3 = 1.3);

  bool solve(const ContactProblem<T, IceCreamCone> &prob,
             const Ref<const VectorXs> &lam0,
             ContactSolverSettings<T> &settings, T eps_reg = 0., T alpha = 1.,
             T alpha_min = 0.1, T beta1 = 1e-2, T beta2 = 0.5, T beta3 = 1.3,
             T gamma = 0.99, T th = 1e-6);

  using Base::computeLam, Base::computeTheta, Base::computeR, Base::computeLamZ,
      Base::computeH1Grad, Base::computeH2Grad, Base::computeGradTheta,
      Base::computeGinv, Base::computeC, Base::computeLamV0, Base::updateC,
      Base::stoppingCriteria, Base::relativeStoppingCriteria,
      Base::computeStatistics, Base::addStatistics;

protected:
  using Base::lam_, Base::lam_pred_, Base::v_, Base::c_, Base::Ginv_;

  bool _solve(ContactProblem<T, IceCreamCone> &prob,
              const Ref<const VectorXs> &lam0,
              ContactSolverSettings<T> &settings);
  using Base::timer_;
};

} // end namespace contactbench

#endif

#include "contactbench/solvers.hxx"

#if CONTACTBENCH_ENABLE_TEMPLATE_INSTANTIATION
#include "contactbench/solvers.txx"
#endif // CONTACTBENCH_ENABLE_TEMPLATE_INSTANTIATION
