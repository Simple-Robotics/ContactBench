import numpy as np
import pinocchio as pin
import abc
from typing import List
from pycontact import ContactProblem, IceCreamCone, ContactSolverSettings
from pycontact.utils.callbacks import CallBackBase
from cvxpy import Problem, Variable, SOC, Minimize, SCS
import scs
import scipy
import time


class PyContactSolver(abc.ABC):
    # Any new solver should follow the template of this class
    def __init__(self) -> None:
        super().__init__()
        self.callbacks: List[CallBackBase] = []
        self.stats_ = PyStatisticsContactSolver()

    def setProblem(self, problem):
        self.nc_ = problem.nc_
        return

    @abc.abstractclassmethod
    def solve(self, problem: ContactProblem, lam0: np.ndarray, settings: ContactSolverSettings):
        pass

    @abc.abstractclassmethod
    def getSolution(self):
        pass

    def addCallback(self, callback: CallBackBase):
        self.callbacks += [callback]

    def call(self, **kwargs):
        for callback in self.callbacks:
            callback.call(**kwargs)

    def resetCallbacks(self, **kwargs):
        for callback in self.callbacks:
            callback.reset(**kwargs)

    def plotCallback(self, figs, axs, block=True):
        callbacks_figs, callbacks_axs = [], []
        for i, callback in enumerate(self.callbacks):
            new_figs, new_axs = callback.plot(figs[i], axs[i], block)
            callbacks_figs += [new_figs]
            callbacks_axs += [new_axs]
        return callbacks_figs, callbacks_axs

    def close_plot_callback(self):
        for callback in self.callbacks:
            callback.close_plot()

class PyStatisticsContactSolver():

    def __init__(self) -> None:
        self.ncp_comp_ = []
        self.sig_comp_ = []
        self.comp_ = []

class PyPrimalContactSolver(PyContactSolver):

    def getSolution(self):
        return self.dq_

    def getDualSolution(self):
        return self.lam_


class PyDualContactSolver(PyContactSolver):

    def getSolution(self):
        return self.lam_

    def getDualSolution(self):
        return self.v_
    
# ====================================================================================================
# ================================== PINOCCHIO SOLVERS ==============================================

class PinNCPPGSSolver(PyDualContactSolver):

    def getCPUTimes(self):
        return self.pin_solver.getCPUTimes()

    def solve(self, problem, lam0, settings:ContactSolverSettings, over_relax= 1., **kwargs):
        self.pin_solver = pin.PGSContactSolver(3*problem.nc_)
        self.pin_solver.setAbsolutePrecision(settings.th_stop_)
        self.pin_solver.setRelativePrecision(settings.rel_th_stop_)
        self.pin_solver.setMaxIterations(settings.max_iter_)
        problem.Del_.computeChol(1e-9)
        problem.Del_.evaluateDel()

        # For saving contact problems
        self.lam_ = lam0.copy()
        self.primal_warm_start = self.lam_.copy()
        self.contact_problem = problem

        self.lam_ = lam0.copy()
        has_converged = self.pin_solver.solve(problem.Del_.G_,
                                              problem.g_,
                                              [pin.CoulombFrictionCone(cons.mu_) for cons in problem.contact_constraints_],
                                              self.lam_)
        print(self.pin_solver.getCPUTimes().user)
        self.n_iter_ = self.pin_solver.getIterationCount()
        self.vc_ = problem.Del_.G_ @ self.lam_ + problem.g_
        return has_converged

class PinNCPADMMSolver(PyDualContactSolver):
    def __init__(self) -> None:
        super().__init__()
        self.rho_power = None
        self.rho = None

    def getCPUTimes(self):
        return self.pin_solver.getCPUTimes()

    def solve(self, problem, lam0, settings:ContactSolverSettings, **kwargs):
        self.pin_solver = pin.ADMMContactSolver(3*problem.nc_, mu_prox=1e-6, tau=0.5)
        self.pin_solver.setAbsolutePrecision(settings.th_stop_)
        self.pin_solver.setRelativePrecision(settings.rel_th_stop_)
        self.pin_solver.setMaxIterations(settings.max_iter_)
        problem.Del_.computeChol(1e-9)

        # For saving contact problems
        self.lam_ = lam0.copy()
        self.primal_warm_start = self.lam_.copy()
        self.contact_problem = problem

        delassus_sparse = problem.Del_.contact_chol_.getDelassusCholeskyExpression()
        delassus_dense = pin.DelassusOperatorDense(delassus_sparse.matrix())
        has_converged = self.pin_solver.solve(delassus_dense,
                                              problem.g_,
                                              [pin.CoulombFrictionCone(cons.mu_) for cons in problem.contact_constraints_],
                                              problem.R_comp_,
                                              primal_solution=lam0,
                                              stat_record=False)
        self.n_iter_ = self.pin_solver.getIterationCount()
        self.rho = self.pin_solver.getRho()
        self.rho_power = self.pin_solver.getRhoPower()
        return has_converged

# ====================================================================================================
# ================================== CVXPY SOLVERS =================================================

class PyCCPCVXSolver(PyDualContactSolver):

    def solve(
        self,
        problem: ContactProblem,
        lam0,
        settings:ContactSolverSettings,
        eps_reg = 0.,
        **kwargs
    ):
        problem.Del_.evaluateDel()
        G_sqrt = np.linalg.cholesky(problem.Del_.G_)
        G_sqrtinv_g, _, _ , _  = np.linalg.lstsq(G_sqrt, problem.g_, rcond=None)
        nc = problem.nc_
        x = Variable(3*nc +1)
        soc_constraints = [
            SOC( problem.contact_constraints_[i].mu_*x[3*i],x[3*i:3*i+2]) for i in range(nc)
        ]
        soc_constraints += [SOC(x[-1], G_sqrt.T @ x[:-1] + G_sqrtinv_g)]
        opt_prob = Problem(Minimize(x[-1]),
                        soc_constraints)
        t = np.linalg.norm(G_sqrt.T @ lam0 + G_sqrtinv_g)
        x.value = np.concatenate([lam0[:,0],np.array([t])])
        opt_prob.solve(solver = SCS, max_iters =settings.max_iter_, th_stop= settings.th_stop_, warm_start = True)
        self.lam_ = x.value[:-1]
        self.prim_feas = opt_prob.solver_stats.extra_stats['info']['res_pri']
        self.dual_feas = opt_prob.solver_stats.extra_stats['info']['res_dual']
        self.n_iter_ = opt_prob.solver_stats.num_iters
        self.prim_infeas = opt_prob.solver_stats.extra_stats['info']['res_infeas']
        self.dual_infeas = max(opt_prob.solver_stats.extra_stats['info']['res_unbdd_a'],opt_prob.solver_stats.extra_stats['info']['res_unbdd_p'])
        return opt_prob.status == 'optimal'


# ====================================================================================================
# ===========================================SCS SOLVER===============================================

class CCPSCSTimings:
    user: float = 0.0

    def __init__(self, time):
        self.user = time

class PyCCPSCSSolver(PyDualContactSolver):

    def getCPUTimes(self):
        return self.solve_time

    def solve(
        self,
        problem: ContactProblem,
        lam0,
        settings:ContactSolverSettings,
        eps_reg = 0.,
        timings = False,
        **kwargs
    ):
        t_start = 0
        t_end = 0
        if timings:
            t_start = time.time_ns()
        nc = len(problem.contact_constraints_)
        problem.Del_.computeChol(1e-9)
        problem.Del_.evaluateDel()
        A = np.zeros((3*nc,3*nc))
        for i in range(nc):
            A[3*i, 3*i+2] = -problem.contact_constraints_[i].mu_
            A[3*i+1, 3*i] = -1.
            A[3*i+2, 3*i+1] = -1.
        b = np.zeros(3*nc)

        # Populate dicts with data to pass into SCS
        data = dict(P=scipy.sparse.csc_matrix(problem.Del_.G_), A=scipy.sparse.csc_matrix(A), b=b, c=problem.g_)
        cone = dict(q=[3]*nc)

        # Initialize solver
        solver = scs.SCS(data, cone,  max_iters=settings.max_iter_, eps_abs=settings.th_stop_, eps_rel=settings.rel_th_stop_, verbose = False)
        if timings:
            t_end = time.time_ns()
        self.solve_time = CCPSCSTimings(t_end - t_start)
        y = (problem.Del_.G_ @ lam0)[:,0]
        y += problem.g_
        sol = solver.solve( x=lam0[:,0], y=y, s=lam0[:,0])
        self.n_iter_ = sol['info']['iter']
        self.lam_ = sol["x"]
        self.v_ = sol["y"]

# ====================================================================================================
# ===========================================PYTHON SOLVERS==============================================


class PyNCPPGSSolver(PyDualContactSolver):
    def stoppingCriteria(self, problem, x, v):
        self.prim_feas = 0.0
        self.dual_feas = 0.0
        self.comp = problem.computeContactComplementarity(x, v)
        return self.comp

    def relativeStoppingCriteria(self, x, x_pred):
        norm_pred = np.linalg.norm(x_pred)
        if norm_pred > 0:
            crit = np.linalg.norm(x - x_pred) / norm_pred
        else:
            crit = np.inf
        return crit

    def solve(self, problem, lam0, settings:ContactSolverSettings, over_relax= 1., **kwargs):
        self.resetCallbacks()
        x = lam0.copy()
        v = np.zeros_like(x)
        problem.Del_.computeChol(1e-9)
        problem.Del_.evaluateDel()
        d = np.diagonal(problem.Del_.G_)
        for j in range(settings.max_iter_):
            x_pred = x.copy()
            for i in range(self.nc_):
                cons_i = problem.contact_constraints_[i]
                v[3 * i + 2] = problem.Del_.G_[3 * i + 2] @ x + problem.g_[3 * i + 2]
                x[3 * i + 2] = x[3 * i + 2] - (over_relax/ d[3 * i + 2]) * (v[3 * i + 2])
                x[3 * i + 2] = max([0.,x[3 * i + 2]])
                v[3 * i : 3 * i + 2] = (
                    problem.Del_.G_[3 * i : 3 * i + 2] @ x + problem.g_[3 * i : 3 * i + 2, None]
                )
                x_tmp = (
                    x[3 * i : 3 * i + 2]
                    - (over_relax / min(d[3 * i], d[3 * i + 1])) * v[3 * i : 3 * i + 2]
                )
                x_tmp2 = np.zeros((3,1))
                cons_i.projectHorizontal(
                    np.array([x_tmp[0], x_tmp[1], x[i * 3 + 2]]), x_tmp2
                )
                x[i * 3 : 3 * i + 2] = x_tmp2[:2]
            rel_crit = self.relativeStoppingCriteria(x, x_pred)
            crit = self.stoppingCriteria(problem, x, v)
            self.call(
                G=problem.Del_.G_,
                g=problem.g_,
                lam=x,
                mu=[cons.mu_ for cons in problem.contact_constraints_],
                prim_feas=self.prim_feas,
                dual_feas=self.dual_feas,
                comp=self.comp,
            )
            if rel_crit < settings.rel_th_stop_ or crit < settings.th_stop_:
                self.lam_ = x
                return True
        self.lam_ = x
        return False

class PyCCPPGSSolver(PyDualContactSolver):
    def setProblem(self, problem):
        PyDualContactSolver.setProblem(self, problem)

    def stoppingCriteria(self, problem, x, v):
        self.prim_feas = 0.0
        projv = np.zeros_like(v)
        problem.projectDual(v, projv)
        self.dual_feas = np.linalg.norm(v - projv, ord=np.inf)
        self.comp = problem.computeConicComplementarity(x, v)
        return np.max([self.comp, self.dual_feas])

    def relativeStoppingCriteria(self, x, x_pred):
        norm_pred = np.linalg.norm(x_pred)
        if norm_pred > 0:
            crit = np.linalg.norm(x - x_pred) / norm_pred
        else:
            crit = np.inf
        return crit

    def solve(
        self, problem: ContactProblem, lam0: np.ndarray, settings:ContactSolverSettings, imp=None, **kwargs
    ):
        self.resetCallbacks()
        problem.Del_.evaluateDel()
        if imp is None:
            R = np.zeros_like(problem.Del_.G_)
        else:
            R = np.diag(((1.0 - imp) / imp) * np.diag(problem.Del_.G_))
        self.Gtild = problem.Del_.G_ + R
        x = lam0.copy()
        v = np.zeros_like(x)
        d = np.diagonal(self.Gtild)
        for j in range(settings.max_iter_):
            x_pred = x.copy()
            for i in range(self.nc_):
                cons_i = problem.contact_constraints_[i]
                v[3 * i : 3 * (i + 1)] = (
                    np.dot(self.Gtild[3 * i : 3 * (i + 1)], x)
                    + problem.g_[3 * i : 3 * (i + 1),None]
                )
                x[3 * i : 3 * (i + 1)] += (
                    -(3 / (d[3 * i] + d[3 * i + 1] + d[3 * i + 2]))
                    * v[3 * i : 3 * (i + 1)]
                )
                cons_i.project(x[i * 3 : 3 * i + 3], x[3 * i : 3 * ((i + 1))])
            rel_crit = self.relativeStoppingCriteria(x, x_pred)
            crit = self.stoppingCriteria(problem, x, v)
            self.call(
                G=problem.Del_.G_,
                g=problem.g_,
                lam=x,
                mu=[cons.mu_ for cons in problem.contact_constraints_],
                prim_feas=self.prim_feas,
                dual_feas=self.dual_feas,
                comp=self.comp,
            )
            if rel_crit < settings.rel_th_stop_ or crit < settings.th_stop_:
                self.lam_ = x
                self.niter_ = j
                return True
        self.niter_ = settings.max_iter_
        self.lam_ = x
        return False

class PyCCPADMMSolver(PyDualContactSolver):

    def updateRho(self, prim_feas, dual_feas):
        if prim_feas / dual_feas > 10.0:
            self.rho *= (self.eigval_max / self.eigval_min) ** 0.1
            return True
        elif prim_feas / dual_feas < 0.1:
            self.rho /= (self.eigval_max / self.eigval_min) ** 0.1
            return True
        else:
            return False

    def computeLargestEigenValue(self, G, th_stop=1e-6, max_iter=10):
        v = np.random.random(G.shape[0])
        for i in range(max_iter):
            v = G @ v
            v /= np.linalg.norm(v)
        eigval_max = v.T @ G @ v
        return eigval_max

    def stoppingCriteria(self, problem, x, z, y, v):
        self.prim_feas = np.linalg.norm(x - z, ord=np.inf)
        self.dual_feas = np.linalg.norm(v + y, ord=np.inf)
        self.comp = problem.computeConicComplementarity(x, v)
        return np.max([self.prim_feas, self.dual_feas])

    def relativeStoppingCriteria(self, z, z_pred):
        norm_pred = np.linalg.norm(z_pred)
        if norm_pred > 0:
            crit = np.linalg.norm(z - z_pred) / norm_pred
        else:
            crit = np.inf
        return crit

    def solve(
        self,
        problem: ContactProblem,
        lam0,
        settings:ContactSolverSettings,
        rho=1e-6,
        over_relax=1.,
        eps_reg = 0.,
        **kwargs
    ):
        problem.Del_.evaluateDel()
        self.resetCallbacks()
        x = lam0.copy()
        z = x.copy()
        y = np.zeros_like(x)
        v = np.zeros_like(x)
        self.eigval_max = self.computeLargestEigenValue(problem.Del_.G_)
        self.eigval_min = rho
        self.rho = np.sqrt(self.eigval_max * self.eigval_min) * (
            (self.eigval_max / self.eigval_min) ** 0.4
        )
        Gtild = problem.Del_.G_ + (self.rho + rho+ eps_reg) * np.eye(3 * self.nc_)
        Ginv = np.linalg.inv(Gtild)
        for j in range(settings.max_iter_):
            x_pred = x.copy()
            z_pred = z.copy()
            v = (
                problem.Del_.G_ @ x + problem.g_[:,None]
            )
            v_reg = v+eps_reg*x
            x -= Ginv @ (
                v_reg + y
                + self.rho * (x_pred - z)
            )
            problem.project(
                over_relax * x
                + (1 - over_relax) * z_pred
                + y / self.rho, z
            )
            y += self.rho * (
                over_relax * x
                + (1 - over_relax) * z_pred
                - z
            )
            self.rel_stop_ = self.relativeStoppingCriteria(x, x_pred)
            self.stop_ = self.stoppingCriteria(problem, x, z, y, v_reg)
            self.call(
                G=problem.Del_.G_,
                g=problem.g_,
                lam=z,
                mu=[cons.mu_ for cons in problem.contact_constraints_],
                prim_feas=self.prim_feas,
                dual_feas=self.dual_feas,
                comp=self.comp,
            )
            newrho = self.updateRho(self.prim_feas, self.dual_feas)
            if newrho:
                Gtild = problem.Del_.G_ + (self.rho + rho+eps_reg) * np.eye(3 * self.nc_)
                Ginv = np.linalg.inv(Gtild)
            if self.rel_stop_ < settings.rel_th_stop_ or self.stop_ < settings.th_stop_:
                self.lam_ = z
                self.lam_2_ = x
                self.niter_ = j
                return True
        self.lam_ = z
        self.lam_2_ = x
        self.niter_ = settings.max_iter_
        return False

class PyCCPADMMPrimalSolver(PyPrimalContactSolver,PyCCPADMMSolver):


    def stoppingCriteria(self, problem, A, x, z, y, v):
        self.prim_feas = np.linalg.norm( A @ x - z, ord=np.inf)
        self.dual_feas = np.linalg.norm(v + A.T @ y, ord=np.inf)
        self.comp = problem.computeConicComplementarity(x, v)
        return np.max([self.prim_feas, self.dual_feas])

    def solve(
        self,
        problem: ContactProblem,
        lam0,
        settings:ContactSolverSettings,
        rho=1e-6,
        over_relax=1.,
        eps_reg = 1e-6,
        **kwargs
    ):
        self.resetCallbacks()
        nc = problem.nc_
        nv = problem.M_.shape[0]
        x = np.zeros(3*nc+nv)
        Minv = np.linalg.inv(problem.M_)
        x[:nv] = problem.dqf_ + Minv @ problem.J_.T @ lam0[:,0]
        x[-3*nc:] = problem.J_ @ x[:nv]
        z = np.zeros(3*nc)
        y = -lam0[:,0].copy()
        v = np.zeros_like(x)
        q = np.zeros(nv+3*nc)
        P = np.zeros((nv+3*nc,nv+3*nc))
        P[:nv,:nv] = problem.M_
        problem.Del_.computeChol(1e-9)
        problem.Del_.evaluateDel()
        R = eps_reg * problem.Del_.G_.diagonal()
        P[-3*nc:,-3*nc:] = np.diag(1/R)
        self.eigval_max = self.computeLargestEigenValue(P)
        self.eigval_max = 250000
        self.eigval_min = rho
        self.rho = np.sqrt(self.eigval_max * self.eigval_min) * (
            (self.eigval_max / self.eigval_min) ** 0.4
        )
        q[:nv] = -problem.M_ @ problem.dqf_
        q[-3*nc:] = -problem.vstar_/R
        A = np.zeros((3*nc,3*nc+nv))
        A[:,:nv] = problem.J_
        A[:,-3*nc:] = -np.eye(3*nc)
        ATA  = A.T @ A
        Ptild = P + rho * np.eye(3 * nc+nv) + self.rho * ATA
        Pinv = np.linalg.inv(Ptild)
        for j in range(settings.max_iter_):
            x_pred = x.copy()
            z_pred = z.copy()
            v = (
                P @ x + q
            )
            v_reg = v
            x -= Pinv @ (
                v_reg + A.T @ y
                + self.rho * (ATA @x_pred -A.T @ z)
            )
            x2 = - Pinv @ (q + A.T @ (y - self.rho *z) - rho*x_pred)
            problem.projectDual(
                over_relax * A @ x
                + (1 - over_relax) * z_pred
                + y / self.rho, z
            )
            y += self.rho * (
                over_relax * A @ x
                + (1 - over_relax) *  z_pred
                -  z
            )
            self.rel_stop_ = self.relativeStoppingCriteria(x, x_pred)
            self.stop_ = self.stoppingCriteria(problem, A,  x, z, y, v_reg)
            self.call(
                G=[],
                g=problem.g_,
                lam=z,
                mu=[cons.mu_ for cons in problem.contact_constraints_],
                prim_feas=self.prim_feas,
                dual_feas=self.dual_feas,
                comp=self.comp,
            )
            newrho = self.updateRho(self.prim_feas, self.dual_feas)
            if newrho:
                Ptild = P + rho * np.eye(3 * nc+nv) + self.rho * ATA
                Pinv = np.linalg.inv(Ptild)
            if self.stop_ < settings.th_stop_:
                self.v_ = z
                self.dq_ = x[:nv]
                self.lam_ = -y
                self.niter_ = j+1
                return True
        self.lam_ = -y
        self.v_ = z
        self.dq_ = x[:nv]
        self.niter_ = settings.max_iter_
        return False

class PyCCPNSNewtonPrimalSolver(PyPrimalContactSolver):

    def compute_mus_tilde(self, problem, R, mus_tilde = None):
        if mus_tilde is None:
            mus_tilde = []
            for j in range(problem.nc_):
                mus_tilde += [problem.contact_constraints_[j].mu_ * np.sqrt(R[3*j]/R[3*j+2])]
        return mus_tilde

    def compliance_mapping(self, problem, R, v):
        return -(v - problem.vstar_)/R

    def projK_R(self, problem, R, y, mus_tilde = None):
        mus_tilde = self.compute_mus_tilde(problem, R, mus_tilde)
        projy = np.zeros_like(y)
        ContactProblem.project(mus_tilde, np.sqrt(R) * y, projy)
        return (1./np.sqrt(R))*projy

    def dqtoy_mapping(self, problem, R, dq, y = None):
        if y is None:
            y = self.compliance_mapping(problem, R, problem.J_ @ dq)
        return y

    def dqtolam_mapping(self, problem, R, dq, y = None, lam=None):
        if lam is None:
            y = self.dqtoy_mapping(problem, R, dq, y)
            lam = self.projK_R(problem,R,y)
        return lam

    def cost_unconstrained(self, problem, R, dq, y = None, lam = None):
        cost = 0.5*(dq - problem.dqf_).T @ problem.M_ @ (dq - problem.dqf_)
        cost += self.cost_regularization(problem, R, dq, y, lam)
        return cost

    def cost_regularization(self, problem, R, dq, y= None, lam = None):
        lam = self.dqtolam_mapping(problem, R, dq, y, lam)
        return 0.5 * lam.dot(R * lam)

    def grady_cost_regularization(self, problem, R, y, lam =None, mus_tilde = None, th_stop = 1e-5):
        y_tilde = np.sqrt(R) *y
        grad_y_lr = np.zeros(3*problem.nc_)
        mus_tilde = self.compute_mus_tilde(problem, R, mus_tilde)
        for j in range(problem.nc_):
            mu_tildej = mus_tilde[j]
            stiction = IceCreamCone.isInside(mu_tildej, y_tilde[3*j:3*j+3], th_stop)
            if stiction:
                grad_y_lr[3*j:3*j+3] = R[3*j:3*j+3] * y[3*j:3*j+3]
            else:
                breaking = y_tilde[3*j+2] < -mu_tildej *np.linalg.norm(y_tilde[3*j:3*j+2])
                if breaking:
                    grad_y_lr[3*j:3*j+3] = 0
                else:
                    yr = np.linalg.norm(y[3*j:3*j+2])
                    muj = problem.contact_constraints_[j].mu_
                    mu_hatj = muj * R[3*j]/R[3*j+2]
                    sy = mu_hatj*yr + y[3*j+2]
                    t_dir = y[3*j:3*j+2]/yr
                    grad_y_lr[3*j:3*j+2] = muj * R[3*j] * t_dir
                    grad_y_lr[3*j+2] = R[3*j+2]
                    grad_y_lr[3*j:3*j+3] *= (sy/(1+mu_tildej*mu_tildej))
        return grad_y_lr

    def hessy_cost_regularization(self, problem, R, y, mus_tilde = None, th_stop = 1e-5):
        y_tilde = np.sqrt(R) *y
        H_yy_lr = np.zeros((3*problem.nc_,3*problem.nc_))
        mus_tilde = self.compute_mus_tilde(problem, R, mus_tilde)
        for j in range(problem.nc_):
            mu_tildej = mus_tilde[j]
            stiction = IceCreamCone.isInside(mu_tildej, y_tilde[3*j:3*j+3], th_stop)
            if stiction:
                H_yy_lr[3*j:3*j+3, 3*j:3*j+3] = np.diag(R[3*j:3*j+3])
            else:
                # TODO not the right condition !!!
                y_r = np.linalg.norm(y[3*j:3*j+2])
                y_r_tilde = np.linalg.norm(y_tilde[3*j:3*j+2])
                muj = problem.contact_constraints_[j].mu_
                mu_hatj = muj * R[3*j]/R[3*j+2]
                breaking = y_tilde[3*j+2] < -mu_tildej*y_r_tilde
                if not breaking:
                    # TODO should correct formula when Rtx != Rty
                    t_dir = y[3*j:3*j+2] /np.linalg.norm(y[3*j:3*j+2])
                    sy = mu_hatj * y_r+ y[3*j+2]
                    P_t = t_dir[:,None] @ t_dir[None,:]
                    P_t_perp = np.eye(2) - P_t
                    H_yy_lr[3*j:3*j+2, 3*j:3*j+2] = mu_hatj* P_t
                    H_yy_lr[3*j:3*j+2, 3*j:3*j+2] += sy *  P_t_perp / y_r
                    H_yy_lr[3*j:3*j+2, 3*j:3*j+2] *= mu_hatj
                    H_yy_lr[3*j:3*j+2, 3*j+2] = mu_hatj*t_dir
                    H_yy_lr[ 3*j+2, 3*j:3*j+2] = mu_hatj*t_dir
                    H_yy_lr[3*j+2, 3*j+2] = 1
                    H_yy_lr[3*j:3*j+3, 3*j:3*j+3] *= R[3*j+2]/(1+mu_tildej**2)
        return H_yy_lr

    def gradv_cost_regularization(self, problem, R, dq, y= None, lam = None):
        lam = self.dqtolam_mapping(problem, R, dq, y, lam)
        return - problem.J_.T @lam

    def hessv_cost_regularization(self, problem, R, dq, y = None, lam = None):
        y = self.dqtoy_mapping(problem, R, dq,y)
        H_yy_lr = self.hessy_cost_regularization(problem, R, y)
        return problem.J_.T @ np.diag(1./R) @ H_yy_lr @ np.diag(1./R) @ problem.J_

    def grad_cost_unconstrained(self, problem, R, dq, y = None, lam = None):
        grad = problem.M_ @ (dq - problem.dqf_)
        grad += self.gradv_cost_regularization(problem, R, dq, y, lam)
        return grad

    def hess_cost_unconstrained(self, problem, R, dq, y = None, lam = None, th_stop = 1e-5):
        H = problem.M_.copy()
        H += self.hessv_cost_regularization(problem, R, dq, y, lam)
        return H

    def computeR(self, problem):
        R = np.zeros(3*problem.nc_)
        for j in range(problem.nc_):
            #  TODO
            R[3*j+2] = np.max([np.linalg.norm(problem.Del_.G_[3*j:3*j+3,3*j:3*j+3])/(4*np.pi*np.pi) , 0.])
            R[3*j] = 1e-3 * R[3*j+2]
            R[3*j+1] = R[3*j]
        return R


    def solve(
        self,
        problem: ContactProblem,
        lam0,
        settings:ContactSolverSettings,
        rho=1e-6,
        over_relax=1.,
        eps_reg = 1e-6,
        debug = False,
        **kwargs
    ):
        nc = problem.nc_
        problem.Del_.computeChol(1e-9)
        problem.Del_.evaluateDel()
        # computing compliance relaxation R
        R = self.computeR(problem)
        Minv = np.linalg.inv(problem.M_)
        self.dq_ = problem.dqf_ + Minv @ problem.J_.T @ lam0[:,0]
        self.mus_tilde = self.compute_mus_tilde(problem, R)
        self.mus_hat = [problem.contact_constraints_[j].mu_ * R[3*j]/R[3*j+2] for j in range(nc)]
        self.prob_tilde = ContactProblem(problem.Del_.G_, problem.g_, self.mus_tilde)

        def cost_unconstrained(v):
            cost = self.cost_unconstrained(problem, R, v)
            return cost
        cost = cost_unconstrained(self.dq_)
        for i in range(settings.max_iter_):
            self.v_ = problem.J_ @self.dq_
            self.y_ = self.compliance_mapping(problem, R, self.v_)
            self.lam_ = self.projK_R(problem, R, self.y_)
            grad = self.grad_cost_unconstrained(problem, R, self.dq_, self.y_, self.lam_)
            H = self.hess_cost_unconstrained(problem, R, self.dq_, self.y_, self.lam_)
            Hinv = np.linalg.inv(H)
            ddq = - Hinv @ grad
            #  linesearch
            alpha = 1.25
            exp_dec = 1e-4*alpha*grad.dot(ddq)
            for k in range(settings.max_iter_):
                dq_try = self.dq_ + alpha *ddq
                cost_try = cost_unconstrained(dq_try)
                if cost_try < cost + exp_dec:
                    break
                else:
                    alpha *= .8
                    exp_dec *= .8
            self.dq_ = dq_try.copy()
            cost = cost_try
            self.niter_ = i+1
            self.stop_ = np.max(np.abs(grad/np.sqrt(problem.M_.diagonal())))
            if self.stop_<settings.th_stop_:
                return True
        return False

class PyCCPADMMSolver2(PyDualContactSolver):

    def updateRho(self, prim_feas, dual_feas):
        if prim_feas > 10.0*dual_feas:
            self.cpt +=1
            if self.cpt %5 ==0:
                if self.rho_up<0:
                    self.tau_inc = 0.1*1.+0.9*self.tau_inc
                    self.tau_dec = 0.1*1.+0.9*self.tau_dec
                self.rho *= self.tau_inc
                self.rho_up = 1
                return True
            return False
        elif prim_feas < 0.1*dual_feas:
            self.cpt +=1
            if self.cpt %5 == 0:
                if self.rho_up >0:
                    self.tau_inc = 0.1*1.+0.9*self.tau_inc
                    self.tau_dec = 0.1*1.+0.9*self.tau_dec
                self.rho /= self.tau_dec
                self.rho_up = -1
                return True
            return False
        else:
            return False

    def computeLargestEigenValue(self, G, th_stop=1e-3, max_iter=10):
        v = np.random.random(G.shape[0])
        v /= np.linalg.norm(v)
        for i in range(max_iter):
            v_pred = v
            v = G @ v
            v /= np.linalg.norm(v)
            if np.linalg.norm(v_pred-v, np.inf)<th_stop:
                break
        eigval_max = v.T @ G @ v
        return eigval_max

    def stoppingCriteria(self, problem, x, z, y, z_pred, rho):
        self.prim_feas = np.linalg.norm(x - z, ord=np.inf)
        self.dual_feas = np.linalg.norm(rho*(z-z_pred), ord=np.inf)
        self.comp = problem.computeConicComplementarity(x, y)
        return np.max([self.prim_feas, self.dual_feas, self.comp])

    def relativeStoppingCriteria(self, z, z_pred):
        norm_pred = np.linalg.norm(z_pred)
        if norm_pred > 0:
            crit = np.linalg.norm(z - z_pred) / norm_pred
        else:
            crit = np.inf
        return crit

    def solve(
        self,
        problem: ContactProblem,
        lam0,
        settings:ContactSolverSettings,
        rho=1e-6,
        over_relax=1.,
        eps_reg = 0.,
        **kwargs
    ):
        self.resetCallbacks()
        problem.Del_.evaluateDel()
        x = lam0.copy()
        z = x.copy()
        y = np.zeros_like(x)
        self.eigval_max = self.computeLargestEigenValue(problem.Del_.G_)
        self.eigval_min = rho
        self.rho = np.sqrt(self.eigval_max * self.eigval_min) * (
            (self.eigval_max / self.eigval_min) ** 0.4
        )
        self.tau_dec = (self.eigval_max / self.eigval_min) ** 0.1
        self.tau_inc = self.tau_dec
        Gtild = problem.Del_.G_ + (self.rho + rho) * np.eye(3 * self.nc_)
        Ginv = np.linalg.inv(Gtild)
        self.rho_up, self.cpt = 0,0
        for j in range(settings.max_iter_):
            x_pred = x.copy()
            z_pred = z.copy()
            x = -Ginv @ (problem.g_[:,None] + y - self.rho*z - rho * x_pred)
            problem.project(x + y/self.rho, z)
            y += self.rho*(x-z)
            self.rel_stop_ = self.relativeStoppingCriteria(x, x_pred)
            self.stop_ = self.stoppingCriteria(problem, x, z, y, z_pred,self.rho)
            self.call(
                G=problem.Del_.G_,
                g=problem.g_,
                lam=z,
                mu=[cons.mu_ for cons in problem.contact_constraints_],
                prim_feas=self.prim_feas,
                dual_feas=self.dual_feas,
                comp=self.comp,
            )
            if self.rel_stop_ < settings.rel_th_stop_ or self.stop_ < settings.th_stop_:
                self.lam_ = z
                self.lam_2_ = x
                self.niter_ = j
                return True
        self.lam_ = z
        self.lam_2_ = x
        self.niter_ = settings.max_iter_
        return False


class PyRaisimSolver(PyDualContactSolver):
    def __init__(self) -> None:
        PyDualContactSolver.__init__(self)
        self.beta1 = 1e-2
        self.beta2 = 0.5
        self.beta3 = 1.3
        self.alpha = 1.0
        self.alpha_min = 0.7
        self.gamma = 0.99
        self.thresh = 1e-3

    def setProblem(self, problem):
        PyDualContactSolver.setProblem(self, problem)

    def computeGinv(self, G):
        self.Ginv = np.zeros((self.nc_, 3, 3))
        for i in range(self.nc_):
            self.Ginv[i] = np.linalg.inv(G[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)])

    def computeGlam(self, G, lam):
        self.Glam = np.zeros((self.nc_, self.nc_, 3, 1))
        for i in range(self.nc_):
            for k in range(self.nc_):
                self.Glam[i, k] = (
                    G[3 * i : 3 * (i + 1), 3 * k : 3 * (k + 1)]
                    @ lam[3 * k : 3 * (k + 1)]
                )

    def computeC(self, G, g, lam):
        self.cs = np.zeros((self.nc_, 3, 1))
        self.v = np.zeros_like(lam)
        self.computeGlam(G, lam)
        for i in range(self.nc_):
            mask = np.ones(self.Glam[i].shape, dtype=bool)
            mask[i, :] = False
            self.cs[i] = np.sum(self.Glam[i][mask], axis=0)
            self.cs[i] += g[3 * i : 3 * (i + 1)]
            self.v[3 * i : 3 * (i + 1)] = self.cs[i] + self.Glam[i, i]

    def updateGlam(self, i, G, lami):
        for j in range(self.nc_):
            self.Glam[j, i] = G[3 * j : 3 * (j + 1), 3 * i : 3 * (i + 1)] @ lami

    def updateC(self, i, G, g, lami):
        self.updateGlam(i, G, lami)
        for j in range(self.nc_):
            mask = np.ones(self.Glam[j].shape, dtype=bool)
            mask[j, :] = False
            self.cs[j] = np.sum(self.Glam[j][mask], axis=0)
            self.cs[j] += g[3 * j : 3 * (j + 1)]
            self.v[3 * i : 3 * (i + 1)] = self.cs[j] + self.Glam[j, j]

    def computeLamv0(self, Ginv, c):
        lam_v0 = -Ginv @ c
        return lam_v0

    def computeTheta(self, lam):
        theta = np.arctan2(lam[1], lam[0])
        return theta

    def computeLam(self, r, theta, lamZ):
        lam = np.array([r * np.cos(theta), r * np.sin(theta), lamZ])
        return lam

    def initTheta(self, Ginv, c):
        lam_v0 = self.computeLamv0(Ginv, c)
        theta0 = self.computeTheta(lam_v0)
        return theta0

    def initLam(self, A, Ginv, c, mu):
        theta0 = self.initTheta(Ginv, c)
        r0 = self.computeR(A, c, mu, theta0)
        lamZ0 = self.computeLamZ(mu, r0)
        lam = self.computeLam(r0, theta0, lamZ0)
        return lam

    def computeR(self, G, c, mu, theta):
        r = -c[2] / (G[2, 2] / mu + G[2, 0] * np.cos(theta) + G[2, 1] * np.sin(theta))
        return r

    def computeLamZ(self, mu, r):
        lamZ = r / mu
        return lamZ

    def computeH1Grad(self, A):
        h1_grad = np.expand_dims(A[2], axis=-1)
        return h1_grad

    def computeH2Grad(self, mu, lam):
        h2_grad = np.array([2 * lam[0], 2 * lam[1], -2 * (mu**2) * lam[2]])
        return h2_grad

    def computeEta(self, A, mu, lam):
        h1_grad = self.computeH1Grad(A)
        h2_grad = self.computeH2Grad(mu, lam)
        eta = np.cross(h1_grad[:, 0], h2_grad[:, 0])
        return eta

    def computeGradTheta(self, A, c, mu, lam):
        eta = self.computeEta(A, mu, lam)
        Abar = A[:, :2] - np.expand_dims(A[:, 2], axis=-1) @ np.array(
            [[A[2, 0] / A[2, 2], A[2, 1] / A[2, 2]]]
        )
        cbar = c - np.expand_dims(A[:, 2], axis=-1) * (c[2] / A[2, 2])
        grad_theta = (Abar @ lam[:2] + cbar).T @ eta
        return grad_theta

    def bisectionStep(self, G, Ginv, c, mu, maxIter=100):
        theta = self.initTheta(Ginv, c)
        r = self.computeR(G, c, mu, theta)
        lamZ = self.computeLamZ(mu, r)
        lam = self.computeLam(r, theta, lamZ)
        d0 = self.computeGradTheta(G, c, mu, lam)
        dtheta = -self.beta1 * np.sign(d0)
        lam_v0 = self.computeLamv0(Ginv, c)
        for i in range(maxIter):  # initial stepping
            theta_pred = theta.copy()
            lam_pred = lam.copy()
            theta += dtheta
            r = self.computeR(G, c, mu, theta)
            lamZ = self.computeLamZ(mu, r)
            lam = self.computeLam(r, theta, lamZ)
            h2_grad = self.computeH2Grad(mu, lam)
            if h2_grad.T @ (lam - lam_v0) < 0 or r < 0:
                dtheta = self.beta2 * dtheta
                theta = theta_pred.copy()
            else:
                grad = self.computeGradTheta(G, c, mu, lam)
                if grad.T @ d0 > 0:
                    dtheta = self.beta3 * dtheta
                else:
                    break
        for i in range(maxIter):  # bisection
            theta_bis = 0.5 * (theta + theta_pred)
            r_bis = self.computeR(G, c, mu, theta_bis)
            lamZ_bis = self.computeLamZ(mu, r_bis)
            lam_bis = self.computeLam(r_bis, theta_bis, lamZ_bis)
            grad_bis = self.computeGradTheta(G, c, mu, lam_bis)
            if grad_bis * d0 > 0:
                theta_pred = theta_bis
                lam_pred = lam_bis.copy()
            else:
                theta = theta_bis
                lam = lam_bis.copy()
            if np.linalg.norm(lam - lam_pred, ord=np.inf) < self.thresh:
                return lam
        return lam

    def stoppingCriteria(self, problem: ContactProblem, x, v):
        self.prim_feas = 0.0
        self.dual_feas = 0.0
        self.comp = problem.computeContactComplementarity(x, v)
        return self.comp

    def relativeStoppingCriteria(self, x, x_pred):
        norm_pred = np.linalg.norm(x_pred)
        if norm_pred > 0:
            crit = np.linalg.norm(x - x_pred) / norm_pred
        else:
            crit = np.inf
        return crit

    def solve(
        self,
        problem: ContactProblem,
        lam0,
        settings:ContactSolverSettings,
        **kwargs
    ):
        self.resetCallbacks()
        problem.Del_.evaluateDel()
        lam = lam0.copy()
        lam_pred = lam0.copy()
        self.computeGinv(problem.Del_.G_)
        self.computeC(problem.Del_.G_, problem.g_, lam)
        for i in range(settings.max_iter_):
            lam_pred = lam.copy()
            for j in range(self.nc_):
                Aj = problem.Del_.G_[3 * j : 3 * (j + 1), 3 * j : 3 * (j + 1)]  # Ajj
                Ajinv = self.Ginv[j]
                cj = self.cs[j]
                lamj_v0 = self.computeLamv0(Ajinv, cj)
                muj = problem.contact_constraints_[j].mu
                if cj[2] > 0:  # opening contact
                    lam[3 * j : 3 * (j + 1)] *= 1 - self.alpha

                elif lamj_v0[2] * muj > np.linalg.norm(lamj_v0[:2]):  # sticking
                    lam[3 * j : 3 * (j + 1)] = (
                        self.alpha * lamj_v0
                        + (1 - self.alpha) * lam[3 * j : 3 * (j + 1)]
                    )
                else:  # slipping
                    lamstarj = self.bisectionStep(Aj, Ajinv, cj, muj)
                    lam[3 * j : 3 * (j + 1)] = (
                        self.alpha * lamstarj
                        + (1 - self.alpha) * lam[3 * j : 3 * (j + 1)]
                    )
                self.updateC(j, problem.Del_.G_, problem.g_, lam[3 * j : 3 * (j + 1)])
            self.rel_stop_ = self.relativeStoppingCriteria(lam, lam_pred)
            self.stop_ = self.stoppingCriteria(problem, lam, self.v)
            self.call(
                G=problem.Del_.G_,
                g=problem.g_,
                lam=lam,
                mu=[cons.mu_ for cons in problem.contact_constraints_],
                prim_feas=self.prim_feas,
                dual_feas=self.dual_feas,
                comp=self.comp,
            )
            if self.rel_stop_ < settings.rel_th_stop_ or self.stop_ < settings.th_stop_:
                self.lam_ = lam
                self.niter_ = i +1
                return True
            self.alpha = self.alpha_min + self.gamma * (self.alpha - self.alpha_min)
        self.lam_ = lam
        self.niter_ = settings.max_iter_
        return False
