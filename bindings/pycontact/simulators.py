import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt

from pycontact.solvers import (
    PyNCPPGSSolver,
    PinNCPPGSSolver,
    PinNCPADMMSolver,
    PyCCPPGSSolver,
    PyCCPCVXSolver,
    PyRaisimSolver,
    PyContactSolver,
    # PyLCPMehrotraSolver,
    # PyLCPIPSolver,
    PyCCPADMMPrimalSolver,
    PyCCPNSNewtonPrimalSolver,
    PyCCPSCSSolver
)
from pycontact import (
    CCPADMMSolver,
    CCPPGSSolver,
    CCPADMMPrimalSolver,
    CCPNewtonPrimalSolver,
    NCPPGSSolver,
    NCPStagProjSolver,
    RaisimSolver,
    RaisimCorrectedSolver,
    LCPPGSSolver,
    LCPQPSolver,
    LCPStagProjSolver,
    ContactProblem,
    LinearContactProblem,
    ContactSolverSettings
)
from pycontact.utils.callbacks import LoggerCallback

from pycontact.utils.pin_utils import (
    computeContacts,
    computeAllContacts
)

import abc
import timeit
import time

class SimulatorBase(abc.ABC):
    def __init__(self, warm_start = True, timings = False, statistics = False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        self.callback_figs = None
        self.callback_axs = None
        self.prev_contacts = {}
        self.prev_Rs = {}
        self.warm_start = warm_start
        self.timings = timings
        self.statistics = statistics
        self.derivatives_data = None
        self.magic_forces = magic_forces
        self.max_dist_mf = max_dist_mf

    def setSimulation(self, model, data, geom_model, geom_data):
        for req in geom_data.collisionRequests:
                req.security_margin = 1e-3
        if self.magic_forces:
            for req in geom_data.collisionRequests:
                req.security_margin += self.max_dist_mf

    def preprocessContactProblem(self, G, active_cols, pen_err):
        return G

    @abc.abstractclassmethod
    def create_contact_problem(self):
        pass

    def step(
        self,
        model,
        data,
        geom_model,
        geom_data,
        q,
        v,
        tau,
        fext,
        dt,
        Kb=1e-4,
        maxIter=100,
        th=1e-6,
        **kwargs
    ):
        # detect collisions and compute Jacobians
        pin.updateGeometryPlacements(model, data, geom_model, geom_data, q)
        M = pin.crba(model, data, q)
        pin.computeCollisions(geom_model, geom_data, False)
        self.J, self.Del, self.M, self.R, self.signed_dist, self.mus, self.comps, self.els, self.active_cols, self.contact_points, self.contact_joints, self.contact_placements = computeContacts(
            model, data, geom_model, geom_data, self.prev_contacts, self.prev_Rs
        )
        v_new = self.computeNewVelocity(model, data, q, v, tau, fext, dt, Kb, maxIter, th, **kwargs)
        # update state
        q_new = pin.integrate(model, q, v_new * dt)
        return q_new, v_new

    def computeDFreeVelocity(self, model, data, q, v, tau, fext, dt):
        return dt * pin.aba(model, data, q, v, tau, fext)

    def computeConstraintCor(self, err, Kb, v, dt):
        constraint_cor = (1./dt) * err
        for i in range(self.nc):
            if constraint_cor[3*i+2] <0 :
                constraint_cor[3*i+2] *= Kb
            if not self.active_cols[i] in self.prev_contacts.keys():
                constraint_cor[3*i+2] += self.els[i]*(self.J[3*i+2] @ v)
        return constraint_cor

    def computeFreeContactPen(self, v_l, constraint_cor):
        g = (self.J @ v_l + constraint_cor)[:,None]
        return g

    @abc.abstractclassmethod
    def computeNewVelocity(self):
        pass

    def solveContactProblem(self, problem, maxIter, th, **kwargs):
        self.solver.setProblem(problem)
        self.lam0 = self.initGuess(self.active_cols)
        contact_settings = ContactSolverSettings()
        contact_settings.max_iter_ = maxIter
        contact_settings.th_stop_ = th
        contact_settings.timings_ = self.timings
        contact_settings.statistics_ = self.statistics
        if "rel_th_stop" in kwargs:
            contact_settings.rel_th_stop_ = kwargs.pop("rel_th_stop")
        solved = self.solver.solve(
            problem =problem, lam0 = self.lam0, settings = contact_settings, **kwargs
        )
        if self.timings:
            N = 10
            timings_cpp_ = np.zeros(N)
            contact_settings.statistics = False
            for i in range(N):
                self.solver.solve(problem, self.lam0, contact_settings, **kwargs)
                timings_cpp_[i] = self.solver.getCPUTimes().user
            timings_cpp_ = np.sort(timings_cpp_)
            self.solver.timings_cpp = np.mean(timings_cpp_[:int(max(1, N*0.8))])
            self.solve_time = self.solver.timings_cpp
        return solved

    def cache_prev_contacts(self):
        self.prev_active_cols = self.active_cols
        for i,j in enumerate(self.active_cols):
            self.prev_contacts[j] = (self.lam[3*i:3*(i+1),:],)
            self.prev_Rs[j] = self.R[:,3*i:3*(i+1)]

    def initGuess(self,active_cols):
        nc = len(active_cols)
        lam0 = np.zeros((3 * nc, 1))
        if self.warm_start:
            for i,j in enumerate(active_cols):
                if j in self.prev_contacts.keys():
                    lam0[3*i:3*(i+1),:] = self.prev_contacts[j][0]
        return lam0

    @abc.abstractclassmethod
    def getSolverResults(self):
        pass

    def resetCallbacks(self):
        self.solver.resetStats()

    def plotCallback(self, block=True):
        if self.callback_figs is None:
            figs, axs = plt.subplots(2, 2, figsize=(20, 15))
            showing = True
            self.callback_figs = figs
        else:
            axs = self.callback_axs
            for i in range(len(axs)):
                for j in range(len(axs[i])):
                    axs[i][j].clear()
            showing = False
        iters = [t for t in range(len(self.solver.stats_.ncp_comp_))]
        axs[0][0].semilogy(iters, self.solver.stats_.ncp_comp_, marker="+")
        axs[0][0].set_xlabel("iterations")
        axs[0][0].set_ylabel("Contact complementarity")

        iters = [t for t in range(len(self.solver.stats_.comp_))]
        axs[0][1].semilogy(iters, self.solver.stats_.comp_, marker="+")
        axs[0][1].set_xlabel("iterations")
        axs[0][1].set_ylabel("Problem complementarity")

        iters = [t + 1 for t in range(len(self.solver.stats_.prim_feas_))]
        axs[1][0].semilogy(iters, self.solver.stats_.prim_feas_, marker="+")
        axs[1][0].set_xlabel("iterations")
        axs[1][0].set_ylabel("Primal feasibility")

        iters = [t + 1 for t in range(len(self.solver.stats_.dual_feas_))]
        axs[1][1].semilogy(iters, self.solver.stats_.dual_feas_, marker="+")
        axs[1][1].set_xlabel("iterations")
        axs[1][1].set_ylabel("Dual feasibility")

        plt.draw()
        if showing:
            plt.tight_layout()
            plt.show(block=block)
        self.callback_axs = axs

    def close_plot_callback(self):
        plt.close(self.callback_figs)

class PrimalSolverSimulator(SimulatorBase):

    def create_contact_problem(self, Del, g,  M, J, dqf, vstar, mus, comp):
        return ContactProblem(Del, g, M, J, dqf, vstar, mus, comp)

    def getSolverResults(self):
        self.ddq_new = self.solver.getSolution()[:,None].copy()
        self.lam = self.solver.getDualSolution()[:,None].copy()

    def computeNewVelocity(self, model, data, q, v, tau, fext, dt, Kb, maxIter, th, **kwargs):
        # compute free velocity
        dv_l = self.computeDFreeVelocity(model, data, q, v, tau,fext, dt)
        if self.J is not None:
            self.nc = int(len(self.signed_dist) / 3)
            # call contact solver
            self.dqf = v + dv_l
            self.vstar = - self.computeConstraintCor(self.signed_dist, Kb, v, dt)
            self.g = self.computeFreeContactPen(self.dqf, -self.vstar)
            self.R_comp_ = np.zeros(3*self.nc)
            for i in range(self.nc):
                self.R_comp_[3*i+2] = self.comps[i]
            problem = self.create_contact_problem(self.Del, self.g/dt, self.M, self.J, self.dqf/dt, self.vstar/dt, self.mus, self.R_comp_)
            solved = self.solveContactProblem(problem, maxIter, th, **kwargs)
            self.getSolverResults()
            dq_new = self.ddq_new[:,0]*dt
            self.cache_prev_contacts()
        else:
            dq_new = v+dv_l
            self.vstar = []
            self.lam = []
            self.g = []
            if self.timings:
                self.solve_time = np.nan
        return dq_new

class DualSolverSimulator(SimulatorBase):

    def create_contact_problem(self, G, g, mus, comp):
        return ContactProblem(G,g,mus, comp)

    def getSolverResults(self):
        self.lam = self.solver.getSolution()[:,None].copy()
        self.vc = self.solver.getDualSolution()[:,None].copy()

    def cache_prev_contacts(self):
        self.prev_active_cols = self.active_cols
        for i,j in enumerate(self.active_cols):
            self.prev_contacts[j] = (self.lam[3*i:3*(i+1),:],)
            self.prev_Rs[j] = self.R[:,3*i:3*(i+1)]

    def computeNewVelocity(self, model, data, q, v, tau, fext, dt, Kb, maxIter, th, **kwargs):
        # compute free velocity
        dv_l = self.computeDFreeVelocity(model, data, q, v, tau, fext, dt)
        if self.J is not None:
            self.nc = int(len(self.signed_dist) / 3)
            # call contact solver
            self.dqf = v + dv_l
            self.vstar = - self.computeConstraintCor(self.signed_dist, Kb, v, dt)
            self.g = self.computeFreeContactPen(self.dqf, -self.vstar)
            self.R_comp_ = np.zeros(3*self.nc)
            for i in range(self.nc):
                self.R_comp_[3*i+2] = self.comps[i]
            problem = self.create_contact_problem(self.Del, self.g/dt, self.mus, self.R_comp_)
            solved = self.solveContactProblem(problem, maxIter, th, **kwargs)
            self.getSolverResults()
            fext_tot = [fexti.copy() for fexti in fext]
            for i in range(len(self.contact_joints)):
                fext_tot[self.contact_joints[i][0]] += -pin.Force(self.contact_placements[i][0].actionInverse.T[:,:3] @ self.lam[3*i:3*(i+1)])
                fext_tot[self.contact_joints[i][1]] += pin.Force(self.contact_placements[i][1].actionInverse.T[:,:3] @ self.lam[3*i:3*(i+1)])
            dv = dt * pin.aba(
                model, data, q, v, tau, fext_tot
            )
            self.cache_prev_contacts()
        else:
            problem = None
            dv = dv_l.copy()
            self.dqf = None
            self.lam = []
            self.g = []
            fext_tot = [fexti.copy() for fexti in fext]
            if self.timings:
                self.solve_time = np.nan
        v_new = v + dv
        return v_new


class CCPSimulator(SimulatorBase):
    def __init__(self, model=None, geom_model=None, data=None, geom_data=None, q0=None, regularize = False, warm_start = True, timings = False, statistics = False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        SimulatorBase.__init__(self, warm_start, timings, statistics, magic_forces, max_dist_mf)
        if regularize:
            self.preComputeApproxA(model, geom_model, data, geom_data, q0)
            self.regularize = True
        else:
            self.regularize =False

    def preComputeApproxA(self, model, geom_model, data, geom_data, q0):
        self.regularize =True
        pin.updateGeometryPlacements(model, data, geom_model, geom_data, q0)
        M = pin.crba(model, data, q0)
        Jc, G, R, signed_dist, mus, els, contact_points = computeAllContacts(
            model, data, geom_model, geom_data
        )
        self.approxA = G
        return

    def computeImpedance(self, pen_err, dmin: float = 0.9, dmax: float =  0.99999, width : float = 0.001, midpoint: float =  0.5, power: float = 2):
        d= np.ones_like(pen_err)*dmax
        return d

    def computeRMatrix(self, d, approxA):
        R = np.diag((1-d)/d * approxA)
        return R

    def preprocessContactProblem(self, G, active_cols, pen_err):
        if self.regularize:
            nc = len(active_cols)
            if nc>0:
                d = self.computeImpedance(pen_err)
                idxs = []
                for i in active_cols:
                    idxs +=[3*i,3*i+1,3*i+2]
                approxA = self.approxA[idxs, idxs]
                R = self.computeRMatrix(d, approxA)
                G += R
        return G


class CCPADMMSimulator(CCPSimulator, DualSolverSimulator):
    def __init__(self, model= None, geom_model= None, data= None, geom_data= None, q0= None, regularize = False, warm_start = True, timings = False, statistics = False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        CCPSimulator.__init__(self, model, geom_model, data, geom_data, q0, regularize, warm_start, timings, statistics, magic_forces, max_dist_mf)
        self.solver = CCPADMMSolver()


    def cache_prev_contacts(self):
        self.prev_active_cols = self.active_cols
        for i,j in enumerate(self.active_cols):
            self.prev_contacts[j] = (self.lam[3*i:3*(i+1),:],self.gamma[3*i:3*(i+1),:])
            self.prev_Rs[j] = self.R[:,3*i:3*(i+1)]
        self.cache_prev_prox_params()

    def cache_prev_prox_params(self):
        self.prev_rho = self.rho
        self.prev_eigval_max = self.eigval_max

    def getPrevR(self):
        return self.prev_contacts[:][2]

    def initGuessGamma(self,active_cols):
        nc = len(active_cols)
        gamma0 = np.zeros((3 * nc, 1))
        if self.warm_start:
            for i,j in enumerate(active_cols):
                if j in self.prev_contacts.keys():
                    gamma0[3*i:3*(i+1),:] = self.prev_contacts[j][1]
        return gamma0

    def initGuessProxParams(self,active_cols):
        if self.warm_start:
            same_contacts = True
            for i,j in enumerate(active_cols):
                if j not in self.prev_contacts.keys():
                    same_contacts = False
                    break
            if same_contacts:
                return self.prev_rho, self.prev_eigval_max
            else:
                rho, eigval_max = None, None
        else:
            rho, eigval_max = None, None
        rho, eigval_max = None, None
        return rho, eigval_max

    def solveContactProblem(self, problem, maxIter, th, **kwargs):
        self.solver.setProblem(problem)
        contact_settings = ContactSolverSettings()
        contact_settings.max_iter_ = maxIter
        contact_settings.th_stop_ = th
        contact_settings.timings_ = self.timings
        contact_settings.statistics_ = self.statistics
        if "rel_th_stop" in kwargs:
            contact_settings.rel_th_stop_ = kwargs.pop("rel_th_stop")
        if self.warm_start: # init rho with previous value
            self.lam0 = self.initGuess(self.active_cols)
            self.gamma0 = self.initGuessGamma(self.active_cols)
            rho, eigval_max = self.initGuessProxParams(self.active_cols)
            if rho is not None:
                solved = self.solver.solve(
                    problem = problem, lam0 = self.lam0, gamma0 = self.gamma0, settings = contact_settings, rho_admm =rho, max_eigval=eigval_max, **kwargs
                )
            else:
                solved = self.solver.solve(
                    problem = problem, lam0 = self.lam0, gamma0 = self.gamma0, settings = contact_settings, **kwargs
                )
        else:
            self.lam0 = self.initGuess(self.active_cols)
            solved = self.solver.solve(
                problem = problem, lam0 = self.lam0, settings = contact_settings, **kwargs
            )

        if self.timings:
            N = 10
            contact_settings.statistics_ = False
            if self.warm_start:
                if rho is not None:
                    timings_cpp_ = np.zeros(N)
                    for i in range(N):
                        self.solver.solve(problem = problem, lam0 = self.lam0, gamma0 = self.gamma0, settings = contact_settings, rho_admm = rho,
                                          max_eigval = eigval_max, **kwargs)
                        timings_cpp_[i] = self.solver.getCPUTimes().user

                else:
                    timings_cpp_ = np.zeros(N)
                    for i in range(N):
                        self.solver.solve(problem = problem, lam0 = self.lam0, gamma0 = self.gamma0,
                                          settings = contact_settings, **kwargs)
                        timings_cpp_[i] = self.solver.getCPUTimes().user
            else:
                timings_cpp_ = np.zeros(N)
                for i in range(N):
                    self.solver.solve(problem = problem, lam0 = self.lam0, settings = contact_settings, **kwargs)
                    timings_cpp_[i] = self.solver.getCPUTimes().user
            timings_cpp_ = np.sort(timings_cpp_)
            self.solver.timings_cpp = np.mean(timings_cpp_[:int(max(1, N*0.8))])
            self.solve_time = self.solver.timings_cpp
        return solved

    def getSolverResults(self):
        self.lam = self.solver.getSolution()[:,None].copy()
        self.gamma = self.solver.getDualSolution()[:,None].copy()
        self.rho = self.solver.rho_
        self.eigval_max = self.solver.eigval_max_

class PyCCPADMMPrimalSimulator(CCPSimulator, PrimalSolverSimulator):

    def __init__(self, model=None, geom_model=None, data=None, geom_data=None, q0=None, regularize=False, warm_start=True, timings=False, statistics=False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        CCPSimulator.__init__(self, model, geom_model, data, geom_data, q0, regularize, warm_start, timings, statistics, magic_forces, max_dist_mf)
        self.solver = PyCCPADMMPrimalSolver()

class CCPADMMPrimalSimulator(CCPSimulator, PrimalSolverSimulator):

    def __init__(self, model=None, geom_model=None, data=None, geom_data=None, q0=None, regularize=False, warm_start=True, timings=False, statistics=False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        CCPSimulator.__init__(self, model, geom_model, data, geom_data, q0, regularize, warm_start, timings, statistics, magic_forces, max_dist_mf)
        self.solver = CCPADMMPrimalSolver()

class CCPNewtonPrimalSimulator(CCPSimulator, PrimalSolverSimulator):

    def __init__(self, model=None, geom_model=None, data=None, geom_data=None, q0=None, regularize=False, warm_start=True, timings=False, statistics=False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        CCPSimulator.__init__(self, model, geom_model, data, geom_data, q0, regularize, warm_start, timings, statistics, magic_forces, max_dist_mf)
        self.solver = CCPNewtonPrimalSolver()

class PyCCPNSNewtonPrimalSimulator(CCPSimulator, PrimalSolverSimulator):

    def __init__(self, model=None, geom_model=None, data=None, geom_data=None, q0=None, regularize=False, warm_start=True, timings=False, statistics=False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        CCPSimulator.__init__(self, model, geom_model, data, geom_data, q0, regularize, warm_start, timings, statistics, magic_forces, max_dist_mf)
        self.solver = PyCCPNSNewtonPrimalSolver()

class CCPPGSSimulator(CCPSimulator, DualSolverSimulator):
    def __init__(self, model= None, geom_model= None, data= None, geom_data= None, q0= None, regularize = False, warm_start = True, timings = False, statistics = False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        CCPSimulator.__init__(self, model, geom_model, data, geom_data, q0, regularize, warm_start, timings, statistics, magic_forces, max_dist_mf)
        self.solver = CCPPGSSolver()


class PyCCPCVXSimulator(CCPSimulator,DualSolverSimulator):
    """
    Warning: only compliant systems are handled by this simulator.
    """
    def __init__(self, model= None, geom_model= None, data= None, geom_data= None, q0= None, regularize = False, warm_start=True, timings = False, statistics = False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        CCPSimulator.__init__(self, model, geom_model, data, geom_data, q0, regularize, warm_start, timings, statistics, magic_forces, max_dist_mf)
        self.solver= PyCCPCVXSolver()

class PyCCPSCSSimulator(CCPSimulator,DualSolverSimulator):

    def __init__(self, model= None, geom_model= None, data= None, geom_data= None, q0= None, regularize = False, warm_start=True, timings = False, statistics = False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        CCPSimulator.__init__(self, model, geom_model, data, geom_data, q0, regularize, warm_start, timings, statistics, magic_forces, max_dist_mf)
        self.solver= PyCCPSCSSolver()

class PyCCPPGSSimulator(DualSolverSimulator):
    def __init__(self) -> None:
        self.solver: PyContactSolver = PyCCPPGSSolver()
        self.solver.addCallback(LoggerCallback())
        DualSolverSimulator.__init__(self)

class NCPPGSSimulator(DualSolverSimulator):
    def __init__(self, warm_start = True, timings = False, statistics = False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        DualSolverSimulator.__init__(self, warm_start, timings, statistics, magic_forces, max_dist_mf)
        self.solver = NCPPGSSolver()

class PyNCPPGSSimulator(DualSolverSimulator):
    def __init__(self, warm_start = True, timings = False, statistics = False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        self.solver: PyContactSolver = PyNCPPGSSolver()
        self.solver.addCallback(LoggerCallback())
        DualSolverSimulator.__init__(self, warm_start, timings, statistics, magic_forces, max_dist_mf)

class PinNCPPGSSimulator(DualSolverSimulator):
    """
    Pinocchio's implementation of PGS algorithm for the NCP.
    """
    def __init__(self, warm_start = True, timings = False, statistics = False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        self.solver: PyContactSolver = PinNCPPGSSolver()
        DualSolverSimulator.__init__(self, warm_start, timings, statistics, magic_forces, max_dist_mf)

    def getSolverResults(self):
        self.lam = self.solver.lam_.copy()
        self.vc = self.solver.vc_.copy()


class PinNCPADMMSimulator(CCPADMMSimulator):
    """
    Pinocchio's implementation of ADMM algorithm for the NCP.
    """
    def __init__(self, warm_start = True, timings = False, statistics = False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        CCPADMMSimulator.__init__(self, warm_start=warm_start, timings=timings, statistics=statistics, magic_forces=magic_forces, max_dist_mf=max_dist_mf)
        self.solver: PyContactSolver = PinNCPADMMSolver()

    def getSolverResults(self):
        self.lam = self.solver.pin_solver.getPrimalSolution()[:,None].copy()
        self.gamma = self.solver.pin_solver.getDualSolution()[:,None].copy()
        self.rho = None # TODO
        self.eigval_max = None # TODO

class NCPStagProjSimulator(DualSolverSimulator):
    def __init__(self, warm_start = True, timings = False, statistics = False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        DualSolverSimulator.__init__(self, warm_start, timings, statistics, magic_forces, max_dist_mf)
        self.solver = NCPStagProjSolver()

class RaisimSimulator(DualSolverSimulator):
    def __init__(self, warm_start = True, timings = False, statistics = False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        DualSolverSimulator.__init__(self, warm_start, timings, statistics, magic_forces, max_dist_mf)
        self.solver = RaisimSolver()

class PyRaisimSimulator(DualSolverSimulator):
    def __init__(self) -> None:
        self.solver: PyContactSolver = PyRaisimSolver()
        self.solver.addCallback(LoggerCallback())
        DualSolverSimulator.__init__(self)

class RaisimCorrectedSimulator(DualSolverSimulator):
    def __init__(self, warm_start = True, timings = False, statistics = False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        DualSolverSimulator.__init__(self, warm_start, timings, statistics, magic_forces, max_dist_mf)
        self.solver = RaisimCorrectedSolver()

class LCPPGSSimulator(DualSolverSimulator):
    def __init__(self, warm_start = True, timings = False, statistics = False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        DualSolverSimulator.__init__(self, warm_start, timings, statistics, magic_forces, max_dist_mf)
        self.solver = LCPPGSSolver()

    def create_contact_problem(self, G, g, mus, comp):
        return LinearContactProblem(G, g, mus, comp)

class LCPQPSimulator(DualSolverSimulator):
    def __init__(self, warm_start = True, timings = False, statistics = False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        DualSolverSimulator.__init__(self, warm_start, timings, statistics, magic_forces, max_dist_mf)
        self.solver = LCPQPSolver()

    def create_contact_problem(self, G, g, mus, comp):
        return LinearContactProblem(G, g, mus, comp)

class LCPStagProjSimulator(DualSolverSimulator):
    def __init__(self, warm_start = True, timings = False, statistics = False, magic_forces = False, max_dist_mf = 1e-3) -> None:
        DualSolverSimulator.__init__(self, warm_start, timings, statistics, magic_forces, max_dist_mf)
        self.solver = LCPStagProjSolver()

    def create_contact_problem(self, G, g, mus, comp):
        return LinearContactProblem(G, g, mus, comp)
