import numpy as np
import pinocchio as pin
from pycontact.utils.pin_utils import  computeContacts, computeDFreeVelocity
from pycontact import ContactProblem


def inverse_contact_dynamics(model,
        data,
        geom_model,
        geom_data,
        q,
        v,
        v_ref,
        dt,
        maxIter=100,
        th=1e-6,
        rho = 1e-4,
        ccp = False):
    """Compute the inverse dynamics of a system with contacts"""
    pin.updateGeometryPlacements(model, data, geom_model, geom_data, q)
    M = pin.crba(model, data, q)
    pin.computeCollisions(geom_model, geom_data, False)
    J, Del, M, R, signed_dist, mus, comps, els, active_cols, contact_points, contact_joints, contact_placements = computeContacts(
        model, data, geom_model, geom_data
    )
    nc = len(mus)
    if nc >0:
        R_comp = np.zeros(3*nc)
        for i in range(nc):
            R_comp[3*i+2] = comps[i]
        R_prox = R_comp + rho * np.ones_like(R_comp)
        R_prox_sqrt = np.sqrt(R_prox)
        problem = ContactProblem(Del, np.zeros(3*nc), [mus[i]*R_prox_sqrt[3*i]/R_prox_sqrt[3*i+2] for i in range(nc)], R_prox)
        problem2 = ContactProblem(Del, np.zeros(3*nc), mus, R_comp)
        lam = np.zeros(3*nc)
        lam_proj = lam.copy()
        c_ref = J @ v_ref
        c_ref += signed_dist/dt
        c_cor = np.zeros_like(lam)
        sig_cor = np.zeros_like(lam)
        c_proj = c_cor.copy()
        problem.computeDeSaxceCorrection(c_ref, c_cor)
        n_iter = maxIter
        for i in range(maxIter):
            lam_pred = lam.copy()
            problem.project(-(c_cor - rho * lam_pred)/R_prox_sqrt, lam)
            lam /= R_prox_sqrt
            sig = c_ref + R_comp*lam
            if ccp:
                c_cor = c_ref + R_comp*lam
            else:
                problem.computeDeSaxceCorrection(sig, sig_cor)
                c_cor = sig_cor - R_comp*lam
            problem2.project(lam, lam_proj)
            prim_res = lam - lam_proj
            problem2.projectDual(sig_cor, c_proj)
            dual_res = sig_cor - c_proj
            contact_comp = problem2.computeConicComplementarity(lam, sig_cor)
            stop = np.max([np.linalg.norm(prim_res, ord = np.inf), np.linalg.norm(dual_res, ord = np.inf), contact_comp])
            if stop < th:
                n_iter = i+1
                break
        has_converged = False if n_iter==maxIter else True
        tau_contact = J.T @ lam
    else:
        lam = np.zeros(0)
        has_converged=True
        n_iter = 0
        tau_contact = np.zeros(model.nv)
    acc_ref = (v_ref - v)/dt
    tau = pin.rnea(model, data, q, v, acc_ref) - tau_contact
    return tau, lam/dt
