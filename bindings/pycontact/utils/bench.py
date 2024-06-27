from typing import List
import numpy as np
from pycontact import IceCreamCone


def compute_contact_vel(J: List, v: List):
    Jv = []
    Jv = [None if len(J[t]) == 0 else J[t] @ v[t] for t in range(len(J))]
    return Jv


def compute_contact_torque(J: List, lam: List):
    Jlam = [
        np.zeros(3) if len(J[i]) == 0 else np.dot(J[i].T, lam[i])[:3, 0]
        for i in range(len(J))
    ]
    return Jlam


def compute_signorini(J: List, lam: List, v: List):
    signorini = []
    lamn = []
    vc = compute_contact_vel(J, v)
    for t in range(len(lam)):
        nc = int(len(lam[t]) / 3)
        lamn = np.zeros(nc)
        vn = np.zeros(nc)
        for i in range(nc):
            lamn[i] = lam[t][3 * i + 2]
            vn[i] = vc[t][3 * i + 2]
        if nc > 0:
            Jlamvn = lamn * vn
            signorini += [np.linalg.norm(Jlamvn, ord=np.inf)]
        else:
            signorini += [0.0]
    return signorini


def compute_mdp(J: List, v: List, lam: List, mu: float):
    mdp_gap = []
    vc = compute_contact_vel(J, v)
    for t in range(len(lam)):
        nc = int(len(lam[t]) / 3)
        gaps = np.zeros(nc)
        for i in range(nc):
            gaps[i] = mu * np.abs(lam[t][3 * i + 2]) - np.linalg.norm(
                lam[t][3 * i : 3 * i + 2]
            )
        if nc > 0:
            mdp_gap += [np.linalg.norm(gaps * vc[t], ord=np.inf)]
        else:
            mdp_gap += [0.0]
    return mdp_gap


def phi(v, mu):
    norm_t = np.linalg.norm(v[:2])
    ph = np.expand_dims(np.array([0.0, 0.0, mu * norm_t]), axis=-1)
    return ph


def compute_ncp(G: List, g: List, lam: List, mu: List):
    ncp_crit = []
    for t in range(len(lam)):
        nc = int(len(lam[t]) / 3)
        if nc > 0:
            v_t = G[t] @ lam[t] + g[t]
            dual = v_t.copy()
            for i in range(nc):
                dual[3 * i : 3 * (i + 1)] += phi(v_t[3 * i : 3 * (i + 1)], mu[t][i])

            ncp_crit += [np.abs((dual.T @ lam[t])[0, 0])]
        else:
            ncp_crit += [0.0]
    return ncp_crit


def compute_dual_feas(G: List, g: List, lam: List, mu: List):
    dual_feas = []
    for t in range(len(lam)):
        nc = int(len(lam[t]) / 3)
        if nc > 0:
            v_t = G[t] @ lam[t] + g[t]
            dual = v_t.copy()
            for i in range(nc):
                dual[3 * i : 3 * (i + 1)] += phi(v_t[3 * i : 3 * (i + 1)], mu[t][i])
            dual_proj = dual.copy()
            for i in range(nc):
                dual_cone = IceCreamCone(1.0 / mu[t][i])
                dual_proj[3 * i : 3 * (i + 1)] = dual_cone.project(
                    dual[3 * i : 3 * (i + 1)]
                )
            dual_feas += [np.linalg.norm(dual - dual_proj, ord=np.inf)]
        else:
            dual_feas += [0.0]
    return dual_feas
