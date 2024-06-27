from functools import partial

try:
    import jax.numpy as jnp
    from jax import custom_jvp
    jax_available = True
except ImportError:
    jax_available = False

import numpy as np
from pycontact import LCPQPSolver, NCPPGSSolver, CCPADMMSolver, LinearContactProblem, ContactProblem, ContactSolverSettings


if jax_available:
    @partial(custom_jvp, nondiff_argnums=(3,4,5,6))
    def contact_solver_layer(G, g, mus, max_iter, eps, contact_problem_type, contact_solver_type):
        batch_size = g.shape[0]
        lam =jnp.zeros(g.shape, dtype = g.dtype)
        settings = ContactSolverSettings()
        settings.max_iter_ = max_iter
        settings.th_stop_ = eps
        for i in range(batch_size):
            Gi,gi,musi = np.array(G[i,:,:]), np.array(g[i,:,:]), np.array(mus[i,:,0])
            probi = contact_problem_type(Gi, gi, musi.tolist())
            solveri = contact_solver_type()
            solveri.setProblem(probi)
            lam_ws = np.zeros(len(gi))
            solveri.solve(probi, lam_ws, settings)
            lam.at[i,:,0].set(jnp.array(solveri.getSolution().copy()))
        return lam

    @contact_solver_layer.defjvp
    def contact_solver_layer_jvp(max_iter, eps, contact_problem_type, contact_solver_type, primals, tangents):
        G, g, mus  = primals
        batch_size = g.shape[0]
        nc = mus.shape[1]
        G_dot, g_dot, mus_dot = tangents
        nparam = mus_dot.shape[2]
        lam =jnp.zeros(g.shape, dtype = g.dtype)
        lam_dot = jnp.zeros_like(lam)
        settings = ContactSolverSettings()
        settings.max_iter_ = max_iter
        settings.th_stop_ = eps
        for i in range(batch_size):
            Gi,gi,musi = np.array(G[i,:,:]), np.array(g[i,:,:]), np.array(mus[i,:,0])
            probi = contact_problem_type(Gi, gi, musi.tolist())
            solveri = contact_solver_type()
            solveri.setProblem(probi)
            lam_ws = np.zeros(len(gi))
            solveri.solve(probi, lam_ws, settings)
            lam_i  = solveri.getSolution().copy()
            lam.at[i,:,0].set(jnp.array(lam_i))
            v_i = solveri.getDualSolution().copy()
            G_dot_i, g_dot_i, mus_dot_i = np.array(G_dot[i]), np.array(g_dot[i]), np.array(mus_dot[i])
            solveri.jvp(probi, lam_i, v_i, G_dot_i, g_dot_i, mus_dot_i, eps, eps_reg=0.)
            lam_dot.at[i,:,:].set(jnp.array(solveri.getdlamdtheta().copy()))
        return lam, lam_dot

    ccp_layer = lambda G, g, mus, max_iter, eps : contact_solver_layer( G, g, mus, max_iter, eps, ContactProblem, CCPADMMSolver)
    # lcp_layer = lambda G, g, mus, max_iter, eps : contact_solver_layer( G, g, mus, max_iter, eps, LinearContactProblem, LCPQPSolver)
else:
    print("JAX not available, skipping JAX solvers")