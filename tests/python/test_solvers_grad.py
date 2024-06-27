from pycontact import CCPADMMSolver, ContactProblem, ContactSolverSettings
import numpy as np
from pathlib import Path
import os

test_python_path = Path(os.path.dirname(os.path.realpath(__file__)))


def test_ccpadmm_solver_jvp():
    npzfile = np.load(test_python_path/'data/sliding_cube.npz')
    G = npzfile["G"]
    g = npzfile["g"]
    mus = npzfile["mus"]
    nc = len(mus)
    solver = CCPADMMSolver()

    def compute_loss_and_grad(G,g,mus,lam_tar):
        nc = len(mus)
        mus2 = np.square(mus)
        prob = ContactProblem(G, g, mus2.tolist())
        solver.setProblem(prob)
        max_iter = 1000
        eps = 1e-7
        settings = ContactSolverSettings()
        settings.max_iter_ = max_iter
        settings.th_stop_ = eps
        lam0 = np.zeros_like(lam_tar)
        solver.solve(prob, lam0, settings)
        lam = solver.getSolution().copy()
        v = solver.getDualSolution().copy()
        value = 0.5*np.square(lam- lam_tar).sum()
        dG_dtheta = np.zeros_like(G)
        dg_dtheta = np.zeros((3*nc, nc))
        dmus_dtheta = np.eye(nc)
        dmus2_dtheta =  (2*np.diag(mus)) @ dmus_dtheta
        solver.jvp(prob, lam, v, dG_dtheta, dg_dtheta, dmus2_dtheta)
        dlam_dtheta = solver.getdlamdtheta().copy()
        grads = (lam - lam_tar).T @ dlam_dtheta.T
        return value, grads, dlam_dtheta

    def compute_loss(G,g,mus,lam_tar):
        mus2 = np.square(mus)
        prob = ContactProblem(G, g, mus2.tolist())
        solver.setProblem(prob)
        max_iter = 1000
        eps = 1e-7
        settings = ContactSolverSettings()
        settings.max_iter_ = max_iter
        settings.th_stop_ = eps
        lam0 = np.zeros_like(lam_tar)
        solver.solve(prob, lam0, settings)
        lam = solver.getSolution().copy()
        v = solver.getDualSolution().copy()
        value = 0.5*np.square(lam- lam_tar).sum()
        return value

    mus2 = np.square(mus*0.9)
    prob_tar = ContactProblem(G, g, mus2.tolist())
    solver.setProblem(prob_tar)
    lam0 = np.zeros_like(g)
    settings = ContactSolverSettings()
    settings.max_iter_ = 1000
    settings.th_stop_ = 1e-7
    solver.solve(prob_tar, lam0, settings)
    lam_tar = solver.getSolution().copy()

    loss = compute_loss(G, g, mus, lam_tar)

    G2, g2, mus2 = G.copy(), g.copy(), mus.copy()
    delta = 1e-8
    for i in range(nc):
        mus2[i,] += delta
        loss2 = compute_loss(G,g,mus2, lam_tar)
        dl_dmusi = (loss2-loss)/delta
        # assert np.abs(dl_dmusi - mus.grad[0,i,0].detach().numpy())< 1e-3
        mus2[i] += -delta

    for i in range(3*nc):
        g2[i,0] += (delta)
        loss2 = compute_loss(G,g2,mus, lam_tar)
        dl_dgi = (loss2-loss)/delta
        # assert np.abs(dl_dgi - g.grad[0,i,0].detach().numpy())< 1e-3
        g2[i,0] += (-delta)

    for i in range(3*nc):
        for j in range(3*nc):
            G2[i,j] += (delta)
            loss2 = compute_loss(G2,g,mus, lam_tar)
            dl_dGij = (loss2-loss)/delta
            # assert np.abs(dl_dGij - G.grad[0,i,j].detach().numpy())< 1e-3
            G2[i,j] += (-delta)

    lr = 5e-5
    def SGD_step(i, mus):
        value, grads, _ = compute_loss_and_grad(G, g, mus, lam_tar)
        mus += -lr * grads
        return mus

    def GN_step(i, mus, rho = 1e-7):
        value, grads, J = compute_loss_and_grad(G, g, mus, lam_tar)
        dtheta = - np.linalg.inv(J.T @ J +rho*np.eye(nc)) @ grads
        alpha = 1.
        exp_dec = .5* dtheta.dot(grads)
        for i in range(100):
            dtheta_try = alpha*dtheta
            mus_try = mus +dtheta_try
            value_try = compute_loss(G, g, mus_try, lam_tar)
            if value_try < value + exp_dec:
                mus = mus_try
                break
            else:
                alpha *=0.5
        return mus
    print("mus", mus)
    loss = compute_loss(G, g, mus, lam_tar)
    print("init loss",loss)
    for i in range(100):
        mus = GN_step(i, mus)


def test_ccpadmm_solver_vjp():
    npzfile = np.load(test_python_path/'data/sliding_cube.npz')
    G = npzfile["G"]
    g = npzfile["g"]
    mus = npzfile["mus"]
    nc = len(mus)
    solver = CCPADMMSolver()

    def compute_loss_and_grad(G,g,mus,lam_tar):
        nc = len(mus)
        mus2 = np.square(mus)
        prob = ContactProblem(G, g, mus2.tolist())
        solver.setProblem(prob)
        max_iter = 1000
        eps = 1e-7
        settings = ContactSolverSettings()
        settings.max_iter_ = max_iter
        settings.th_stop_ = eps
        lam0 = np.zeros_like(lam_tar)
        solver.solve(prob, lam0, settings)
        lam = solver.getSolution().copy()
        v = solver.getDualSolution().copy()
        value = 0.5*np.square(lam- lam_tar).sum()
        dl_dlam = (lam - lam_tar).T
        solver.vjp(prob, dl_dlam, settings)
        dl_dmus2 = solver.getdLdmus()
        grads =  dl_dmus2 @ 2*np.diag(mus)
        return value, grads

    def compute_loss(G,g,mus,lam_tar):
        mus2 = np.square(mus)
        prob = ContactProblem(G, g, mus2.tolist())
        solver.setProblem(prob)
        max_iter = 1000
        eps = 1e-7
        settings = ContactSolverSettings()
        settings.max_iter_ = max_iter
        settings.th_stop_ = eps
        lam0 = np.zeros_like(lam_tar)
        solver.solve(prob, lam0, settings)
        lam = solver.getSolution().copy()
        v = solver.getDualSolution().copy()
        value = 0.5*np.square(lam- lam_tar).sum()
        return value

    mus2 = np.square(mus*0.9)
    prob_tar = ContactProblem(G, g, mus2)
    solver.setProblem(prob_tar)
    lam0 = np.zeros_like(lam_tar)
    settings = ContactSolverSettings()
    settings.max_iter_ = 1000
    settings.th_stop_ = 1e-7
    solver.solve(prob_tar, lam0, settings)
    lam_tar = solver.getSolution().copy()

    loss = compute_loss(G, g, mus, lam_tar)

    G2, g2, mus2 = G.copy(), g.copy(), mus.copy()
    delta = 1e-8
    for i in range(nc):
        mus2[i] += delta
        loss2 = compute_loss(G,g,mus2, lam_tar)
        dl_dmusi = (loss2-loss)/delta
        # assert np.abs(dl_dmusi - mus.grad[0,i,0].detach().numpy())< 1e-3
        mus2[i] += -delta

    for i in range(3*nc):
        g2[i,0] += (delta)
        loss2 = compute_loss(G,g2,mus, lam_tar)
        dl_dgi = (loss2-loss)/delta
        # assert np.abs(dl_dgi - g.grad[0,i,0].detach().numpy())< 1e-3
        g2[i,0] += (-delta)

    for i in range(3*nc):
        for j in range(3*nc):
            G2[i,j] += (delta)
            loss2 = compute_loss(G2,g,mus, lam_tar)
            dl_dGij = (loss2-loss)/delta
            # assert np.abs(dl_dGij - G.grad[0,i,j].detach().numpy())< 1e-3
            G2[i,j] += (-delta)

    lr = 5e-5
    def SGD_step(i,mus):
        value, grads = compute_loss_and_grad(G, g, mus, lam_tar)
        mus += -lr * grads
        return mus
    print("mus", mus)
    loss = compute_loss(G, g, mus, lam_tar)
    print("init loss",loss)
    for i in range(100):
        mus = SGD_step(i, mus)


if __name__ == '__main__':
    import sys
    import pytest
    # sys.exit(pytest.main(sys.argv))
    test_ccpadmm_solver_jvp()
