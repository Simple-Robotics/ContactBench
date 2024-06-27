import torch

import numpy as np
from pycontact.torch.solvers import lcp_layer, ncp_layer, ccp_layer

from pathlib import Path
import os

test_python_path = Path(os.path.dirname(os.path.realpath(__file__)))


def test_lcpqp_solver_forward():
    npzfile = np.load(test_python_path/'data/sliding_cube.npz')
    G = npzfile["G"]
    g = npzfile["g"]
    mus = npzfile["mus"]
    G = torch.from_numpy(G).unsqueeze(0)
    g = torch.from_numpy(g).unsqueeze(0)
    mus = torch.from_numpy(mus).unsqueeze(0).unsqueeze(-1)
    max_iter = 1000
    eps = 1e-7
    lam = lcp_layer(G,g,mus, max_iter, eps)
    loss = 0.5*torch.square(lam).sum()
    assert True

def test_lcpqp_solver_backward():
    npzfile = np.load(test_python_path/'data/sliding_cube.npz')
    G = npzfile["G"]
    g = npzfile["g"]
    mus = npzfile["mus"]
    nc = len(mus)
    G = torch.from_numpy(G).unsqueeze(0)
    g = torch.from_numpy(g).unsqueeze(0)
    mus = torch.from_numpy(mus).unsqueeze(0).unsqueeze(-1)
    G.requires_grad_()
    g.requires_grad_()
    mus.requires_grad_()

    def compute_loss(G,g,mus):
        max_iter = 1000
        eps = 1e-7
        # solver = LCPLayer()
        lam = lcp_layer(G,g,mus, max_iter, eps)
        loss = 0.5*torch.square(lam).sum()
        return loss

    loss = compute_loss(G, g, mus)
    loss.backward()

    G2, g2, mus2 = G.clone(), g.clone(), mus.clone()
    delta = 1e-5
    for i in range(nc):
        mus2[0,i,0] += delta
        loss2 = compute_loss(G,g,mus2)
        dl_dmusi = (loss2-loss).detach().numpy()/delta
        assert np.abs(dl_dmusi - mus.grad[0,i,0].detach().numpy())< 1e-3
        mus2[0,i,0] -= delta

    for i in range(3*nc):
        g2[0,i,0] += delta
        loss2 = compute_loss(G,g2,mus)
        dl_dgi = (loss2-loss).detach().numpy()/delta
        assert np.abs(dl_dgi - g.grad[0,i,0].detach().numpy())< 1e-3
        g2[0,i,0] -= delta

    for i in range(3*nc):
        for j in range(3*nc):
            G2[0,i,j] += delta
            loss2 = compute_loss(G2,g,mus)
            dl_dGij = (loss2-loss).detach().numpy()/delta
            assert np.abs(dl_dGij - G.grad[0,i,j].detach().numpy())< 1e-3
            G2[0,i,j] -= delta

def test_ncppgs_solver_forward():
    npzfile = np.load(test_python_path/'data/sliding_cube.npz')
    G = npzfile["G"]
    g = npzfile["g"]
    mus = npzfile["mus"]
    G = torch.from_numpy(G).unsqueeze(0)
    g = torch.from_numpy(g).unsqueeze(0)
    mus = torch.from_numpy(mus).unsqueeze(0).unsqueeze(-1)
    max_iter = 1000
    eps = 1e-7
    lam = ncp_layer(G,g,mus, max_iter, eps)
    loss = 0.5*torch.square(lam).sum()
    assert True

def test_ncppgs_solver_backward():
    npzfile = np.load(test_python_path/'data/sliding_cube.npz')
    G = npzfile["G"]
    g = npzfile["g"]
    mus = npzfile["mus"]
    nc = len(mus)
    G = torch.from_numpy(G).unsqueeze(0)
    g = torch.from_numpy(g).unsqueeze(0)
    mus = torch.from_numpy(mus).unsqueeze(0).unsqueeze(-1)
    G.requires_grad_()
    g.requires_grad_()
    mus.requires_grad_()

    def compute_loss(G,g,mus):
        max_iter = 1000
        eps = 1e-7
        # solver = LCPLayer()
        lam = ncp_layer(G,g,mus, max_iter, eps)
        loss = 0.5*torch.square(lam).sum()
        return loss

    loss = compute_loss(G, g, mus)
    loss.backward()

    G2, g2, mus2 = G.clone(), g.clone(), mus.clone()
    delta = 1e-5
    for i in range(nc):
        mus2[0,i,0] += delta
        loss2 = compute_loss(G,g,mus2)
        dl_dmusi = (loss2-loss).detach().numpy()/delta
        assert np.abs(dl_dmusi - mus.grad[0,i,0].detach().numpy())< 1e-3
        mus2[0,i,0] -= delta

    for i in range(3*nc):
        g2[0,i,0] += delta
        loss2 = compute_loss(G,g2,mus)
        dl_dgi = (loss2-loss).detach().numpy()/delta
        assert np.abs(dl_dgi - g.grad[0,i,0].detach().numpy())< 1e-3
        g2[0,i,0] -= delta

    for i in range(3*nc):
        for j in range(3*nc):
            G2[0,i,j] += delta
            loss2 = compute_loss(G2,g,mus)
            dl_dGij = (loss2-loss).detach().numpy()/delta
            assert np.abs(dl_dGij - G.grad[0,i,j].detach().numpy())< 1e-3
            G2[0,i,j] -= delta

def test_ccppgs_solver_forward():
    npzfile = np.load(test_python_path/'data/sliding_cube.npz')
    G = npzfile["G"]
    g = npzfile["g"]
    mus = npzfile["mus"]
    G = torch.from_numpy(G).unsqueeze(0)
    g = torch.from_numpy(g).unsqueeze(0)
    mus = torch.from_numpy(mus).unsqueeze(0).unsqueeze(-1)
    max_iter = 1000
    eps = 1e-7
    lam = ccp_layer(G,g,mus, max_iter, eps)
    loss = 0.5*torch.square(lam).sum()
    assert True

def test_ccppgs_solver_backward():
    npzfile = np.load(test_python_path/'data/sliding_cube.npz')
    G = npzfile["G"]
    g = npzfile["g"]
    mus = npzfile["mus"]
    nc = len(mus)
    G = torch.from_numpy(G).unsqueeze(0)
    g = torch.from_numpy(g).unsqueeze(0)
    mus = torch.from_numpy(mus).unsqueeze(0).unsqueeze(-1)
    G.requires_grad_()
    g.requires_grad_()
    mus.requires_grad_()

    def compute_loss(G,g,mus,lam_tar):
        max_iter = 1000
        eps = 1e-7
        # solver = LCPLayer()
        lam = ccp_layer(G,g,mus, max_iter, eps)
        # print("lam : ", lam)
        # print("lam_tar : ", lam_tar)
        loss = 0.5*torch.square(lam- lam_tar).sum()
        return loss


    # lam = ccp_layer(G,g,mus, 1000, 1e-7)
    # print("g", g)
    # print("lam", lam)
    # print("c", G @ lam + g)
    with torch.no_grad():
        mus2 = torch.square(mus*0.9)
        lam_tar = ccp_layer(G,g,mus2, 1000, 1e-7).clone()
    # lam_tar = torch.zeros_like(g)
    loss = compute_loss(G, g, mus, lam_tar)
    loss.backward()

    G2, g2, mus2 = G.clone(), g.clone(), mus.clone()
    delta = 1e-8
    for i in range(nc):
        mus2[0,i,0] += delta
        loss2 = compute_loss(G,g,mus2, lam_tar)
        dl_dmusi = (loss2-loss).detach().numpy()/delta
        # assert np.abs(dl_dmusi - mus.grad[0,i,0].detach().numpy())< 1e-3
        mus2[0,i,0] -= delta

    for i in range(3*nc):
        g2[0,i,0] += delta
        loss2 = compute_loss(G,g2,mus, lam_tar)
        dl_dgi = (loss2-loss).detach().numpy()/delta
        # assert np.abs(dl_dgi - g.grad[0,i,0].detach().numpy())< 1e-3
        g2[0,i,0] -= delta

    for i in range(3*nc):
        for j in range(3*nc):
            G2[0,i,j] += delta
            loss2 = compute_loss(G2,g,mus, lam_tar)
            dl_dGij = (loss2-loss).detach().numpy()/delta
            # assert np.abs(dl_dGij - G.grad[0,i,j].detach().numpy())< 1e-3
            G2[0,i,j] -= delta

    lr = 5e-5
    # lr = 5e-4
    optimizer = optimizer = torch.optim.Adam([mus], lr=lr)
    print("mus", mus)
    G.requires_grad_(False)
    g.requires_grad_(False)
    lam_tar.requires_grad_(False)
    loss = compute_loss(G, g, mus2, lam_tar)
    print("init loss",loss.detach().numpy())
    for i in range(10000):
        # print(i)
        optimizer.zero_grad()
        mus2 = torch.square(mus)
        loss = compute_loss(G, g, mus2, lam_tar)
        loss.backward()
        optimizer.step()
        # print("mus",mus.detach().numpy())
        # print("grad mus: ", mus.grad.detach().numpy())
        # print("loss",loss.detach().numpy())
    print("mus", mus)
    print("final loss",loss.detach().numpy())
    print("lam : ", ccp_layer(G,g,torch.square(mus), 1000, 1e-7))
    print("lam_tar : ", lam_tar)

def test_ncppgs_solver_backward2():
    npzfile = np.load(test_python_path/'data/sliding_cube.npz')
    G = npzfile["G"]
    g = npzfile["g"]
    mus = npzfile["mus"]
    nc = len(mus)
    print("nc", nc)
    G = torch.from_numpy(G).unsqueeze(0)
    g = torch.from_numpy(g).unsqueeze(0)
    mus = torch.from_numpy(mus).unsqueeze(0).unsqueeze(-1)
    G.requires_grad_()
    g.requires_grad_()
    mus.requires_grad_()

    def compute_loss(G,g,mus,lam_tar):
        max_iter = 200
        eps = 1e-7
        # solver = LCPLayer()
        lam = ncp_layer(G,g,mus, max_iter, eps)
        # print("lam : ", lam)
        # print("lam_tar : ", lam_tar)
        loss = 0.5*torch.square(lam- lam_tar).sum()
        return loss


    # lam = ccp_layer(G,g,mus, 1000, 1e-7)
    # print("g", g)
    # print("lam", lam)
    # print("c", G @ lam + g)
    with torch.no_grad():
        mus2 = torch.square(mus*0.9)
        print("mus^2 target", mus2)
        lam_tar = ncp_layer(G,g,mus2, 1000, 1e-7).clone()
    # lam_tar = torch.zeros_like(g)
    loss = compute_loss(G, g, mus, lam_tar)
    loss.backward()

    G2, g2, mus2 = G.clone(), g.clone(), mus.clone()
    delta = 1e-8
    for i in range(nc):
        mus2[0,i,0] += delta
        loss2 = compute_loss(G,g,mus2, lam_tar)
        dl_dmusi = (loss2-loss).detach().numpy()/delta
        # assert np.abs(dl_dmusi - mus.grad[0,i,0].detach().numpy())< 1e-3
        mus2[0,i,0] -= delta

    for i in range(3*nc):
        g2[0,i,0] += delta
        loss2 = compute_loss(G,g2,mus, lam_tar)
        dl_dgi = (loss2-loss).detach().numpy()/delta
        # assert np.abs(dl_dgi - g.grad[0,i,0].detach().numpy())< 1e-3
        g2[0,i,0] -= delta

    for i in range(3*nc):
        for j in range(3*nc):
            G2[0,i,j] += delta
            loss2 = compute_loss(G2,g,mus, lam_tar)
            dl_dGij = (loss2-loss).detach().numpy()/delta
            # assert np.abs(dl_dGij - G.grad[0,i,j].detach().numpy())< 1e-3
            G2[0,i,j] -= delta

    lr = 1e-3
    # lr = 5e-4
    optimizer = optimizer = torch.optim.Adam([mus], lr=lr)
    print("init mus", mus)
    G.requires_grad_(False)
    g.requires_grad_(False)
    lam_tar.requires_grad_(False)
    loss = compute_loss(G, g, mus2, lam_tar)
    print("init loss",loss.detach().numpy())
    for i in range(200):
        # print(i)
        optimizer.zero_grad()
        mus2 = torch.square(mus)
        # print("mus2", mus2)
        loss = compute_loss(G, g, mus2, lam_tar)
        loss.backward()
        optimizer.step()
        print("mus",mus.detach().numpy())
        # print("grad mus: ", mus.grad.detach().numpy())
        # print("loss",loss.detach().numpy())
    print("final mus", mus)
    print("final loss",loss.detach().numpy())
    print("lam : ", ncp_layer(G,g,torch.square(mus), 1000, 1e-7))
    print("lam_tar : ", lam_tar)




if __name__ == '__main__':
    import sys
    import pytest
    # sys.exit(pytest.main(sys.argv))
    # test_ccppgs_solver_backward()
    test_ncppgs_solver_backward2()
