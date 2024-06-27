
from pycontact.jax.solvers import ccp_layer
from jax.example_libraries.optimizers import adam
from jax import value_and_grad, jvp, jacfwd, vmap
import jax.numpy as jnp
from jax.lax import stop_gradient
from jax.test_util import check_grads

import numpy as np
from pathlib import Path
import os

test_python_path = Path(os.path.dirname(os.path.realpath(__file__)))

def test_contact_solver_init():
    ccp_layer


def test_ccpadmm_solver_jvp():
    npzfile = np.load(test_python_path/'data/sliding_cube.npz')
    G = npzfile["G"]
    g = npzfile["g"]
    mus = npzfile["mus"]
    nc = len(mus)
    G = jnp.array(G)[None]
    g = jnp.array(g)[None]
    mus = jnp.array(mus)[None, :, None]

    def compute_loss(G,g,mus,lam_tar):
        mus2 = jnp.square(mus)
        max_iter = 1000
        eps = 1e-7
        lam = ccp_layer(G,g,mus2, max_iter, eps)
        loss = 0.5*jnp.square(lam- lam_tar).sum()
        return loss


    mus2 = jnp.square(stop_gradient(mus)*0.9)
    lam_tar = ccp_layer(G,g,mus2, 1000, 1e-7).clone()
    # lam_tar = torch.zeros_like(g)
    loss = compute_loss(G, g, mus, lam_tar)

    G2, g2, mus2 = G.clone(), g.clone(), mus.clone()
    delta = 1e-8
    for i in range(nc):
        mus2.at[0,i,0].add(delta)
        loss2 = compute_loss(G,g,mus2, lam_tar)
        dl_dmusi = (loss2-loss)/delta
        # assert np.abs(dl_dmusi - mus.grad[0,i,0].detach().numpy())< 1e-3
        mus2.at[0,i,0].add(-delta)

    for i in range(3*nc):
        g2.at[0,i,0].add(delta)
        loss2 = compute_loss(G,g2,mus, lam_tar)
        dl_dgi = (loss2-loss)/delta
        # assert np.abs(dl_dgi - g.grad[0,i,0].detach().numpy())< 1e-3
        g2.at[0,i,0].add(-delta)

    for i in range(3*nc):
        for j in range(3*nc):
            G2.at[0,i,j].add(delta)
            loss2 = compute_loss(G2,g,mus, lam_tar)
            dl_dGij = (loss2-loss)/delta
            # assert np.abs(dl_dGij - G.grad[0,i,j].detach().numpy())< 1e-3
            G2.at[0,i,j].add(-delta)

    lr = 5e-5
    # lr = 5e-4
    opt_init, opt_update, get_params = adam(lr)
    opt_state = opt_init(mus)
    def step(step, opt_state):
        musb = get_params(opt_state)
        grads = jacfwd(compute_loss, argnums=2)(G, g, musb, lam_tar)
        value = 0.
        # value, grads = jvp(compute_loss, (G, g, musb, lam_tar),(jnp.zeros_like(G), jnp.zeros_like(g), jnp.identity(nc)[None, :,:], jnp.zeros_like(lam_tar)))
        # _jvp = lambda s: jvp(compute_loss, (G, g, musb, lam_tar),(jnp.zeros_like(G), jnp.zeros_like(g), s, jnp.zeros_like(lam_tar)))[1]
        # grads = vmap(_jvp, in_axes=1)(jnp.identity(nc)[None, :,:])
        # value = 0.
        # value, grads = value_and_grad(compute_loss)(G,g, musb, lam_tar)
        opt_state = opt_update(step, grads, opt_state)
        return value, opt_state
    print("mus", mus)
    loss = compute_loss(G, g, mus, lam_tar)
    print("init loss",loss)
    for i in range(40000):
        value, opt_state = step(i, opt_state)





if __name__ == '__main__':
    import sys
    import pytest
    # sys.exit(pytest.main(sys.argv))
    test_ccpadmm_solver_jvp()
