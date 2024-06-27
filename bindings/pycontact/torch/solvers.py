try:
    import torch
    from torch.autograd import Function
    torch_available = True
except ImportError:
    torch_available = False

import numpy as np
from pycontact import LCPQPSolver, NCPPGSSolver, CCPPGSSolver, LinearContactProblem, ContactProblem, ContactSolverSettings

if torch_available:
    class ContactLayer(Function):
        @staticmethod
        def forward(ctx,G, g, mus, max_iter, eps, contact_problem_type, contact_solver_type):
            batch_size = g.size()[0]
            lam =torch.zeros(g.size(), dtype = g.dtype)
            ctx.probs = []
            ctx.solvers = []
            settings = ContactSolverSettings()
            settings.max_iter_ = max_iter
            settings.th_stop_ = eps
            for i in range(batch_size):
                Gi,gi,musi = G[i,:,:].detach().numpy(),g[i,:,:].detach().numpy(), mus[i,:,0].detach().numpy()
                probi = contact_problem_type(Gi, gi, musi.tolist())
                solveri = contact_solver_type()
                solveri.setProblem(probi)
                lam_ws = np.zeros(len(gi))
                solveri.solve(probi, lam_ws, settings)
                lam[i,:,0] = torch.from_numpy(solveri.getSolution().copy())
                ctx.probs += [probi]
                ctx.solvers += [solveri]
            ctx.save_for_backward(G,g,mus)
            return lam

        @staticmethod
        def backward(ctx, grad_l):
            '''
            Compute derivatives of the solution of the contact problem
            via implicit differentiation.
            '''
            G,g,mus = ctx.saved_tensors
            batch_size = g.size()[0]
            grad_G, grad_g, grad_mus = None, None, None
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                grad_G =  torch.zeros(G.size(), dtype = g.dtype)
                grad_g = torch.zeros(g.size(), dtype = g.dtype)
                grad_mus = torch.zeros(mus.size(), dtype = g.dtype)
                for i in range(batch_size):
                    solveri = ctx.solvers[i]
                    probi = ctx.probs[i]
                    grad_li = grad_l[i,:,:].detach().numpy()
                    solveri.vjp(probi, grad_li)
                    grad_mus[i] = torch.from_numpy(solveri.getdLdmus().copy()).unsqueeze(-1)
                    grad_G[i] = torch.from_numpy(solveri.getdLdDel().copy())
                    grad_g[i] = torch.from_numpy(solveri.getdLdg().copy()).unsqueeze(-1)
            return grad_G, grad_g, grad_mus, None, None, None, None

    lcp_layer = lambda G, g, mus, max_iter, eps : ContactLayer.apply( G, g, mus, max_iter, eps, LinearContactProblem, LCPQPSolver)
    ccp_layer = lambda G, g, mus, max_iter, eps : ContactLayer.apply( G, g, mus, max_iter, eps, ContactProblem, CCPPGSSolver)


    class ContactLayerApproxBackward(Function):
        @staticmethod
        def forward(ctx,G, g, mus, max_iter, eps, contact_problem_type, contact_solver_type):
            ContactLayer.forward(ctx, G, g, mus, max_iter, eps, contact_problem_type, contact_solver_type)

        @staticmethod
        def backward(ctx, grad_l):
            '''
            Compute derivatives of the solution of the contact problem
            via approximate linear backward.
            '''
            G,g,mus = ctx.saved_tensors
            batch_size = g.size()[0]
            grad_G, grad_g, grad_mus = None, None, None
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                grad_G =  torch.zeros(G.size(), dtype = g.dtype)
                grad_g = torch.zeros(g.size(), dtype = g.dtype)
                grad_mus = torch.zeros(mus.size(), dtype = g.dtype)
                for i in range(batch_size):
                    solveri = ctx.solvers[i]
                    probi = ctx.probs[i]
                    grad_li = grad_l[i,:,:].detach().numpy()
                    solveri.vjp_approx(probi, grad_li)
                    grad_mus[i] = torch.from_numpy(solveri.getdLdmus().copy()).unsqueeze(-1)
                    grad_G[i] = torch.from_numpy(solveri.getdLdDel().copy())
                    grad_g[i] = torch.from_numpy(solveri.getdLdg().copy()).unsqueeze(-1)
            return grad_G, grad_g, grad_mus, None, None, None, None

    ncp_layer = lambda G, g, mus, max_iter, eps : ContactLayerApproxBackward.apply( G, g, mus, max_iter, eps, ContactProblem, NCPPGSSolver)

    class ContactLayerADBackward(Function):
        @staticmethod
        def forward(ctx,G, g, mus, max_iter, eps, contact_problem_type, contact_solver_type):
            ContactLayer.forward(ctx, G, g, mus, max_iter, eps, contact_problem_type, contact_solver_type)

        @staticmethod
        def backward(ctx, grad_l):
            '''
            Compute derivatives of the solution of the contact problem
            via automatic differentiation
            '''
            G,g,mus = ctx.saved_tensors
            batch_size = g.size()[0]
            grad_G, grad_g, grad_mus = None, None, None

            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
                grad_G =  torch.zeros(G.size(), dtype = g.dtype)
                grad_g = torch.zeros(g.size(), dtype = g.dtype)
                grad_mus = torch.zeros(mus.size(), dtype = g.dtype)
                for i in range(batch_size):
                    solveri = ctx.solvers[i]
                    probi = ctx.probs[i]
                    grad_li = grad_l[i,:,:].detach().numpy()
                    solveri.vjp_cppad(probi, grad_li)
                    grad_mus[i] = torch.from_numpy(solveri.getdLdmus().copy()).unsqueeze(-1)
                    grad_G[i] = torch.from_numpy(solveri.getdLdDel().copy())
                    grad_g[i] = torch.from_numpy(solveri.getdLdg().copy()).unsqueeze(-1)
            return grad_G, grad_g, grad_mus, None, None, None, None

    ncp_ad_layer_ = lambda G, g, mus, max_iter, eps : ContactLayerADBackward.apply( G, g, mus, max_iter, eps, ContactProblem, NCPPGSSolver)
    ccp_ad_layer_ = lambda G, g, mus, max_iter, eps : ContactLayerADBackward.apply( G, g, mus, max_iter, eps, ContactProblem, CCPPGSSolver)

else:
    print("Torch not available, skipping torch solvers")