import abc
from pycontact.utils.bench import compute_ncp, compute_dual_feas
import matplotlib.pyplot as plt


class CallBackBase(abc.ABC):
    """Base class for callbacks in python contact solvers."""
    def __init__(self) -> None:
        super().__init__()

    @abc.abstractclassmethod
    def call(self, **kwargs):
        pass


class LoggerCallback(CallBackBase):
    def __init__(self) -> None:
        CallBackBase.__init__(self)
        self.ncp_criterion = []
        self.prim_feas = []
        self.dual_feas = []
        self.comp = []

    def call(self, G, g, lam, mu, prim_feas, dual_feas, comp, **kwargs):
        self.ncp_criterion += [compute_ncp([G], [g], [lam], [mu])]
        self.prim_feas += [prim_feas]
        self.dual_feas += [dual_feas]
        self.comp += [comp]

    def reset(self):
        self.ncp_criterion = []
        self.prim_feas = []
        self.dual_feas = []
        self.comp = []

    def plot(self, figs=None, axs=None, block=True):
        if figs is None:
            figs, axs = plt.subplots(2, 2, figsize=(20, 15))
            showing = True
        else:
            for i in range(len(axs)):
                for j in range(len(axs[i])):
                    axs[i][j].clear()
            showing = False
        iters = [t for t in range(len(self.comp))]
        axs[0][0].semilogy(iters, self.ncp_criterion)
        axs[0][0].set_xlabel("iterations")
        axs[0][0].set_ylabel("Contact complementarity")

        axs[0][1].semilogy(iters, self.comp)
        axs[0][1].set_xlabel("iterations")
        axs[0][1].set_ylabel("Problem complementarity")

        axs[1][0].semilogy(iters, self.prim_feas)
        axs[1][0].set_xlabel("iterations")
        axs[1][0].set_ylabel("Primal feasibility")

        axs[1][1].semilogy(iters, self.dual_feas)
        axs[1][1].set_xlabel("iterations")
        axs[1][1].set_ylabel("Dual feasibility")

        plt.draw()
        if showing:
            plt.tight_layout()
            plt.show(block=block)
        return figs, axs

    def close_plot(self):
        plt.close(self.figs)
