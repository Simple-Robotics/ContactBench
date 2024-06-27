import numpy as np
from utils.models import build_solo_problem, build_talos_problem, build_allegro_hand_problem, build_cube_problem
from pycontact import ContactProblem, NCPPGSSolver, CCPPGSSolver, CCPADMMSolver, RaisimSolver, NCPStagProjSolver, LCPPGSSolver, CCPNewtonPrimalSolver
from tap import Tap
import os
from pathlib import Path

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
os.makedirs(dir_path / "log/timings", exist_ok=True)


class Args(Tap):
    filename: str = None
    save: bool = False
    plot: bool = False
    timings: bool = False
    accuracy : bool = False
    stats : bool = False
    solo: bool = False
    talos: bool = False
    hand: bool = False
    cube: bool = False
    # savedir: str = "log/ncp/others"

    def process_args(self):
        # if self.filename is not None:
        #     if not os.path.exists(self.filename):
        #         raise ValueError("filename does not exist.")
        # else:
        #     raise ValueError("filename does not exist.")
        return


args = Args().parse_args()

if args.filename is not None:
    npzfile = np.load(dir_path / args.filename)
    G = npzfile["G"]
    g = npzfile["g"]
    mus = npzfile["mus"].tolist()
    prob = ContactProblem(G, g, mus)
elif args.solo:
    prob = build_solo_problem(dense = False, drag = True)
    if args.timings:
        os.makedirs(dir_path / "log/timings/solo", exist_ok=True)
        dir_file = Path(dir_path / "log/timings/solo")
    if args.stats:
        os.makedirs(dir_path / "log/iterations/solo", exist_ok=True)
        dir_file = Path(dir_path / "log/iterations/solo")
elif args.talos:
    prob = build_talos_problem(dense = False)
    if args.timings:
        os.makedirs(dir_path / "log/timings/talos", exist_ok=True)
        dir_file = Path(dir_path / "log/timings/talos")
    if args.stats:
        os.makedirs(dir_path / "log/iterations/talos", exist_ok=True)
        dir_file = Path(dir_path / "log/iterations/talos")
elif args.hand:
    prob = build_allegro_hand_problem(dense = False)
    if args.timings:
        os.makedirs(dir_path / "log/timings/hand", exist_ok=True)
        dir_file = Path(dir_path / "log/timings/hand")
    if args.stats:
        os.makedirs(dir_path / "log/iterations/hand", exist_ok=True)
        dir_file = Path(dir_path / "log/iterations/hand")
elif args.cube:
    prob = build_cube_problem(dense = False, drag = True)
    if args.timings:
        os.makedirs(dir_path / "log/timings/cube", exist_ok=True)
        dir_file = Path(dir_path / "log/timings/cube")
    if args.stats:
        os.makedirs(dir_path / "log/iterations/cube", exist_ok=True)
        dir_file = Path(dir_path / "log/iterations/cube")
else:
    raise Exception(
        "A npz file containing a contact problem (G, g, and mus) must be provided"
    )

nc = int(len(prob.g_) / 3)

# solver = NCPPGSSolver()
# solver = CCPPGSSolver()
# solver = LCPPGSSolver()
# solver = RaisimSolver()
solver = CCPADMMSolver()
# solver = CCPNewtonPrimalSolver()
# solver = NCPStagProjSolver()
solver.setProblem(prob)
eps_reg = 1e-0
stats = args.stats
max_iter =  10000
solver.solve(prob, np.zeros((3 * nc, 1)), 100, 1e-6, statistics=stats, eps_reg = eps_reg)
# solver.solve(prob, np.zeros((3 * nc, 1)), max_iter, 1e-12, rel_th_stop=1e-12, statistics=stats)
print(solver.n_iter_)
print(solver.ncp_comp_reg_)

if args.save:
    lam = solver.getSolution()
    v = np.zeros_like(prob.g_)
    prob.Del_.evaluateDel()
    v = prob.Del_.G_ @ lam
    v += prob.g_
    R = np.eye(3*nc)*eps_reg
    lam_stiction_analytical = - np.linalg.inv(prob.Del_.G_ + R) @ prob.g_
    v_stiction_analytical = prob.Del_.G_ @ lam_stiction_analytical + prob.g_
    np.savez(dir_file/"forces_velocities.npz", lam = lam, v= v,  eps_reg = eps_reg)
    np.savez(dir_file/"analytical_forces_velocities.npz", lam = lam_stiction_analytical, v= v_stiction_analytical,  eps_reg = eps_reg)
    if args.stats:
        np.save(dir_file/"stop.npy", solver.stats_.stop_)
        np.save(dir_file/"prim_feas.npy", solver.stats_.prim_feas_)
        np.save(dir_file/"dual_feas.npy", solver.stats_.dual_feas_)
        np.save(dir_file/"comp.npy", solver.stats_.comp_)
        np.save(dir_file/"ncp_comp.npy", solver.stats_.ncp_comp_)


if args.timings:
    import timeit
    timer = timeit.Timer(lambda : solver.solve(prob, np.zeros((3 * nc, 1)), 100, 1e-6))
    timings = timer.repeat(repeat = 100, number = 1)
    if args.save:
        np.save(dir_file/"timings.npy", timings)

if args.plot:

    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    fig = plt.figure()
    iters = [t + 1 for t in range(len(solver.stats_.stop_))]
    plt.semilogy(iters, solver.stats_.stop_, label="Stopping crit", marker="+")
    plt.semilogy(iters, solver.stats_.rel_stop_, label="Rel stopping crit", marker="+")
    plt.xlabel("iterations")
    plt.ylabel("Contact complementarity")
    plt.tight_layout()
    if args.save:
        plt.savefig(dir_file / "contact_comp.pdf")
    plt.close()

    fig = plt.figure()
    iters = [t + 1 for t in range(len(solver.stats_.ncp_comp_))]
    plt.semilogy(iters, solver.stats_.ncp_comp_, marker="+")
    plt.xlabel("iterations")
    plt.ylabel("Contact complementarity")
    plt.tight_layout()
    if args.save:
        plt.savefig(dir_file / "contact_comp.pdf")
    plt.close()

    fig = plt.figure()
    iters = [t + 1 for t in range(len(solver.stats_.comp_))]
    plt.semilogy(iters, solver.stats_.comp_, marker="+")
    plt.xlabel("iterations")
    plt.ylabel("Problem complementarity")
    plt.tight_layout()
    if args.save:
        plt.savefig(dir_file / "prob_comp.pdf")
    plt.close()

    fig = plt.figure()
    iters = [t + 1 for t in range(len(solver.stats_.prim_feas_))]
    plt.semilogy(iters, solver.stats_.prim_feas_, marker="+")
    plt.xlabel("iterations")
    plt.ylabel("Primal feasibility")
    plt.tight_layout()
    if args.save:
        plt.savefig(dir_file / "prim_feas.pdf")
    plt.close()

    fig = plt.figure()
    iters = [t + 1 for t in range(len(solver.stats_.dual_feas_))]
    plt.semilogy(iters, solver.stats_.dual_feas_, marker="+")
    plt.xlabel("iterations")
    plt.ylabel("Dual feasibility")
    plt.tight_layout()
    if args.save:
        plt.savefig(dir_file / "dual_feas.pdf")
    plt.close()
