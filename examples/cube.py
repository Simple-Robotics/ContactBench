import numpy as np
import pinocchio as pin
from tap import Tap
from utils.models import create_cubes
from utils.visualize import sub_sample
from pycontact.simulators import (
    CCPADMMSimulator,
    CCPADMMPrimalSimulator,
    CCPNewtonPrimalSimulator,
    CCPPGSSimulator,
    RaisimSimulator,
    RaisimCorrectedSimulator,
    NCPPGSSimulator,
    PinNCPPGSSimulator,
    PinNCPADMMSimulator,
    PyNCPPGSSimulator,
    NCPStagProjSimulator,
    LCPPGSSimulator,
    LCPQPSimulator,
    LCPStagProjSimulator,
    PyCCPADMMPrimalSimulator,
    PyCCPNSNewtonPrimalSimulator,
    PyCCPCVXSimulator,
    PyCCPSCSSimulator
)
import meshcat
import os
from pathlib import Path
from tqdm import trange


class Args(Tap):
    drag: bool = False
    slide: bool = False
    display: bool = False
    timings: bool = False
    record: bool = False
    plot: bool = False
    seed: int = 1234
    save: bool = False
    debug: bool = False

    def process_args(self):
        if self.record:
            self.display = True


args = Args().parse_args()

np.random.seed(args.seed)
pin.seed(args.seed)


a = 0.2  # size of cube
m = 1.0  # mass of cube
mu = 0.4  # friction parameter
eps = 0.0  # elasticity
model, geom_model, visual_model, data, geom_data, visual_data, actuation = create_cubes(
    [a], [m], mu, eps
)

# Number of time steps
duration = 1.098
# time steps
dt = 1e-3
T = int(duration/dt)

# Physical parameters of the contact problem
Kb = 1e-4  # Baumgarte

# numerical precision
th = 1e-9

# initial state
q0 = model.qinit
v0 = np.zeros(model.nv)
if  args.drag or args.slide:
    q0[1] = -5*a
    if args.slide:
        v0[1] = 1.
        # v0[1] = 0.
    # q0[2] += 2e-3
else:
    q0[2] = a
    rand_place = pin.SE3.Random()
    q0[-4:] = pin.pin.SE3ToXYZQUAT(rand_place)[-4:]

q, v = q0.copy(), v0.copy()


# simulator
warm_start = False
statistics = False
magic_forces = False
max_dist_mf = 1e-1
# simulator = NCPPGSSimulator(warm_start=warm_start, statistics=statistics)
# simulator = PyCCPSCSSimulator(warm_start=warm_start, statistics=statistics)
simulator = PinNCPADMMSimulator(warm_start=warm_start, timings = args.timings, statistics=statistics,  magic_forces=magic_forces, max_dist_mf = max_dist_mf)
# simulator = PinNCPPGSSimulator(warm_start=warm_start, statistics=statistics)
# simulator = LCPPGSSimulator(warm_start=warm_start, statistics=statistics)
# simulator = CCPPGSSimulator(model, geom_model, data, geom_data, q0, regularize = False, warm_start=warm_start, statistics=statistics,  magic_forces=magic_forces, max_dist_mf = max_dist_mf)
# simulator = CCPADMMSimulator(warm_start=warm_start, timings = args.timings, statistics=statistics, magic_forces=magic_forces, max_dist_mf = max_dist_mf)
# simulator = NCPStagProjSimulator(warm_start = warm_start, statistics=statistics)
# simulator = RaisimSimulator(warm_start=warm_start, statistics=statistics, magic_forces = magic_forces, max_dist_mf = max_dist_mf)
# simulator = LCPQPSimulator(warm_start=warm_start, statistics=statistics)
# simulator = PyLCPMehrotraSimulator(warm_start=warm_start, statistics=False)
# simulator = PyLCPIPSimulator(warm_start=warm_start, statistics=False)
# simulator = RaisimCorrectedSimulator(warm_start=warm_start, statistics=statistics)
# simulator2 = PyCCPADMMPrimalSimulator(warm_start=True, statistics=statistics)
# simulator = CCPADMMPrimalSimulator(warm_start=warm_start, statistics=statistics, magic_forces=magic_forces, max_dist_mf = max_dist_mf)
# simulator = PyCCPNSNewtonPrimalSimulator(warm_start=True, statistics=statistics, magic_forces=magic_forces, max_dist_mf = max_dist_mf)
# simulator = CCPNewtonPrimalSimulator(warm_start=warm_start, statistics=statistics, magic_forces=magic_forces, max_dist_mf = max_dist_mf)

simulator.setSimulation(model, data, geom_model, geom_data)


# record quantities during trajectory
xs = [np.concatenate((q0, v0))]
pin.computeMechanicalEnergy(model, data, q0,v0)
energies = [data.mechanical_energy]
lams = []
Js = []
us = []
Rs = []
es = []
Gs = []
gs = []
mus = []
signorini_crits =[]
comp_crits = []
ncp_crits = []
if args.timings:
    timings = []
if args.drag:
    def ext_torque(t):
        return np.min([0.9*9.81*2, t*4*9.81])
else:
    def ext_torque(t):
        return 0.
for t in trange(T):
    tau = np.zeros(model.nv)
    fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
    fext[1].linear[1] = ext_torque(t*dt)
    fext[1] = pin.Force(data.oMi[1].actionInverse @ fext[1])
    q, v = simulator.step(
        model = model, data = data, geom_model = geom_model, geom_data = geom_data, q = q, v = v, tau = tau,fext = fext, dt = dt, Kb = Kb,maxIter = 10000, th =th, rel_th_stop =1e-12, eps_reg = 1e-8
    )
    if args.debug:
        print("t=", t)
        simulator.plotCallback(block=False)
        input("Press enter for next step!")
    lams += [simulator.lam]
    Js += [simulator.J]
    Rs += [simulator.R]
    es += [simulator.signed_dist]
    mus += [simulator.mus]
    xs += [np.concatenate((q, v))]
    us += [tau]
    pin.computeMechanicalEnergy(model, data, q,v)
    energies += [data.mechanical_energy]
    if args.timings:
        timings += [simulator.solve_time]

if args.debug:
    simulator.close_plot_callback()

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
os.makedirs(dir_path / "log", exist_ok=True)
os.makedirs(dir_path / "log/cube/", exist_ok=True)
dir_file = dir_path/"log/cube"

if args.drag:
    dir_file = dir_file/"dragged_cube"
    os.makedirs(dir_file,exist_ok=True)

if args.save:
    if simulator.Del is not None:
        simulator.Del.evaluateDel()
        np.savez(
            dir_file/"contact_problem.npz", G=simulator.Del.G_, g=simulator.g, mus=simulator.mus
        )
    np.save(dir_file/"traj.npy", xs)
    np.save(dir_file/"energy.npy", energies)
    np.save(dir_file/"sig_comp.npy",signorini_crits)
    np.save(dir_file/"prob_comp.npy",comp_crits)
    np.save(dir_file/"ncp_comp.npy",ncp_crits)
    if args.timings:
        np.save(dir_file/"timings.npy", timings)

if args.plot:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    figs, axs = plt.subplots(2, 1, figsize=(20, 5))
    iters = [t for t in range(len(xs))]
    axs[0].plot(iters, [x[:3] for x in xs], marker="+")
    axs[0].set_xlabel("time steps")
    axs[0].set_ylabel("q")

    iters = [t for t in range(len(lams))]
    Jlam = [
        np.zeros(3) if len(Js[i]) == 0 else np.dot(Js[i].T, lams[i])[:3, 0]
        for i in range(len(Js))
    ]
    axs[1].plot(iters, Jlam, marker="+")
    axs[1].set_xlabel("time steps")
    axs[1].set_ylabel("contact forces")

    plt.tight_layout()
    if args.save:
        plt.savefig(dir_path/"log/cube/traj.pdf")
    plt.close()

    plt.figure()
    internal_forces  = np.zeros((len(iters), 4))
    internal_forces_cor  = np.zeros((len(iters), 4))
    for i in range(len(iters)):
        Ri = Rs[i]
        lami = lams[i]
        nci = int(len(lami)/3)
        for j in range(nci):
            Rij = Ri[:,3*j:3*(j+1)]
            lamij = lami[3*j:3*(j+1),:]
            fij = Rij @lamij
            fij_cor = fij
            # fij_cor[:2] += Rij[:2] @( (mus[i][j]**2)*lamij[2]/Gs[i][3*j+2, 3*j+2] *Gs[i][3*j+2,3*j:3*j+2])
            internal_forces[i,j] = fij[0]
            # internal_forces_cor[i,j] = fij_cor[0]
    plt.plot(iters, internal_forces, marker="+")
    plt.xlabel("time step")
    plt.ylabel("Internal forces")
    if args.save:
        np.save(dir_file/"internal_forces.npy", internal_forces)
        plt.savefig(dir_path/"log/cube/internal_forces.pdf")
    plt.close()

    plt.figure()
    plt.xlabel("time step")
    plt.ylabel("Internal forces corrected")
    if args.save:
        np.save(dir_path/"log/cube/internal_forces_cor.npy", internal_forces_cor)
        plt.savefig(dir_path/"log/cube/internal_forces_cor.pdf")
    plt.close()

    plt.figure()
    iters = [t for t in range(len(ncp_crits))]
    plt.plot(iters,ncp_crits)
    plt.xlabel("Time steps")
    plt.ylabel("NCP criterion")
    plt.tight_layout()
    if args.save:
        plt.savefig(dir_path/"log/cube/ncp_criterion.pdf")
    plt.close()


if args.display:
    from pinocchio.visualize import MeshcatVisualizer

    # visualize the trajectory
    vizer = MeshcatVisualizer(model, geom_model, visual_model)
    vizer.initViewer(open=True, loadModel=True)

    vizer.viewer["plane"].set_object(meshcat.geometry.Box(np.array([20, 20, 0.1])))
    placement = np.eye(4)
    placement[:3, 3] = np.array([0, 0, -0.05])
    vizer.viewer["plane"].set_transform(placement)
    vizer.display(q0)

    if args.drag or args.slide:
        cp1 = [1.8, 0.0, 0.2]
        cp2 = [0., -1.2, 1.5]
    else:
        cp1 = [0.8, 0.0, 0.2]
        cp2=cp1
    cps_ = [cp1, cp2]
    numrep = len(cps_)
    rps_ = [np.zeros(3)]*numrep
    qs = [x[:model.nq] for x in xs]
    vs = [x[model.nq:] for x in xs]

    max_fps = 30.
    fps = min([max_fps,1./dt])
    qs = sub_sample(qs,dt*T, fps)
    vs = sub_sample(vs,dt*T, fps)

    def get_callback(i: int):
        def _callback(t):
            vizer.setCameraPosition(cps_[i])
            vizer.setCameraTarget(rps_[i])
            pin.forwardKinematics(model, vizer.data, qs[t], vs[t])
            # vizer.drawFrameVelocities(base_link_id)

        return _callback

    input("[Press enter]")

    if args.record:
        ctx = vizer.create_video_ctx(dir_path/"log/cube/simulation.mp4", fps=fps)
    else:
        import contextlib

        ctx = contextlib.nullcontext()
    with ctx:
        for i in range(numrep):
            vizer.play(
                qs,
                1./fps,
                get_callback(i)
            )
