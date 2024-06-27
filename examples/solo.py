import numpy as np
import pinocchio as pin
from tap import Tap
from utils.models import create_solo
from utils.visualize import sub_sample
from pycontact.simulators import (
    CCPADMMSimulator,
    CCPPGSSimulator,
    RaisimSimulator,
    RaisimCorrectedSimulator,
    NCPPGSSimulator,
    NCPStagProjSimulator,
    CCPNewtonPrimalSimulator,
    PinNCPADMMSimulator
)
from pycontact.inverse_dynamics import inverse_contact_dynamics
import meshcat
import os
from pathlib import Path
from tqdm import trange
import cProfile, pstats


class Args(Tap):
    id: bool = False
    drag: bool = False
    perturb: bool = False
    heightfield: bool = False
    timings: bool = False
    display: bool = False
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

mu = 0.9  # friction parameter
model, geom_model, visual_model, data, geom_data, visual_data, actuation = create_solo(
    mu,
    0.,
    0.,
    False,
    True,
    args.heightfield,

)

# Number of time steps
T = 100
# T = 1
# time steps
dt = 1e-3

# Physical parameters of the contact problem
Kb = 1e-4  # Baumgarte
eps = 0.0  # elasticity

# initial state
q0 = model.qinit
v0 = np.zeros(model.nv)
q, v = q0.copy(), v0.copy()


# simulator
warm_start = True
simulator = NCPPGSSimulator(warm_start=warm_start, timings=args.timings)
# simulator = CCPPGSSimulator(warm_start=warm_start, timings=args.timings)
# simulator = CCPADMMSimulator(warm_start=warm_start, timings=args.timings)
# simulator = RaisimSimulator(warm_start=warm_start, timings=args.timings)
# simulator = CCPNewtonPrimalSimulator(warm_start=warm_start, timings=args.timings)
# simulator = RaisimCorrectedSimulator(warm_start=warm_start, timings=args.timings)
# simulator = NCPStagProjSimulator(warm_start=warm_start, timings=args.timings)

simulator.setSimulation(model, data, geom_model, geom_data)

# record quantities during trajectory
xs = [np.concatenate((q0, v0))]
lams = []
Js = []
us = []
Rs = []
es = []
Gs = []
gs = []
timings = []
if args.drag:
    def ext_torque(t):
        fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
        fext[1].linear[1] = np.min([mu*9.81*8, t*10*9.81])
        fext[1] = pin.Force(data.oMi[1].actionInverse @ fext[1])
        return fext
elif args.perturb:
    def ext_torque(t):
        fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
        noise_intensity = 12.
        freq = 2.
        linear_force =(np.array([np.sin(freq*t), np.sin(freq*t+1.), np.sin(freq*t+2)]))*noise_intensity
        fext[1] = pin.Force(np.concatenate((linear_force,np.zeros(3))))
        return fext
else:
    def ext_torque(t):
        fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
        return fext
for t in trange(T):
    if args.id:
        v_ref = np.zeros(model.nv)
        tau = inverse_contact_dynamics(model,data, geom_model, geom_data, q, v, v_ref, dt, 1000, 1e-6)
        print("tau", tau)
    else:
        tau = np.zeros(model.nv)
    # fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
    # fext[1].linear[1] = ext_torque(t*dt)
    # fext[1] = pin.Force(data.oMi[1].actionInverse @ fext[1])
    fext = ext_torque(t*dt)
    # q, v = simulator.step(
    #     model, data, geom_model, geom_data, q, v, tau, fext, dt, Kb, 10000, 1e-6, rel_th_stop = 1e-6, alpha_min = 1e-4, gamma = 0.9
    # )
    q, v = simulator.step(
        model, data, geom_model, geom_data, q, v, tau, fext, dt, Kb, 100, 1e-6, rel_th_stop = 1e-6)
    # q2, v2 = simulator2.step(
    #     model, data, geom_model, geom_data, q, v, tau, fext, dt, Kb, 200, 1e-6
    # )
    if args.debug:
        if t < 3:
            profiler =cProfile.Profile()
            profiler.enable()
            simulator.step(model, data, geom_model, geom_data, q, v, tau, fext, dt, Kb, 500, 1e-7)
            profiler.disable()
            stats = pstats.Stats(profiler)
            # stats.sort_stats('cumtime').print_stats()
            stats.sort_stats('cumtime')
            stats.print_callees(3)
        print("t=", t)
        simulator.plotCallback(block=False)
        input("Press enter for next step!")
    # print(t, simulator.solver.n_iter_)
    # print("lam", simulator.lam)
    lams += [simulator.lam]
    Js += [simulator.J]
    Rs += [simulator.R]
    es += [simulator.signed_dist]
    # Gs += [simulator.G]
    gs += [simulator.g]
    xs += [np.concatenate((q, v))]
    us += [tau]
    if args.timings:
        timings += [simulator.solve_time]

if args.debug:
    simulator.close_plot_callback()


dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
os.makedirs(dir_path / "log", exist_ok=True)
os.makedirs(dir_path / "log/solo/", exist_ok=True)

if args.save:
    if simulator.Del is not None:
        simulator.Del.evaluateDel()
        np.savez(
            dir_path /"log/solo/contact_problem.npz", G=simulator.Del.G_, g=simulator.g, mus=simulator.mus
        )
    np.save(dir_path/"log/solo/traj.npy", xs)
    if args.timings:
        np.save(dir_path/"log/solo/timings.npy", timings)

if args.plot:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    plt.figure()
    iters = [t for t in range(len(lams))]
    internal_forces  = np.zeros((len(iters), 4))
    for i in range(len(iters)):
        Ri = Rs[i]
        lami = lams[i]
        nci = int(len(lami)/3)
        for j in range(nci):
            Rij = Ri[:,3*j:3*(j+1)]
            lamij = lami[3*j:3*(j+1),:]
            fij = Rij @lamij
            internal_forces[i,j] = fij[0]
    plt.plot(iters, internal_forces, marker="+")
    plt.xlabel("time step")
    plt.ylabel("Internal forces")
    if args.save:
        plt.savefig(dir_path/"log/solo/internal_forces.pdf")
    plt.close()

    plt.figure()
    nc = simulator.G.shape[0]//3
    contact_forces = simulator.lam
    resulting_force = simulator.J.T @ simulator.lam
    forces_name = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
    if args.save:
        np.save(dir_path/"log/solo/contact_forces.npy", contact_forces)
        np.save(dir_path/"log/solo/resulting_force.npy", resulting_force)

    if nc ==4:
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')

        x = [1,1,3,3]
        y = [1,3,1,3]
        z = np.zeros(4)

        dx = np.ones(4)
        dy = np.ones(4)
        dz = contact_forces[[2,5,8,11]].squeeze(-1)

        ax1.bar3d(x, y, z, dx, dy, dz, label="contact forces")

        x = [5]
        y = [2]
        z = np.zeros(1)

        dx = np.ones(1)
        dy = np.ones(1)
        dz = resulting_force[2]

        ax1.bar3d(x, y, z, dx, dy, dz, label="resulting force", color = "red")

        ax1.set_zlabel('Forces (N)')

        ax1.view_init(elev=10., azim=-110.)

        if args.save:
            plt.savefig(dir_path/"log/solo/contact_forces.pdf")
        plt.close()



if args.display:
    from pinocchio.visualize import MeshcatVisualizer

    # visualize the trajectory
    vizer = MeshcatVisualizer(model, geom_model, visual_model)
    vizer.initViewer(open=True, loadModel=True)

    if not args.heightfield:
        vizer.viewer["plane"].set_object(meshcat.geometry.Box(np.array([20, 20, 0.1])))
        placement = np.eye(4)
        placement[:3, 3] = np.array([0, 0, -0.05])
        vizer.viewer["plane"].set_transform(placement)
    vizer.displayCollisions(True)
    vizer.display(q0)

    numrep = 2
    cp = [1.6, 0.0, 0.6]
    # camera positions for visualization
    cps_ = [cp.copy() for _ in range(numrep)]
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
        ctx = vizer.create_video_ctx(dir_path/"log/solo/simulation.mp4", fps=fps)
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
