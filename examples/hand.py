import numpy as np
import pinocchio as pin
from tap import Tap
from utils.models import create_hand, addBall, addCube
from utils.visualize import sub_sample
from pycontact.simulators import (
    CCPADMMSimulator,
    CCPPGSSimulator,
    RaisimSimulator,
    RaisimCorrectedSimulator,
    NCPPGSSimulator,
    CCPNewtonPrimalSimulator
)
import meshcat
import os
from pathlib import Path
from tqdm import trange


class Args(Tap):
    waving: bool = False
    ball: bool = False
    cube: bool = False
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

mu = 0.9  # friction parameter
model, geom_model, visual_model, data, geom_data, visual_data, actuation = create_hand(
    mu,
    0.,
    0.,
    True,
    True
)
if args.ball:
    ball_mass = 10.0
    ball_radius = 0.1
    model, geom_model, visual_model, data, geom_data, visual_data, actuation = addBall(model, geom_model, visual_model, actuation,ball_radius,ball_mass, mu, 0., 0.)
    model.qinit[-7:-4] = np.array([-0.05,-0.04,0.27])
elif args.cube:
    cube_mass = 10.0
    cube_radius = 0.08
    model, geom_model, visual_model, data, geom_data, visual_data, actuation = addCube(model, geom_model, visual_model, actuation,cube_radius,cube_mass, mu, 0., 0., geom_with_box=True)
    model.qinit[-7:-4] = np.array([-0.05,-0.04,0.30])

# Number of time steps
T = 1000
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
# simulator = NCPPGSSimulator(warm_start = warm_start,timings = args.timings)
# simulator = CCPPGSSimulator(warm_start =warm_start, timings = args.timings)
simulator = CCPADMMSimulator(warm_start=warm_start, timings = args.timings)
# simulator = CCPNewtonPrimalSimulator(warm_start=warm_start, timings = args.timings)
# simulator = RaisimSimulator( warm_start=warm_start, timings = args.timings)
# simulator = RaisimCorrectedSimulator(warm_start=warm_start, timings = args.timings)

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
if args.timings:
    timings =  []
def ext_torque(t):
    return 0.
if args.waving:
    def control_torque(t, model, data, q, v, actuation):
        des_acc = np.zeros(model.nv)
        des_acc[2] = 20.*np.cos(10*t)
        b = pin.rnea(model, data, q, v, des_acc)
        tau, _, _, _= np.linalg.lstsq(actuation, b, rcond=None)
        return actuation @tau
else:
    def control_torque(t, model, data, q, v, actuation):
        des_acc = np.zeros(model.nv)
        b = pin.rnea(model, data, q, v, des_acc)
        tau, _, _, _= np.linalg.lstsq(actuation, b, rcond=None)
        return actuation @tau

for t in trange(T):
    tau = control_torque(t*dt, model, data, q, v, actuation)
    fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
    # q, v = simulator.step(
    #     model, data, geom_model, geom_data, q, v, tau, fext, dt, Kb, 10000, 1e-6, alpha_min = 1e-4, gamma = 0.9
    # )
    q, v = simulator.step(
        model, data, geom_model, geom_data, q, v, tau, fext, dt, Kb, 10000, 1e-6
    )
    if args.debug:
        print("t=", t)
        simulator.plotCallback(block=False)
        input("Press enter for next step!")
    # print( "stop_", simulator.solver.stop_)
    # print( "comp_reg_", simulator.solver.comp_reg_)
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
os.makedirs(dir_path / "log/hand/", exist_ok=True)

if args.save:
    if simulator.Del is not None:
        simulator.Del.evaluateDel()
        np.savez(
            dir_path /"log/hand/contact_problem.npz", G=simulator.Del.G_, g=simulator.g, mus=simulator.mus
        )
    np.save(dir_path/"log/hand/traj.npy", xs)
    if args.timings:
        np.save(dir_path/"log/hand/timings.npy", timings)

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
        plt.savefig(dir_path/"log/hand/internal_forces.pdf")
    plt.close()

    plt.figure()
    nc = simulator.G.shape[0]//3
    contact_forces = simulator.lam
    resulting_force = simulator.J.T @ simulator.lam
    forces_name = ["FL_FOOT", "FR_FOOT", "HL_FOOT", "HR_FOOT"]
    if args.save:
        np.save(dir_path/"log/hand/contact_forces.npy", contact_forces)
        np.save(dir_path/"log/hand/resulting_force.npy", resulting_force)

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
            plt.savefig(dir_path/"log/hand/contact_forces.pdf")
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
    # vizer.displayCollisions(True)
    vizer.display(q0)
    pin.SE3ToXYZQUAT( pin.SE3.Random())
    numrep = 2
    cp = [-0.3, -0.3, 0.5]
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
