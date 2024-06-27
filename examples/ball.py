import numpy as np
import pinocchio as pin
from tap import Tap
from utils.models import create_balls, addPlaneToGeomModel
from utils.visualize import sub_sample
from pycontact.utils.bench import (
    compute_contact_torque,
    compute_contact_vel,
    compute_mdp,
    compute_signorini,
    compute_ncp,
    compute_dual_feas,
)
from pycontact.simulators import (
    CCPADMMSimulator,
    PyCCPCVXSimulator,
    CCPPGSSimulator,
    RaisimSimulator,
    NCPPGSSimulator,
    PyCCPADMMPrimalSimulator
)
import meshcat
import os
from pathlib import Path


class Args(Tap):
    drag: bool = False
    lift: bool = False
    display: bool = False
    record: bool = False
    plot: bool = False
    seed: int = 1234
    save: bool = False

    def process_args(self):
        if self.record:
            self.display = True


args = Args().parse_args()

np.random.seed(args.seed)
pin.seed(args.seed)

# Physical parameters of the contact problem
a = 0.2  # size of ball
m = 1.0  # mass of ball
eps = 1.0  # elasticity
mu = 0.9  # friction parameter
model, geom_model, visual_model, data, geom_data, visual_data, actuation = create_balls(
    [a], [m], mu, eps
)
# if args.drag:
#     geom_model = addPlaneToGeomModel(geom_model, np.array([0,0,-1]),np.array([0.,0.,a]),mu,eps,0.,False)
#     geom_data = geom_model.createData()
#     visual_model = addPlaneToGeomModel(visual_model, np.array([0,0,-1]),np.array([0.,0.,a]),mu,eps,0.,True)
#     visual_data = visual_model.createData()

# Number of time steps
T = 800
# time steps
dt = 1e-3
Kb = 1e-4  # Baumgarte

# initial state
q0 = pin.neutral(model)
q0[2] = a / 2
# rand_place = pin.SE3.Random()
# q0[-4:] = pin.pin.SE3ToXYZQUAT(rand_place)[-4:]
# v0 = np.zeros(model.nv)
# v0 = np.random.rand(model.nv)
v0 = np.zeros(model.nv)
# if not args.drag:
#     v0[1] = 1.0
q, v = q0.copy(), v0.copy()

# simulator
# simulator = RaisimSimulator()
# simulator = CCPADMMSimulator()
# simulator = PyCCPCVXSimulator()
# simulator = NCPPGSSimulator()
simulator = PyCCPADMMPrimalSimulator()

simulator.setSimulation(model, data, geom_model, geom_data)


# record quantities during trajectory
xs = [np.concatenate((q0, v0))]
lams = []
Js = []
Rs = []
es = []
us = []
Gs = []
gs = []
if args.drag:
    def ext_torque(t):
        # return np.array([0.,np.min([mu*9.81*1.2, t*2*9.81]),0.])
        return np.array([0.,mu*9.81*1.5,0.])
        # return np.array([0.,mu*9.81*0.01,0.])
elif args.lift:
    def ext_torque(t):
        return np.array([0.,0.,10.])
        # return np.array([0.,0.,5.])
else:
    def ext_torque(t):
        return np.zeros(3)
for t in range(T):
    tau = np.zeros(model.nv)
    fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
    fext[1].linear = ext_torque(t*dt)
    fext[1] = pin.Force(data.oMi[1].actionInverse @ fext[1])
    q, v = simulator.step(
        model, data, geom_model, geom_data, q, v, tau, fext, dt, Kb, 100
    )
    lams += [simulator.lam]
    Js += [simulator.J]
    Rs += [simulator.R]
    es += [simulator.signed_dist]
    Gs += [simulator.Del]
    gs += [simulator.g]
    xs += [np.concatenate((q, v))]
    us += [tau]


dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
os.makedirs(dir_path / "log", exist_ok=True)
os.makedirs(dir_path /"log/ball/", exist_ok=True)

if args.save:
    np.savez(
        dir_path /"log/ball/contact_problem.npz", G=simulator.G, g=simulator.g, mus=simulator.mus
    )

if args.plot:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    # igs, axs = plt.subplots(9, 1, figsize=(20, 15))

    plt.figure()
    iters = [t for t in range(len(xs))]
    plt.plot(iters, [x[:3] for x in xs], marker="+")
    plt.xlabel("time steps")
    plt.tight_layout()
    if args.save:
        plt.savefig(dir_path / "log/ball/positions.pdf")

    plt.figure()
    plt.plot(iters, [x[model.nq : model.nq + 3] for x in xs], marker="+")
    plt.xlabel("time steps")
    plt.ylabel("Joint velocity")
    plt.tight_layout()
    if args.save:
        plt.savefig(dir_path / "log/ball/velocities.pdf")

    plt.figure()
    v_world = []
    for i in range(len(iters)-1):
        # pin.computeJointJacobian(model, data, xs[i][:model.nq], 1)
        # Ji = pin.getJointJacobian(model, data, 1, pin.LOCAL_WORLD_ALIGNED)
        # v_world += [(Ji @xs[i][model.nq:])[:3] ]
        v_world += [(xs[i+1][:3] - xs[i][:3])/dt]
    plt.plot(iters[:-1], v_world, marker="+")
    plt.xlabel("time steps")
    plt.ylabel("Base velocity")
    plt.tight_layout()
    if args.save:
        plt.savefig(dir_path / "log/ball/base_velocities.pdf")

    plt.figure()
    iters = [t for t in range(len(Js))]
    vs = [xs[t + 1][model.nq :] for t in range(len(lams))]
    Jv = compute_contact_vel(Js, vs)
    ids = []
    for i in range(len(iters)):
        if Jv[i] is not None:
            ids += [i]
    # plt.plot([iters[i] for i in ids], [Jv[i] for i in ids], marker="+")
    plt.xlabel("time steps")
    plt.ylabel(" Contact velocity")
    plt.tight_layout()
    if args.save:
        plt.savefig(dir_path / "log/ball/contact_vel.pdf")

    plt.figure()
    iters = [t for t in range(len(lams))]
    lam_plt = [lam[:, 0] if len(lam) != 0 else np.zeros(3) for lam in lams]
    # axs[3].plot(iters, lam_plt, marker="+")
    plt.xlabel("time steps")
    plt.ylabel("contact forces")
    plt.tight_layout()
    if args.save:
        plt.savefig(dir_path / "log/ball/contact_forces.pdf")

    plt.figure()
    iters = [t for t in range(len(lams))]
    Jlam = compute_contact_torque(Js, lams)
    plt.plot(iters, Jlam, marker="+")
    plt.xlabel("time steps")
    plt.ylabel("contact torque")
    plt.tight_layout()
    if args.save:
        plt.savefig(dir_path / "log/ball/contact_torque.pdf")

    plt.figure()
    iters = [t for t in range(len(lams))]
    signorini = compute_signorini(Js, lams, vs)
    plt.plot(iters, signorini, marker="+")
    plt.xlabel("time steps")
    plt.ylabel("Signorini constraint")
    plt.tight_layout()
    if args.save:
        plt.savefig(dir_path / "log/ball/sig_comp.pdf")

    plt.figure()
    iters = [t for t in range(len(lams))]
    # mdp_gap = compute_mdp(Js, vs, lams, mu)
    # plt.plot(iters, mdp_gap, marker="+")
    plt.xlabel("time steps")
    plt.ylabel("MDP gap")
    plt.tight_layout()
    if args.save:
        plt.savefig(dir_path / "log/ball/mdp_gap.pdf")

    plt.figure()
    ncp_crit = compute_ncp(
        Gs,
        gs,
        lams,
        [[mu for i in range(int(len(gs[t]) / 3))] for t in range(len(iters))],
    )
    plt.plot(iters, ncp_crit, marker="+")
    plt.xlabel("time steps")
    plt.ylabel("NCP crit")
    plt.tight_layout()
    if args.save:
        plt.savefig(dir_path / "log/ball/ncp_crit.pdf")

    plt.figure()
    dual_crit = compute_dual_feas(
        Gs,
        gs,
        lams,
        [[mu for i in range(int(len(gs[t]) / 3))] for t in range(len(iters))],
    )
    plt.plot(iters, dual_crit, marker="+")
    plt.xlabel("time steps")
    plt.ylabel("Dual feas")
    plt.tight_layout()
    if args.save:
        plt.savefig(dir_path / "log/ball/dual_feas.pdf")


if args.display:
    from pinocchio.visualize import MeshcatVisualizer

    # visualize the trajectory
    vizer = MeshcatVisualizer(model, geom_model, visual_model)
    vizer.initViewer(open=True, loadModel=True)

    vizer.viewer["plane"].set_object(meshcat.geometry.Box(np.array([1., 10., 0.1])))
    placement = np.eye(4)
    placement[:3, 3] = np.array([0, 0, -0.05])
    vizer.viewer["plane"].set_transform(placement)
    # if args.drag:
    #     vizer.viewer["plane2"].set_object(meshcat.geometry.Box(np.array([1., 10., 0.1])))
    #     placement = np.eye(4)
    #     placement[:3, 3] = np.array([0, 0, a+0.05])
    #     vizer.viewer["plane2"].set_transform(placement)
    vizer.display(q0)

    numrep = 2
    cp = [1.3, 0.0, 0.1]
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
        ctx = vizer.create_video_ctx(dir_path/"log/ball/simulation.mp4", fps=fps)
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
