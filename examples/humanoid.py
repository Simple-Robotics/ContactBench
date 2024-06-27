import numpy as np
import pinocchio as pin
from tap import Tap
from utils.models import create_humanoid, addBall, addHumanoid
from utils.visualize import sub_sample
from pycontact.simulators import (
    NCPPGSSimulator,
    CCPADMMSimulator,
    CCPPGSSimulator,
    RaisimSimulator,
    RaisimCorrectedSimulator,
    NCPStagProjSimulator,
    CCPNewtonPrimalSimulator,
    PinNCPADMMSimulator
)
import meshcat
import os
from pathlib import Path
from tqdm import trange


class Args(Tap):
    perturb: bool = False
    ball: bool = False
    twotalos: bool = False
    display: bool = False
    display_collision: bool = False
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
model, geom_model, visual_model, data, geom_data, visual_data, actuation = create_humanoid(
    mu,
    0.,
    0.,
    True,
    True,
    args.twotalos
)
if args.ball:
    model, geom_model, visual_model, data, geom_data, visual_data, actuation = addBall(model, geom_model, visual_model, actuation,0.1,1e-3, mu, 0.)
    model.qinit[-7:-5] = np.array([0.,2.])
if args.twotalos:
    model, geom_model, visual_model, data, geom_data, visual_data, actuation = addHumanoid(model, geom_model, visual_model, actuation, mu, 0. , 1, True, True, True)
    model.qref[7] = 2.1
    place = pin.SE3.Identity()
    place.rotation[:,0] = np.array([-1.,0.,0.])
    place.rotation[:,1] = np.array([0.,-1.,0.])
    model.qref[-4:] = pin.SE3ToXYZQUAT(place)[-4:]
    model.qinit = model.qref.copy()


# Number of time steps
T = 2000
# time steps
dt = 1*1e-3

# Physical parameters of the contact problem
Kb = 1e-4  # Baumgarte
eps = 0.0  # elasticity

# initial state
q0 = model.qinit
v0 = np.zeros(model.nv)
q, v = q0.copy(), v0.copy()


# simulator
warm_start = True
# simulator = NCPPGSSimulator(warm_start=warm_start, timings = args.timings)
# simulator = CCPNewtonPrimalSimulator(warm_start=warm_start, timings = args.timings)
simulator = PinNCPADMMSimulator(warm_start=warm_start, timings = args.timings)
# simulator = CCPPGSSimulator(warm_start=warm_start, timings = args.timings)
# simulator = CCPADMMSimulator(warm_start=warm_start, timings = args.timings)
# simulator = RaisimSimulator(warm_start=warm_start, timings = args.timings)
# simulator = RaisimCorrectedSimulator(warm_start=warm_start, timings = args.timings)
# simulator = NCPStagProjSimulator(warm_start=warm_start, timings = args.timings)

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
    timings = []
if args.perturb:
    def ext_torque(t):
        fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
        noise_intensity = 75.
        freq = 2.
        linear_force =(np.array([np.sin(freq*t), np.sin(freq*t+1.), np.sin(freq*t+2)]))*noise_intensity
        fext[1] = pin.Force(np.concatenate((linear_force,np.zeros(3))))
        return fext
else:
    def ext_torque(t):
        fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
        return fext
for t in trange(T):
    tau = np.zeros(model.nv)
    fext = ext_torque(t*dt)
    q, v = simulator.step(
        model, data, geom_model, geom_data, q, v, tau, fext, dt, Kb, 10000, 1e-6, rel_th_stop = 1e-12
    )
    if args.debug:
        print("t=", t)
        simulator.plotCallback(block=False)
        input("Press enter for next step!")
    # print( "stop_", simulator.solver.stop_)
    # print( "rel_stop_", simulator.solver.rel_stop_)
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
os.makedirs(dir_path / "log/talos/", exist_ok=True)

if args.save:
    if simulator.Del is not None:
        simulator.Del.evaluateDel()
        np.savez(
            dir_path /"log/talos/contact_problem.npz", G=simulator.Del.G_, g=simulator.g, mus=simulator.mus
        )
    if args.timings:
        np.save(dir_path/"log/talos/timings.npy", timings)

if args.plot:
    if args.save:
        np.savez(
            dir_path /"log/talos/traj.npz", Gs=Gs, gs=gs, lams=lams, xs =xs, us= us
        )


if args.display:
    from pinocchio.visualize import MeshcatVisualizer
    # visualize the trajectory
    vizer = MeshcatVisualizer(model, geom_model, visual_model)
    vizer.initViewer(open=True, loadModel=True)

    vizer.viewer["plane"].set_object(meshcat.geometry.Box(np.array([20, 20, 0.1])))
    placement = np.eye(4)
    placement[:3, 3] = np.array([0, 0, -0.05])
    # vizer.displayVisual(True)
    if args.display_collision:
        # vizer.displayVisual(False)
        vizer.displayCollisions(True)
    vizer.display(q0)

    numrep = 2
    if args.twotalos:
        cp = [2.6, 1.0, 1.5]
    else:
        cp = [2.6, 0.0, 1.5]
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
