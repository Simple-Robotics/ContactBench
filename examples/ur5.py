import numpy as np
import pinocchio as pin
from tap import Tap
from utils.models import create_ur5, addBall
from utils.visualize import sub_sample
from pycontact.simulators import (
    CCPADMMSimulator,
    CCPPGSSimulator,
    NCPPGSSimulator,
    PinNCPADMMSimulator,
    NCPStagProjSimulator,
    RaisimSimulator,
)
from pycontact.utils.pin_utils import inverse_kinematics_trans
from pycontact.inverse_dynamics import inverse_contact_dynamics
import meshcat
import os
from pathlib import Path


class Args(Tap):
    id: bool = False
    wall: bool = False
    ball: bool = False
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
comp = 10000
model, geom_model, visual_model, data, geom_data, visual_data, actuation = create_ur5(
    mu, comp = comp, add_ball=True, add_wall=args.wall
)

if args.ball:
    a = 0.1
    m = 1e-4
    model, geom_model, visual_model, data, geom_data, visual_data, actuation = addBall(model, geom_model, visual_model, actuation,a, m, mu, 0.)
    model.qinit[6:9] = np.array([.85,0.1,a/2])

# Number of time steps
T = 10
# T = 1
# time steps
dt = 1e-3

# Physical parameters of the contact problem
# Kb = 1e-4  # Baumgarte
Kb = 1.0  # Baumgarte
eps = 0.0  # elasticity

# initial state
frame_id = model.getFrameId("wrist_3_link")
if args.wall:
    # frame_trans = np.array([0.0,0.42,0.7])
    frame_trans = np.array([0.0,0.43,0.7])
else:
    frame_trans = np.array([0.0,0.62,0.08])
q0 = inverse_kinematics_trans(model, data, frame_trans, frame_id)
v0 = np.zeros(model.nv)
q, v = q0.copy(), v0.copy()


# simulator
# simulator = NCPPGSSimulator()
simulator = PinNCPADMMSimulator()
# simulator = NCPStagProjSimulator()
# simulator = CCPPGSSimulator()
# simulator = CCPADMMSimulator()
# simulator = RaisimSimulator()


# record quantities during trajectory
xs = [np.concatenate((q0, v0))]
lams = []
lams_id = []
Js = []
us = []
Rs = []
es = []
Gs = []
gs = []
contact_points = []

c_ref = np.array([0.05,0.0,0.0])
for t in range(T):
    if args.id:
        v_ref = np.zeros(model.nv)
        J_wrist = pin.computeJointJacobian(model, data, q, model.getJointId("wrist_3_joint"))
        rot_wrist = data.oMi[model.getJointId("wrist_3_joint")].rotation
        v_ref, _, _, _ = np.linalg.lstsq(rot_wrist@J_wrist[:3,:], c_ref, rcond = None)
        print("v_ref", v_ref)
        tau, lam_id = inverse_contact_dynamics(model,data, geom_model, geom_data, q, v, v_ref, dt, 100, 1e-10, rho = 1e-8, ccp = False)
        # print("v_ref", v_ref)
    else:
        tau = np.zeros(model.nv)
    fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
    q, v = simulator.step(
        model, data, geom_model, geom_data, q, v, tau, fext, dt, Kb, 200, 1e-6
    )
    if args.debug:
        print("t=", t)
        simulator.plotCallback(block=False)
        input("Press enter for next step!")
    lams += [simulator.lam]
    if args.id:
        lams_id += [lam_id]
    Js += [simulator.J]
    Rs += [simulator.R]
    es += [simulator.signed_dist]
    Gs += [simulator.Del]
    gs += [simulator.g]
    xs += [np.concatenate((q, v))]
    us += [tau]
    contact_points += [simulator.contact_points]

if args.debug:
    simulator.close_plot_callback()


dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
os.makedirs(dir_path / "log", exist_ok=True)
os.makedirs(dir_path / "log/ur5/", exist_ok=True)
if args.id:
    os.makedirs(dir_path / "log/ur5/id/", exist_ok=True)

if args.save:
    if args.id:
        np.savez(
            dir_path/"log/ur5/id/contact_forces.npz", lams = lams, lams_id = lams_id
        )


if args.display:
    from pinocchio.visualize import MeshcatVisualizer

    # visualize the trajectory
    vizer = MeshcatVisualizer(model, geom_model, visual_model)
    vizer.initViewer(open=True, loadModel=True)

    vizer.viewer["plane"].set_object(meshcat.geometry.Box(np.array([10, 10, 0.1])))
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
        ctx = vizer.create_video_ctx(dir_path/"log/ur5/simulation.mp4", fps=fps)
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
