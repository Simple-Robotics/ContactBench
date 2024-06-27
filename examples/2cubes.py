import numpy as np
import pinocchio as pin
from tap import Tap
from utils.models import create_cubes, addCube
from utils.visualize import sub_sample
from pycontact.simulators import (
    CCPADMMSimulator,
    CCPADMMPrimalSimulator,
    CCPNewtonPrimalSimulator,
    PyCCPCVXSimulator,
    CCPPGSSimulator,
    NCPPGSSimulator,
    NCPStagProjSimulator,
    RaisimSimulator,
    PinNCPADMMSimulator
)

import meshcat
import os
from pathlib import Path


class Args(Tap):
    drag: bool = False
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


a = 0.2  # size of cube
m1 = 1e-3  # mass of cube 1
m2 = 1e3  # mass of cube 2
mu1 = 0.9  # friction parameter between cube and floor
mu2 = 0.95 # friction parameter between the 2 cubes
el = 0.
comp = 0.
model, geom_model, visual_model, data, geom_data, visual_data, actuation = create_cubes(
    [a], [m1], mu1, el
)

model, geom_model, visual_model, data, geom_data, visual_data, actuation = addCube(model, geom_model, visual_model, actuation, a, m2, mu2,el, comp, color = np.array([1.0,0.2,0.2,1.0]))

# Number of time steps
T = 100
# T = 1
# time steps
dt = 1e-3

# Physical parameters of the contact problem
Kb = 1e-4*0.  # Baumgarte
eps = 0.0  # elasticity

# initial state
q0 = pin.neutral(model)
q0[2] = a / 2 + a/50.
q0[9] = 3. * a / 2 + 3*a/50.
v0 = np.zeros(model.nv)
q, v = q0.copy(), v0.copy()


# simulator
# simulator = NCPPGSSimulator(statistics = True)
# simulator = CCPPGSSimulator(statistics = True)
# simulator = NCPStagProjSimulator()
# simulator = RaisimSimulator(statistics = True)
simulator = PinNCPADMMSimulator(warm_start=True)
# simulator = CCPNewtonPrimalSimulator(statistics=True)
# simulator = CCPADMMSimulator(statistics=True)
# simulator = PyCCPCVXSimulator()

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
ncp_crits = []
signorini_crits =[]
comp_crits = []
if args.drag:
    def ext_torque(t):
        return np.min([.5*(mu1+mu2)*9.81*(m1+m2), t*4*9.81*(m1+m2)])
else:
    def ext_torque(t):
        return 0.
for t in range(T):
    tau_act = np.zeros(actuation.shape[1])
    tau = actuation @ tau_act
    fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
    fext[1].linear[1] = ext_torque(t*dt)
    fext[1] = pin.Force(data.oMi[1].actionInverse @ fext[1])
    q, v = simulator.step(
        model, data, geom_model, geom_data, q, v, tau, fext, dt, Kb, 100, 1e-12, rel_th_stop = 1e-12# , eps_reg = 0.
    )
    if args.debug:
        print("t=", t)
        simulator.plotCallback(block=False)
        input("Press enter for next step!")
    lams += [simulator.lam]
    Js += [simulator.J]
    Rs += [simulator.R]
    es += [simulator.signed_dist]
    Gs += [simulator.Del.G_]
    gs += [simulator.g]
    xs += [np.concatenate((q, v))]
    us += [tau]
    if len(simulator.solver.stats_.ncp_comp_)>0:
        ncp_crits += [simulator.solver.stats_.ncp_comp_[-1]]
        comp_crits += [simulator.solver.stats_.comp_[-1]]
        signorini_crits += [simulator.solver.stats_.sig_comp_[-1]]
    else:
        ncp_crits += [np.nan]
        comp_crits += [np.nan]
        signorini_crits += [np.nan]

if args.debug:
    simulator.close_plot_callback()

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
os.makedirs(dir_path / "log", exist_ok=True)
os.makedirs(dir_path / "log/2cubes/", exist_ok=True)

if args.save:
    np.savez(
        dir_path /"log/2cubes/contact_problem.npz",
        G=simulator.Del.G_,
        g=simulator.g,
        mus=simulator.mus,
    )

if args.plot:  # plotting quantities accross time_steps
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42

    steps = [t for t in range(T)]
    pen_err = []
    for ei in es:
        nci = len(ei)//3
        pen_erri = 0.
        for j in range(nci):
            pen_erri = max(pen_erri, ei[3*j+2])
        if nci==0:
            pen_erri = np.nan
        pen_err+=[pen_erri]

    pen_err = np.array(pen_err)
    plt.figure()
    plt.plot(steps, pen_err, marker="+")
    plt.xlabel("time step")
    plt.ylabel("Penetration error")
    if args.save:
        plt.savefig(dir_path / "log/2cubes/pen_error.pdf")
    plt.close()

    plt.figure()
    plt.plot(steps, ncp_crits, marker="+")
    plt.xlabel("time step")
    plt.ylabel("Contact complementarity")
    if args.save:
        plt.savefig(dir_path / "log/2cubes/ncp_comp.pdf")
    plt.close()

    plt.figure()
    plt.plot(steps, signorini_crits, marker="+")
    plt.xlabel("time step")
    plt.ylabel("Signorini complementarity")
    if args.save:
        plt.savefig(dir_path / "log/2cubes/sig_comp.pdf")
    plt.close()

    plt.figure()
    plt.plot(steps, comp_crits, marker="+")
    plt.xlabel("time step")
    plt.ylabel("Problem complementarity")
    if args.save:
        plt.savefig(dir_path / "log/2cubes/prob_comp.pdf")
    plt.close()

    if args.save:
        np.save(dir_path /"log/2cubes/pen_err.npy",pen_err)
        np.save(dir_path /"log/2cubes/sig_comp.npy",signorini_crits)
        np.save(dir_path /"log/2cubes/ncp_comp.npy",ncp_crits)
        print(np.max(ncp_crits))
        np.save(dir_path /"log/2cubes/prob_comp.npy",comp_crits)


if args.display:
    from pinocchio.visualize import MeshcatVisualizer

    # visualize the trajectory
    vizer = MeshcatVisualizer(model, geom_model, visual_model)
    vizer.initViewer(open=True, loadModel=True)

    vizer.viewer["plane"].set_object(meshcat.geometry.Box(np.array([10, 10, 0.1])))
    placement = np.eye(4)
    placement[:3, 3] = np.array([0, 0, -0.05])
    vizer.viewer["plane"].set_transform(placement)
    vizer.display(q0)

    numrep = 2
    cp = [0.8, 0.0, 0.2]
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
        ctx = vizer.create_video_ctx(dir_path/"log/2cubes/simulation.mp4", fps=fps)
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
