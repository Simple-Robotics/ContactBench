import pinocchio as pin
import numpy as np
from hppfcl import Plane, Sphere, Ellipsoid, Halfspace, Box, HeightFieldAABB
import example_robot_data as erd
from pycontact.utils.pin_utils import complete_orthonormal_basis
from pycontact import ContactProblem, DelassusDense, DelassusPinocchio
from pycontact.simulators import CCPNewtonPrimalSimulator
from pathlib import Path
import os


def create_balls(length=[0.2], mass=[1.0], mu=0.9, el=0.5, comp = 0.):
    assert len(length) == len(mass) or len(length) == 1 or len(mass) == 1
    N = max(len(length), len(mass))
    if len(length) == 1:
        length = length * N
    if len(mass) == 1:
        mass = mass * N
    rmodel = pin.Model()
    rgeomModel = pin.GeometryModel()

    rgeomModel.frictions = []
    rgeomModel.compliances = []
    rgeomModel.elasticities = []
    # create plane for floor
    n = np.array([0.0, 0.0, 1])
    plane_shape = Halfspace(n, 0)
    T = pin.SE3(np.eye(3), np.zeros(3))
    plane = pin.GeometryObject("plane", 0, 0, T, plane_shape)
    plane.meshColor = np.array([0.5, 0.5, 0.5, 1.0])
    plane_id = rgeomModel.addGeometryObject(plane)
    ball_ids = []
    for n_ball in range(N):
        a = length[n_ball]
        m = mass[n_ball]
        freeflyer = pin.JointModelFreeFlyer()
        joint = rmodel.addJoint(0, freeflyer, pin.SE3.Identity(), "ball_" + str(n_ball))
        rmodel.appendBodyToJoint(
            joint, pin.Inertia.FromSphere(m, a / 2), pin.SE3.Identity()
        )
        ball_shape = Sphere(a / 2)
        geom_ball = pin.GeometryObject(
            "ball_" + str(n_ball), joint, joint, pin.SE3.Identity(), ball_shape
        )
        geom_ball.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball_id = rgeomModel.addGeometryObject(geom_ball)
        for id in ball_ids:
            col_pair = pin.CollisionPair(id, ball_id)
            rgeomModel.addCollisionPair(col_pair)
            rgeomModel.frictions += [mu]
            rgeomModel.compliances += [comp]
            rgeomModel.elasticities += [el]
        ball_ids += [ball_id]
        col_pair = pin.CollisionPair(plane_id, ball_id)
        rgeomModel.addCollisionPair(col_pair)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]

    rmodel.qref = pin.neutral(rmodel)
    rmodel.qinit = rmodel.qref.copy()
    for n_ball in range(N):
        rmodel.qinit[7 * n_ball] += a
        rmodel.qinit[7 * n_ball + 2] += 0.1

    data = rmodel.createData()
    rgeom_data = rgeomModel.createData()
    for req in rgeom_data.collisionRequests:
        req.security_margin = 1e-3
    actuation = np.zeros((rmodel.nv, 1))
    actuation[2, 0] = 1.0
    visual_model = rgeomModel.copy()
    visual_data = visual_model.createData()
    return rmodel, rgeomModel, visual_model, data, rgeom_data, visual_data, actuation


def random_configurations_balls(model, data, geom_model, geom_data):
    valid_conf = False
    N_balls = int(model.nq / 7)
    lower_limit = np.array([-0.5, -0.5, 0.2, 0.0, 0.0, 0.0, 0.0] * N_balls)
    upper_limit = np.array([0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0] * N_balls)
    for _ in range(100):
        q = pin.randomConfiguration(model, lower_limit, upper_limit * 1.0)
        valid_conf = True
        pin.updateGeometryPlacements(model, data, geom_model, geom_data, q)
        pin.computeCollisions(geom_model, geom_data, True)
        for res in geom_data.collisionResults:
            if res.isCollision():
                valid_conf = False
        if valid_conf:
            return q
    return model.q_init


def create_cubes(length=[0.2], mass=[1.0], mu=0.9, el=0.1, comp = 0.):
    assert len(length) == len(mass) or len(length) == 1 or len(mass) == 1
    N = max(len(length), len(mass))
    if len(length) == 1:
        length = length * N
    if len(mass) == 1:
        mass = mass * N
    rmodel = pin.Model()
    rgeomModel = pin.GeometryModel()
    rgeomModel.frictions = []
    rgeomModel.compliances = []
    rgeomModel.elasticities = []

    n = np.array([0.0, 0.0, 1])
    # plane_shape = Plane(n, 0)
    plane_shape = Halfspace(n,0)
    T = pin.SE3(np.eye(3), np.zeros(3))
    plane = pin.GeometryObject("plane", 0, 0, T, plane_shape)
    plane.meshColor = np.array([0.5, 0.5, 0.5, 1.0])
    plane_id = rgeomModel.addGeometryObject(plane)

    ball_ids = []
    for n_cube in range(N):
        a = length[n_cube]
        m = mass[n_cube]
        freeflyer = pin.JointModelFreeFlyer()
        jointCube = rmodel.addJoint(
            0, freeflyer, pin.SE3.Identity(), "joint1_" + str(n_cube)
        )
        M = pin.SE3(np.eye(3), np.matrix([0.0, 0.0, 0.0]).T)
        rmodel.appendBodyToJoint(jointCube, pin.Inertia.FromBox(m, a, a, a), M)
        # rmodel.qref = pin.neutral(rmodel)
        # rmodel.qinit = rmodel.qref.copy()
        # rmodel.qinit[2] += 0.1
        # data = rmodel.createData()
        r = np.array([a / 4, a / 4, a / 4])

        # add balls to cube

        R = pin.utils.eye(3)
        t = np.matrix([a / 2, -a / 2, -a / 2]).T
        # ball_shape1 = Ellipsoid(r)
        ball_shape1 = Sphere(a / 50)
        geom_ball1 = pin.GeometryObject(
            "ball1_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape1
        )
        geom_ball1.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball1_id = rgeomModel.addGeometryObject(geom_ball1)
        col_pair1 = pin.CollisionPair(plane_id, ball1_id)
        rgeomModel.addCollisionPair(col_pair1)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([a / 2, a / 2, -a / 2]).T
        # ball_shape2 = Ellipsoid(r)
        ball_shape2 = Sphere(a / 50)
        geom_ball2 = pin.GeometryObject(
            "ball2_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape2
        )
        geom_ball2.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball2_id = rgeomModel.addGeometryObject(geom_ball2)
        col_pair2 = pin.CollisionPair(plane_id, ball2_id)
        rgeomModel.addCollisionPair(col_pair2)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([a / 2, a / 2, a / 2]).T
        # ball_shape3 = Ellipsoid(r)
        ball_shape3 = Sphere(a / 50)
        geom_ball3 = pin.GeometryObject(
            "ball3_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape3
        )
        geom_ball3.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball3_id = rgeomModel.addGeometryObject(geom_ball3)
        col_pair3 = pin.CollisionPair(plane_id, ball3_id)
        rgeomModel.addCollisionPair(col_pair3)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([a / 2, -a / 2, a / 2]).T
        # ball_shape4 = Ellipsoid(r)
        ball_shape4 = Sphere(a / 50)
        geom_ball4 = pin.GeometryObject(
            "ball4_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape4
        )
        geom_ball4.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball4_id = rgeomModel.addGeometryObject(geom_ball4)
        col_pair4 = pin.CollisionPair(plane_id, ball4_id)
        rgeomModel.addCollisionPair(col_pair4)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([-a / 2, -a / 2, -a / 2]).T
        # ball_shape5 = Ellipsoid(r)
        ball_shape5 = Sphere(a / 50)
        geom_ball5 = pin.GeometryObject(
            "ball5_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape5
        )
        geom_ball5.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball5_id = rgeomModel.addGeometryObject(geom_ball5)
        col_pair5 = pin.CollisionPair(plane_id, ball5_id)
        rgeomModel.addCollisionPair(col_pair5)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([-a / 2, a / 2, -a / 2]).T
        # ball_shape6 = Ellipsoid(r)
        ball_shape6 = Sphere(a / 50)
        geom_ball6 = pin.GeometryObject(
            "ball6_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape6
        )
        geom_ball6.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball6_id = rgeomModel.addGeometryObject(geom_ball6)
        col_pair6 = pin.CollisionPair(plane_id, ball6_id)
        rgeomModel.addCollisionPair(col_pair6)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([-a / 2, a / 2, a / 2]).T
        # ball_shape7 = Ellipsoid(r)
        ball_shape7 = Sphere(a / 50)
        geom_ball7 = pin.GeometryObject(
            "ball7_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape7
        )
        geom_ball7.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball7_id = rgeomModel.addGeometryObject(geom_ball7)
        col_pair7 = pin.CollisionPair(plane_id, ball7_id)
        rgeomModel.addCollisionPair(col_pair7)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]

        R = pin.utils.eye(3)
        t = np.matrix([-a / 2, -a / 2, a / 2]).T
        # ball_shape8 = Ellipsoid(r)
        ball_shape8 = Sphere(a / 50)
        geom_ball8 = pin.GeometryObject(
            "ball8_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape8
        )
        geom_ball8.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        ball8_id = rgeomModel.addGeometryObject(geom_ball8)
        col_pair8 = pin.CollisionPair(plane_id, ball8_id)
        rgeomModel.addCollisionPair(col_pair8)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]
        for id in ball_ids:
            col_pair = pin.CollisionPair(id, ball1_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball2_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball3_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball4_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball5_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball6_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball7_id)
            rgeomModel.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball8_id)
            rgeomModel.addCollisionPair(col_pair)
            rgeomModel.frictions += [mu] * 8
            rgeomModel.compliances += [comp] * 8
            rgeomModel.elasticities += [el] * 8
        ball_ids += [
            ball1_id,
            ball2_id,
            ball3_id,
            ball4_id,
            ball5_id,
            ball6_id,
            ball7_id,
            ball8_id,
        ]

    rmodel.qref = pin.neutral(rmodel)
    rmodel.qinit = rmodel.qref.copy()
    rmodel.qinit[2] += a / 2 + a/50
    for n_cube in range(1, N):
        a = length[n_cube]
        rmodel.qinit[7 * n_cube + 1] = rmodel.qinit[7 * (n_cube - 1) + 1] + a + 0.03
        rmodel.qinit[7 * n_cube + 2] += a / 2
    data = rmodel.createData()
    rgeom_data = rgeomModel.createData()
    for req in rgeom_data.collisionRequests:
        req.security_margin = 1e-3
    actuation = np.eye(rmodel.nv)
    visual_model = rgeomModel.copy()
    for n_cube in range(N):
        R = pin.utils.eye(3)
        t = np.matrix([0.0, 0.0, 0.0]).T
        box_shape = Box(a, a, a)
        geom_box = pin.GeometryObject(
            "box_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), box_shape
        )
        geom_box.meshColor = np.array([0.0, 0.0, 1.0, 0.6])
        box_id = visual_model.addGeometryObject(geom_box)  # only for visualisation
    visual_data = visual_model.createData()
    return (
        rmodel,
        rgeomModel,
        visual_model,
        data,
        rgeom_data,
        visual_data,
        actuation,
    )


def create_solo(mu=0.9, el=0.0, comp = 0., reduced=False, add_balls = False, height_field = False):
    # TODO add reduced model by fixing som joints
    # robot model
    robot = erd.load("solo12")
    rmodel = robot.model.copy()
    if height_field:
        rmodel.qref = np.array([ 0.09906518, 0.20099078, 0.32502457,  0.19414175,
       -0.00524735, -0.97855773, 0.06860185, 0.00968163,
        0.60963582, -1.61206407, -0.02543309, 0.66709088,
       -1.50870083, 0.32405118,  -1.15305599, 1.56867351,
       -0.39097222, -1.29675892,  1.39741073])
    else:
        rmodel.qref = rmodel.referenceConfigurations["standing"]
    if not add_balls and not height_field:
        rmodel.qref[2] += -0.002
    rmodel.qinit = rmodel.qref.copy()
    # Geometry model
    rgeomModel = robot.collision_model
    visual_model = robot.visual_model
    if reduced:
        joints_to_lock = [i for i in range(2,14)]
        model_red,  geom_visual_models= pin.buildReducedModel(rmodel, [rgeomModel, visual_model], joints_to_lock, rmodel.qref)
        geom_model_red, visual_model_red = geom_visual_models[0], geom_visual_models[1]
        model_red.qref = rmodel.qref[:7].copy()
        model_red.qinit = rmodel.qinit[:7].copy()
        rmodel = model_red.copy()
        rmodel.qref, rmodel.qinit = model_red.qref, model_red.qinit
        rgeomModel = geom_model_red.copy()
        visual_model = visual_model_red.copy()

    if not height_field:
        n = np.array([0.0, 0.0, 1])
        p = np.array([0.0, 0.0, 0.0])
        h = np.array([100.0, 100.0, 0.01])
        plane_shape = Halfspace(n, 0)
        # plane_shape = Halfspace(n, 0)
        T = pin.SE3(np.eye(3), np.zeros(3))
        ground_go = pin.GeometryObject("plane", 0, 0, T, plane_shape)
        ground_go.meshColor = np.array([0.5, 0.5, 0.5, 1.0])
    else:
        def ground(xy):
            return (
                np.sin(xy[0] * 30) / 5
                + np.cos(xy[1] ** 2 * 30) / 20
                + np.sin(xy[1] * xy[0] * 50) / 10
            )
        xg = np.arange(-5, 5, .02)
        nx = xg.shape[0]
        xy_g = np.meshgrid(xg, xg)
        xy_g = np.stack(xy_g)
        elev_g = np.zeros((nx, nx))
        elev_g[:, :] = ground(xy_g)

        sx = xg[-1] - xg[0]
        sy = xg[-1] - xg[0]
        elev_g[:, :] = elev_g[::-1, :]
        heightField = HeightFieldAABB(sx, sy, elev_g, np.min(elev_g)-.5)
        pl = pin.SE3.Identity()
        ground_go = pin.GeometryObject("ground", 0, pl, heightField)
        ground_go.meshColor[:] = np.array([128, 149, 255, 200]) / 255.0

    ground_id = rgeomModel.addGeometryObject(ground_go)
    if height_field:
        visual_model.addGeometryObject(ground_go)
    rgeomModel.removeAllCollisionPairs()
    rgeomModel.frictions = []
    rgeomModel.compliances = []
    rgeomModel.elasticities = []

    if add_balls:
        a = 0.01910275
        frames_names = ["HR_FOOT", "HL_FOOT", "FR_FOOT", "FL_FOOT"]

        for name in frames_names:
            frame_id = rmodel.getFrameId(name)
            frame = rmodel.frames[frame_id]
            joint_id = frame.parentJoint
            frame_placement = frame.placement

            shape_name = name + "_shape"
            shape = Sphere(a)
            geometry = pin.GeometryObject(shape_name, joint_id, frame_placement, shape)
            geometry.meshColor = np.array([1.0, 0.2, 0.2, 1.0])

            geom_id = rgeomModel.addGeometryObject(geometry)

            foot_plane = pin.CollisionPair(ground_id, geom_id)  # order should be inverted ?
            rgeomModel.addCollisionPair(foot_plane)
            rgeomModel.frictions += [mu]
            rgeomModel.compliances += [comp]
            rgeomModel.elasticities += [el]

    else:
        for id in range(len(rgeomModel.geometryObjects) - 1):
            col_pair = pin.CollisionPair(id, ground_id)
            rgeomModel.addCollisionPair(col_pair)
            rgeomModel.frictions += [mu]
            rgeomModel.compliances += [comp]
            rgeomModel.elasticities += [el]
    rdata = rmodel.createData()
    rgeom_data = rgeomModel.createData()
    for req in rgeom_data.collisionRequests:
        req.security_margin = 1e-3
    actuation = np.zeros((rmodel.nv, rmodel.nv - 6))
    actuation[6:, :] = np.eye(rmodel.nv - 6)
    visual_data = visual_model.createData()
    return rmodel, rgeomModel, visual_model, rdata, rgeom_data, visual_data, actuation


def create_hand(mu=0.9, el=0.0, comp = 0., reduced=False, add_ff_base = True):
    # robot model
    robot = erd.load("allegro_right_hand")
    hand_model = robot.model.copy()
    hand_model.referenceConfigurations["grasp"] = np.array([0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1., 0.])
    hand_model.qref = hand_model.referenceConfigurations["grasp"]
    hand_model.qinit = hand_model.referenceConfigurations["grasp"]

    # Geometry model
    hand_geomModel = robot.collision_model
    hand_visual_model = robot.visual_model
    if add_ff_base:
        ff_model = pin.Model()
        ff_id = ff_model.addJoint(0,pin.JointModelFreeFlyer(),pin.SE3.Identity(), "base_0")
        ff_model.addJointFrame(ff_id)
        body_inertia = pin.Inertia.FromSphere(.1, .1)
        body_placement = pin.SE3.Identity()
        ff_model.appendBodyToJoint(ff_id, body_inertia, body_placement)
        ff_model.referenceConfigurations['grasp'] = np.array([ 0.,0.,0.,  0.76063856, -0.1055307 , 0.63329215, -0.09609012])
        norm_quat = np.linalg.norm(ff_model.referenceConfigurations['grasp'][-4:])
        ff_model.referenceConfigurations['grasp'][-4:]  /= norm_quat
        ff_model.qref = ff_model.referenceConfigurations['grasp']
        ff_model.qref[2] = .2
        ff_model.qinit = ff_model.referenceConfigurations['grasp']
        ff_model.qinit[2] = .2
        ff_geom_model = pin.GeometryModel()
        frame_id = ff_model.getFrameId('base_0')
        model, geom_model = pin.appendModel(ff_model, hand_model, ff_geom_model, hand_geomModel,frame_id, pin.SE3.Identity())
        _, visual_model = pin.appendModel(ff_model, hand_model, ff_geom_model, hand_visual_model,frame_id, pin.SE3.Identity())
        model.qref = np.concatenate([ff_model.qref, hand_model.qref])
        model.qinit = np.concatenate([ff_model.qinit, hand_model.qinit])
    else:
        model = hand_model
        geom_model = hand_geomModel
        visual_model = hand_visual_model
        model.qref= hand_model.qref
        model.qinit = hand_model.qinit
    if reduced:
        joints_to_lock = [i for i in range(1,model.njoints)]
        model_red,  geom_visual_models= pin.buildReducedModel(model, [geom_model, visual_model], joints_to_lock, model.qref)
        geom_model_red, visual_model_red = geom_visual_models[0], geom_visual_models[1]
        model_red.qref = pin.neutral(model_red)
        model_red.qinit = pin.neutral(model_red)
        model = model_red.copy()
        model.qref, model.qinit = model_red.qref, model_red.qinit
        geom_model = geom_model_red.copy()
        visual_model = visual_model_red.copy()
    geom_model.removeAllCollisionPairs()
    geom_model.frictions = []
    geom_model.compliances = []
    geom_model.elasticities = []
    rdata = model.createData()
    rgeom_data = geom_model.createData()
    for req in rgeom_data.collisionRequests:
        req.security_margin = 1e-3
    actuation = np.eye(model.nv)
    visual_data = visual_model.createData()
    return model, geom_model, visual_model, rdata, rgeom_data, visual_data, actuation



def create_ur5(mu=0.9, el=0.0, comp = 0., add_ball = False, add_wall = False):
    robot = erd.load("ur5")
    model = robot.model
    model.qinit = np.zeros(model.nq)
    model.qref = pin.neutral(model)
    model.frames[-1].name = "end_effector_frame"
    geom_model = robot.collision_model
    geom_model.removeAllCollisionPairs()
    geom_model.frictions = []
    geom_model.compliances = []
    geom_model.elasticities = []
    # adding a plane accounting for the floor
    n = np.array([0.0, 0.0, 1])
    p = np.array([0.0, 0.0, 0.0])
    h = np.array([100.0, 100.0, 0.01])
    plane_shape = Halfspace(n, 0)
    T = pin.SE3(np.eye(3), np.zeros(3))
    plane = pin.GeometryObject("plane", 0, 0, T, plane_shape)
    plane.meshColor = np.array([0.5, 0.5, 0.5, 1.0])
    plane_id = geom_model.addGeometryObject(plane)
    if add_wall:
        n = np.array([0.0, -1.0, 0.0])
        wall_shape = Halfspace(n, 0)
        T = pin.SE3(np.eye(3), np.array([0.0,0.5,0.0]))
        wall = pin.GeometryObject("wall", 0, 0, T, wall_shape)
        wall.meshColor = np.array([0.5, 0.5, 0.5, 1.0])
        wall_id = geom_model.addGeometryObject(wall)
    if add_ball:
        geom_model.col_ids = [plane_id]
        if add_wall:
            geom_model.col_ids += [wall_id]
        model, geom_model = add_ur5_englobing_volumes(model, geom_model, mu, el, comp)
    else:
        # adding collision pairs
        for id in range(len(geom_model.geometryObjects) - 1):
            if not geom_model.geometryObjects[id].name == 'base_link_0':
                col_pair = pin.CollisionPair(id, plane_id)
                geom_model.addCollisionPair(col_pair)
                geom_model.frictions += [mu]
                geom_model.compliances += [comp]
                geom_model.elasticities += [el]
    visual_model = robot.visual_model
    visual_data = visual_model.createData()
    data = model.createData()
    geom_data = geom_model.createData()
    for req in geom_data.collisionRequests:
        req.security_margin = 1e-3
    actuation = np.eye(model.nv)
    model.qinit = pin.neutral(model)
    model.qinit[1] = -np.pi / 4.0
    return model, geom_model, visual_model, data, geom_data, visual_data, actuation

def add_ur5_englobing_volumes(model, geom_model, mu, el, comp):
    # ell_dims = {"wrist_3_link_0": np.array([0.05, 0.09, 0.05])}
    ell_dims = {"wrist_3_link_0": np.array([0.08, 0.08, 0.08])}
    for obj in geom_model.geometryObjects.tolist():
        if obj.name in ell_dims:
            ell_dim = ell_dims[obj.name]
            joint_id = obj.parentJoint
            frame_placement = obj.placement

            shape_name = obj.name + "_shape"
            # shape = Sphere(a)
            shape = Ellipsoid(ell_dim)
            geometry = pin.GeometryObject(shape_name, joint_id, frame_placement, shape)
            geometry.meshColor = np.array([1.0, 0.2, 0.2, .30])

            geom_id = geom_model.addGeometryObject(geometry)

            for id in geom_model.col_ids:
                col_pair = pin.CollisionPair(id, geom_id)
                geom_model.addCollisionPair(col_pair)
                geom_model.frictions += [mu]
                geom_model.compliances += [comp]
                geom_model.elasticities += [el]
            geom_model.col_ids += [geom_id]
    return model, geom_model


def create_humanoid(mu=0.9, el=0.0, comp = 0., reduced=False, add_balls = False, lifted_arms = False):
    # robot model
    robot = erd.load("talos")
    rmodel = robot.model.copy()
    rmodel.qref = rmodel.referenceConfigurations["half_sitting"]
    rmodel.qinit = rmodel.referenceConfigurations["half_sitting"]
    if lifted_arms:
        rmodel.qref[2] += 0.065
        rmodel.qref[9:12] = 0.
        rmodel.qref[15:18] = 0.
        rmodel.qref[20] = np.pi/4.
        rmodel.qref[22] = 3*np.pi/4 #left arm lifted
        rmodel.qref[21] = -1.
        rmodel.qref[30] = -3*np.pi/4 #right arm lifted
        rmodel.qref[29] = 1.
        rmodel.qinit = rmodel.qref.copy()
    # Geometry model
    rgeomModel = robot.collision_model
    visual_model = robot.visual_model
    if add_balls: # adding balls to model contact at end-effectors
        rgeomModel = addBallsToTalos(rgeomModel)
    if reduced:
        joints_to_lock = [i for i in range(2,34)]
        model_red,  geom_visual_models= pin.buildReducedModel(rmodel, [rgeomModel, visual_model], joints_to_lock, rmodel.qref)
        geom_model_red, visual_model_red = geom_visual_models[0], geom_visual_models[1]
        model_red.qref = rmodel.qref[:7].copy()
        model_red.qinit = rmodel.qinit[:7].copy()
        rmodel = model_red.copy()
        rmodel.qref, rmodel.qinit = model_red.qref, model_red.qinit
        rgeomModel = geom_model_red.copy()
        visual_model = visual_model_red.copy()
    # add feet
    a = 0.01910275
    r = np.array([a, a, a])

    n = np.array([0.0, 0.0, 1])
    p = np.array([0.0, 0.0, 0.0])
    h = np.array([100.0, 100.0, 0.01])
    plane_shape = Halfspace(n, 0)
    T = pin.SE3(np.eye(3), np.zeros(3))
    plane = pin.GeometryObject("plane", 0, 0, T, plane_shape)
    plane.meshColor = np.array([0.5, 0.5, 0.5, 1.0])
    plane_id = rgeomModel.addGeometryObject(plane)
    rgeomModel.removeAllCollisionPairs()
    rgeomModel.frictions = []
    rgeomModel.compliances = []
    rgeomModel.elasticities = []
    left_hand_pieces_id = [i for i in range(24,32)]
    right_hand_pieces_id = [i for i in range(41,49)]
    left_foot_id = rgeomModel.getGeometryId('leg_left_6_link_0')
    right_foot_id = rgeomModel.getGeometryId('leg_right_6_link_0')
    for id in range(len(rgeomModel.geometryObjects) - 1):
        if id == left_foot_id or id == right_foot_id or id in left_hand_pieces_id or id in right_hand_pieces_id:
            continue
        col_pair = pin.CollisionPair(id, plane_id)
        rgeomModel.addCollisionPair(col_pair)
        rgeomModel.frictions += [mu]
        rgeomModel.compliances += [comp]
        rgeomModel.elasticities += [el]
    rdata = rmodel.createData()
    rgeom_data = rgeomModel.createData()
    for req in rgeom_data.collisionRequests:
        req.security_margin = 1e-3
    actuation = np.zeros((rmodel.nv, rmodel.nv - 6))
    actuation[6:, :] = np.eye(rmodel.nv - 6)
    visual_data = visual_model.createData()
    return rmodel, rgeomModel, visual_model, rdata, rgeom_data, visual_data, actuation

def addBallsToTalos(rgeomModel):
    #adding balls to feet
    left_foot_id = rgeomModel.getGeometryId('leg_left_6_link_0')
    right_foot_id = rgeomModel.getGeometryId('leg_right_6_link_0')
    left_foot_go = rgeomModel.geometryObjects[left_foot_id]
    right_foot_go = rgeomModel.geometryObjects[right_foot_id]
    left_foot_frame = left_foot_go.parentFrame
    right_foot_frame = right_foot_go.parentFrame
    left_foot_joint = left_foot_go.parentJoint
    right_foot_joint = right_foot_go.parentJoint
    left_foot_placement = left_foot_go.placement
    right_foot_placement = right_foot_go.placement
    ball_radius = .01
    balls_position =[]
    balls_position +=[np.array([0.09,0.06,-0.1])]
    balls_position +=[np.array([-0.1,0.06,-0.1])]
    balls_position +=[np.array([0.09,-0.06,-0.1])]
    balls_position +=[np.array([-0.1,-0.06,-0.1])]
    shape = Sphere(ball_radius)
    for i, ball_position in enumerate(balls_position):
        shape_name = "right_foot_ball"+str(i)
        ball_placement = right_foot_placement.copy()
        ball_placement.translation += ball_position
        ball_geometry = pin.GeometryObject(shape_name, right_foot_joint, ball_placement, shape)
        ball_geometry.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        geom_id = rgeomModel.addGeometryObject(ball_geometry)
        shape_name = "left_foot_ball"+str(i)
        ball_placement = left_foot_placement.copy()
        ball_placement.translation += ball_position
        ball_geometry = pin.GeometryObject(shape_name, left_foot_joint, ball_placement, shape)
        ball_geometry.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
        geom_id = rgeomModel.addGeometryObject(ball_geometry)
    #TODO adding balls to hands
    left_hand_id = rgeomModel.getGeometryId('gripper_left_base_link_0')
    right_hand_id = rgeomModel.getGeometryId('gripper_right_base_link_0')
    left_hand_go = rgeomModel.geometryObjects[left_hand_id]
    right_hand_go = rgeomModel.geometryObjects[right_hand_id]
    left_hand_frame = left_hand_go.parentFrame
    right_hand_frame = right_hand_go.parentFrame
    left_hand_joint = left_hand_go.parentJoint
    right_hand_joint = right_hand_go.parentJoint
    left_hand_placement = left_hand_go.placement
    right_hand_placement = right_hand_go.placement
    ball_radius = .09
    ball_position =np.array([0.00,0.0,-0.09])
    shape = Sphere(ball_radius)
    shape_name = "right_hand_ball"
    ball_placement = right_hand_placement.copy()
    ball_placement.translation += ball_position
    ball_geometry = pin.GeometryObject(shape_name, right_hand_joint, ball_placement, shape)
    ball_geometry.meshColor = np.array([1.0, 0.2, 0.2, .30])
    geom_id = rgeomModel.addGeometryObject(ball_geometry)
    shape_name = "left_hand_ball"
    ball_placement = left_hand_placement.copy()
    ball_placement.translation += ball_position
    ball_geometry = pin.GeometryObject(shape_name, left_hand_joint, ball_placement, shape)
    ball_geometry.meshColor = np.array([1.0, 0.2, 0.2, .3])
    geom_id = rgeomModel.addGeometryObject(ball_geometry)
    return rgeomModel


def random_configurations(model, data, geom_model, geom_data):
    valid_conf = False
    lower_limit = np.zeros(model.nq)
    upper_limit = np.ones(model.nq)
    for _ in range(100):
        q = pin.randomConfiguration(model, lower_limit, upper_limit * 2.0)
        valid_conf = True
        pin.updateGeometryPlacements(model, data, geom_model, geom_data, q)
        pin.computeCollisions(geom_model, geom_data, True)
        for res in geom_data.collisionResults:
            if res.isCollision():
                valid_conf = False
        if valid_conf:
            return q
    return model.q_init


def addPlaneToGeomModel(geom_model, n=np.array([0.0, 0.0, 1.0]), center = np.zeros(3), mu=0.9, el=0., comp = 0., visual=False):  # adding a plane to the current model
    ex, ey = complete_orthonormal_basis(n)
    R = np.array([ex, ey,n]).T
    plane_shape = Plane(n, 0)
    T = pin.SE3(R, center)
    n_plane = geom_model.ngeoms
    plane = pin.GeometryObject("plane"+str(n_plane), 0, 0, T, plane_shape)
    plane.meshColor = np.array([0.5, 0.5, 0.5, 1.0])
    plane_id = geom_model.addGeometryObject(plane)
    if not visual:
        # adding collision pairs
        for id in range(len(geom_model.geometryObjects) - 1):
            #TODO avoid plane/plane collision detection
            if not isinstance(geom_model.geometryObjects[id].geometry,Plane):
                col_pair = pin.CollisionPair(id, plane_id)
                geom_model.addCollisionPair(col_pair)
                geom_model.frictions += [mu]
                geom_model.compliances += [comp]
                geom_model.elasticities += [el]
    return geom_model


def addBallToGeomModel(geom_model, joint, n_ball, a, mu, el, comp, visual=False):
    ball_shape = Sphere(a / 2)
    geom_ball = pin.GeometryObject(
        "ball_" + str(n_ball), joint, joint, pin.SE3.Identity(), ball_shape
    )
    geom_ball.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball_id = geom_model.addGeometryObject(geom_ball)
    if not visual:
        for id in range(len(geom_model.geometryObjects) - 1):
            col_pair = pin.CollisionPair(id, ball_id)
            geom_model.addCollisionPair(col_pair)
            geom_model.frictions += [mu]
            geom_model.compliances += [comp]
            geom_model.elasticities += [el]
    return geom_model


def addCubeToGeomModel(geom_model, jointCube, n_cube, a, mu, el, comp, color = np.array([0.0, 0.0, 1.0, 0.6]), visual=False, geom_with_box = False):
    R = pin.utils.eye(3)
    t = np.matrix([a / 2, -a / 2, -a / 2]).T
    ball_shape1 = Sphere(a / 50)
    geom_ball1 = pin.GeometryObject(
        "ball1_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape1
    )
    geom_ball1.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball1_id = geom_model.addGeometryObject(geom_ball1)

    R = pin.utils.eye(3)
    t = np.matrix([a / 2, a / 2, -a / 2]).T
    ball_shape2 = Sphere(a / 50)
    geom_ball2 = pin.GeometryObject(
        "ball2_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape2
    )
    geom_ball2.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball2_id = geom_model.addGeometryObject(geom_ball2)

    R = pin.utils.eye(3)
    t = np.matrix([a / 2, a / 2, a / 2]).T
    ball_shape3 = Sphere(a / 50)
    geom_ball3 = pin.GeometryObject(
        "ball3_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape3
    )
    geom_ball3.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball3_id = geom_model.addGeometryObject(geom_ball3)

    R = pin.utils.eye(3)
    t = np.matrix([a / 2, -a / 2, a / 2]).T
    ball_shape4 = Sphere(a / 50)
    geom_ball4 = pin.GeometryObject(
        "ball4_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape4
    )
    geom_ball4.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball4_id = geom_model.addGeometryObject(geom_ball4)

    R = pin.utils.eye(3)
    t = np.matrix([-a / 2, -a / 2, -a / 2]).T
    ball_shape5 = Sphere(a / 50)
    geom_ball5 = pin.GeometryObject(
        "ball5_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape5
    )
    geom_ball5.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball5_id = geom_model.addGeometryObject(geom_ball5)

    R = pin.utils.eye(3)
    t = np.matrix([-a / 2, a / 2, -a / 2]).T
    ball_shape6 = Sphere(a / 50)
    geom_ball6 = pin.GeometryObject(
        "ball6_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape6
    )
    geom_ball6.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball6_id = geom_model.addGeometryObject(geom_ball6)

    R = pin.utils.eye(3)
    t = np.matrix([-a / 2, a / 2, a / 2]).T
    ball_shape7 = Sphere(a / 50)
    geom_ball7 = pin.GeometryObject(
        "ball7_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape7
    )
    geom_ball7.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball7_id = geom_model.addGeometryObject(geom_ball7)

    R = pin.utils.eye(3)
    t = np.matrix([-a / 2, -a / 2, a / 2]).T
    ball_shape8 = Sphere(a / 50)
    geom_ball8 = pin.GeometryObject(
        "ball8_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape8
    )
    geom_ball8.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball8_id = geom_model.addGeometryObject(geom_ball8)
    if visual or geom_with_box:
        R = pin.utils.eye(3)
        t = np.matrix([0.0, 0.0, 0.0]).T
        box_shape = Box(a, a, a)
        geom_box = pin.GeometryObject(
            "box_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), box_shape
        )
        geom_box.meshColor = color
        box_id = geom_model.addGeometryObject(geom_box)  # only for visualisation
    if not visual:
        n_self_collide = 9 if geom_with_box else 8
        for id in range(len(geom_model.geometryObjects) - n_self_collide):
            col_pair = pin.CollisionPair(id, ball1_id)
            geom_model.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball2_id)
            geom_model.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball3_id)
            geom_model.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball4_id)
            geom_model.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball5_id)
            geom_model.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball6_id)
            geom_model.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball7_id)
            geom_model.addCollisionPair(col_pair)
            col_pair = pin.CollisionPair(id, ball8_id)
            geom_model.addCollisionPair(col_pair)
            if geom_with_box:
                col_pair = pin.CollisionPair(id, box_id)
                geom_model.addCollisionPair(col_pair)
            geom_model.frictions += [mu] * 8
            geom_model.compliances += [comp] * 8
            geom_model.elasticities += [el] * 8
    return geom_model

def addCubeToGeomModelFull(geom_model, jointCube, n_cube, a, mu, el, comp,
                           color = np.array([0.0, 0.0, 1.0, 0.6])):
    # Add the cube itself
    M = pin.SE3.Identity()
    box_shape = Box(a, a, a)
    geom_box = pin.GeometryObject(
        "box_" + str(n_cube), jointCube, jointCube, M, box_shape
    )
    geom_box.meshColor = color
    box_id = geom_model.addGeometryObject(geom_box)

    # Add balls for each corner of the cube
    R = pin.utils.eye(3)
    t = np.matrix([a / 2, -a / 2, -a / 2]).T
    ball_shape1 = Sphere(a / 50)
    geom_ball1 = pin.GeometryObject(
        "ball1_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape1
    )
    geom_ball1.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball1_id = geom_model.addGeometryObject(geom_ball1)

    R = pin.utils.eye(3)
    t = np.matrix([a / 2, a / 2, -a / 2]).T
    ball_shape2 = Sphere(a / 50)
    geom_ball2 = pin.GeometryObject(
        "ball2_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape2
    )
    geom_ball2.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball2_id = geom_model.addGeometryObject(geom_ball2)

    R = pin.utils.eye(3)
    t = np.matrix([a / 2, a / 2, a / 2]).T
    ball_shape3 = Sphere(a / 50)
    geom_ball3 = pin.GeometryObject(
        "ball3_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape3
    )
    geom_ball3.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball3_id = geom_model.addGeometryObject(geom_ball3)

    R = pin.utils.eye(3)
    t = np.matrix([a / 2, -a / 2, a / 2]).T
    ball_shape4 = Sphere(a / 50)
    geom_ball4 = pin.GeometryObject(
        "ball4_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape4
    )
    geom_ball4.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball4_id = geom_model.addGeometryObject(geom_ball4)

    R = pin.utils.eye(3)
    t = np.matrix([-a / 2, -a / 2, -a / 2]).T
    ball_shape5 = Sphere(a / 50)
    geom_ball5 = pin.GeometryObject(
        "ball5_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape5
    )
    geom_ball5.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball5_id = geom_model.addGeometryObject(geom_ball5)

    R = pin.utils.eye(3)
    t = np.matrix([-a / 2, a / 2, -a / 2]).T
    ball_shape6 = Sphere(a / 50)
    geom_ball6 = pin.GeometryObject(
        "ball6_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape6
    )
    geom_ball6.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball6_id = geom_model.addGeometryObject(geom_ball6)

    R = pin.utils.eye(3)
    t = np.matrix([-a / 2, a / 2, a / 2]).T
    ball_shape7 = Sphere(a / 50)
    geom_ball7 = pin.GeometryObject(
        "ball7_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape7
    )
    geom_ball7.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball7_id = geom_model.addGeometryObject(geom_ball7)

    R = pin.utils.eye(3)
    t = np.matrix([-a / 2, -a / 2, a / 2]).T
    ball_shape8 = Sphere(a / 50)
    geom_ball8 = pin.GeometryObject(
        "ball8_" + str(n_cube), jointCube, jointCube, pin.SE3(R, t), ball_shape8
    )
    geom_ball8.meshColor = np.array([1.0, 0.2, 0.2, 1.0])
    ball8_id = geom_model.addGeometryObject(geom_ball8)

    # Add collision pairs:
    for id in range(len(geom_model.geometryObjects) - (8 + 1)):
        col_pair = pin.CollisionPair(id, box_id)
        geom_model.addCollisionPair(col_pair)
        #
        col_pair = pin.CollisionPair(id, ball1_id)
        geom_model.addCollisionPair(col_pair)
        #
        col_pair = pin.CollisionPair(id, ball2_id)
        geom_model.addCollisionPair(col_pair)
        #
        col_pair = pin.CollisionPair(id, ball3_id)
        geom_model.addCollisionPair(col_pair)
        #
        col_pair = pin.CollisionPair(id, ball4_id)
        geom_model.addCollisionPair(col_pair)
        #
        col_pair = pin.CollisionPair(id, ball5_id)
        geom_model.addCollisionPair(col_pair)
        #
        col_pair = pin.CollisionPair(id, ball6_id)
        geom_model.addCollisionPair(col_pair)
        #
        col_pair = pin.CollisionPair(id, ball7_id)
        geom_model.addCollisionPair(col_pair)
        #
        col_pair = pin.CollisionPair(id, ball8_id)
        geom_model.addCollisionPair(col_pair)
        #
        geom_model.frictions += [mu] * (8 + 1)
        geom_model.compliances += [comp] * (8 + 1)
        geom_model.elasticities += [el] * (8 + 1)
    return geom_model


def addCube(
    model, geom_model, visual_model, actuation, a, m, mu, el, comp, color = np.array([0.0, 0.0, 1.0, 0.6]), actuated = False, geom_with_box=False
):  # adding a cube to the current model
    freeflyer = pin.JointModelFreeFlyer()
    n_cube = model.nbodies
    jointCube = model.addJoint(
        0, freeflyer, pin.SE3.Identity(), "joint1_" + str(n_cube)
    )
    M = pin.SE3(np.eye(3), np.matrix([0.0, 0.0, 0.0]).T)
    model.appendBodyToJoint(jointCube, pin.Inertia.FromBox(m, 0.2, 0.2, 0.2), M)

    # add balls to cube
    geom_model = addCubeToGeomModel(geom_model, jointCube, n_cube, a, mu, el, comp, geom_with_box=geom_with_box)
    visual_model = addCubeToGeomModel(visual_model, jointCube, n_cube, a, mu, el, comp, color, True)
    data = model.createData()
    geom_data = geom_model.createData()
    for req in geom_data.collisionRequests:
        req.security_margin = 1e-3
    visual_data = visual_model.createData()
    actuation_pred = actuation.copy()
    if actuated:
        actuation = np.zeros((model.nv, actuation_pred.shape[1] + 6))
        actuation[-6:, -6:] = np.eye(6)
    else:
        actuation = np.zeros((model.nv, actuation_pred.shape[1]))
    actuation[: actuation_pred.shape[0], : actuation_pred.shape[1]] = actuation_pred
    model.qinit = np.concatenate([model.qinit, np.array([0.,2*a,2*a,0.,0.,0.,1.])])
    model.qref = np.concatenate([model.qref, np.array([0.,2*a,2*a,0.,0.,0.,1.])])
    return model, geom_model, visual_model, data, geom_data, visual_data, actuation


def addBall(
    model, geom_model, visual_model, actuation, a, m, mu, el, comp, actuated = False
):  # adding a ball to the current model
    freeflyer = pin.JointModelFreeFlyer()
    n_ball = model.nbodies
    joint = model.addJoint(0, freeflyer, pin.SE3.Identity(), "ball_" + str(n_ball))
    model.appendBodyToJoint(joint, pin.Inertia.FromSphere(m, a / 2), pin.SE3.Identity())
    geom_model = addBallToGeomModel(geom_model, joint, n_ball, a, mu, el, comp)
    visual_model = addBallToGeomModel(visual_model, joint, n_ball, a, mu, el, comp, True)
    data = model.createData()
    geom_data = geom_model.createData()
    for req in geom_data.collisionRequests:
        req.security_margin = 1e-3
    visual_data = visual_model.createData()
    actuation_pred = actuation.copy()
    if actuated:
        actuation = np.zeros((model.nv, actuation_pred.shape[1] + 6))
        actuation[-6:, -6:] = np.eye(6)
    else:
        actuation = np.zeros((model.nv, actuation_pred.shape[1]))
    actuation[: actuation_pred.shape[0], : actuation_pred.shape[1]] = actuation_pred
    model.qinit = np.concatenate([model.qinit, np.array([0.,2*a,2*a,0.,0.,0.,1.])])
    model.qref = np.concatenate([model.qref, np.array([0.,2*a,2*a,0.,0.,0.,1.])])
    return model, geom_model, visual_model, data, geom_data, visual_data, actuation


def addSolo(model, geom_model, visual_model, actuation, mu, el, comp, reduced=False):
    robot = erd.load("solo12")
    solo_model = robot.model.copy()
    solo_model.qref = solo_model.referenceConfigurations["standing"]
    solo_model.qinit = solo_model.referenceConfigurations["standing"]
    solo_geom_model = robot.collision_model
    solo_visual_model = robot.visual_model
    if reduced:
        joints_to_lock = [i for i in range(2,14)]
        model_red,  geom_visual_models= pin.buildReducedModel(solo_model, [solo_geom_model, solo_visual_model], joints_to_lock, solo_model.qref)
        geom_model_red, visual_model_red = geom_visual_models[0], geom_visual_models[1]
        model_red.qref = solo_model.qref[:7].copy()
        model_red.qinit = solo_model.qinit[:7].copy()
        solo_model = model_red.copy()
        solo_model.qref, solo_model.qinit = model_red.qref, model_red.qinit
        solo_geom_model = geom_model_red.copy()
        solo_visual_model = visual_model_red.copy()
    R = np.eye(3)
    t = np.zeros(3)
    new_model, new_geom_model = pin.appendModel(
        model,
        solo_model,
        geom_model,
        solo_geom_model,
        0,
        pin.SE3(R, t),
    )
    _, new_visual_model = pin.appendModel(
        model, solo_model, visual_model, solo_visual_model, 0, pin.SE3(R, t)
    )
    # new_model, new_geom_visual_models = pin.appendModel(
    #     model,
    #     solo_model,
    #     [geom_model, visual_model],
    #     [solo_geom_model, solo_visual_model],
    #     0,
    #     pin.SE3(R, t),
    # )
    # new_geom_model, new_visual_model = new_geom_visual_models[0], new_geom_visual_models[1]
    new_geom_model.frictions = geom_model.frictions + [mu] * (
        solo_geom_model.ngeoms * geom_model.ngeoms
    )
    new_geom_model.compliances = geom_model.compliances + [comp] * (
        solo_geom_model.ngeoms * geom_model.ngeoms
    )
    new_geom_model.elasticities = geom_model.elasticities + [el] * (
        solo_geom_model.ngeoms * geom_model.ngeoms
    )
    # TODO keep qref, qinit
    new_model.qinit = np.concatenate([model.qinit, solo_model.qinit])
    new_model.qref = np.concatenate([model.qref, solo_model.qref])
    data = new_model.createData()
    geom_data = new_geom_model.createData()
    for req in geom_data.collisionRequests:
        req.security_margin = 1e-3
    visual_data = new_visual_model.createData()
    solo_actuation = np.zeros((solo_model.nv, solo_model.nv - 6))
    solo_actuation[6:, :] = np.eye(solo_model.nv - 6)
    new_actuation = np.block(
        [
            [actuation, np.zeros((model.nv, solo_actuation.shape[1]))],
            [np.zeros((solo_model.nv, actuation.shape[1])), solo_actuation],
        ]
    )
    return (
        new_model,
        new_geom_model,
        new_visual_model,
        data,
        geom_data,
        visual_data,
        new_actuation,
    )

def addHumanoid(model, geom_model, visual_model, actuation, mu, el, id, reduced=False, collision_balls=False, lifted_arms = False):
    robot = erd.load("talos")
    talos_model = robot.model.copy()
    talos_model.qref = talos_model.referenceConfigurations["half_sitting"]
    talos_model.qinit = talos_model.referenceConfigurations["half_sitting"]
    if lifted_arms:
        talos_model.qref[2] += 0.065
        talos_model.qref[9:12] = 0.
        talos_model.qref[15:18] = 0.
        talos_model.qref[20] = np.pi/4.
        talos_model.qref[22] = 3.*np.pi/4. #left arm lifted
        talos_model.qref[21] = -1.
        talos_model.qref[30] = -3.*np.pi/4. #right arm lifted
        talos_model.qref[29] = 1.
        talos_model.qinit = talos_model.qref.copy()
    # Geometry model
    talos_geom_model = robot.collision_model
    talos_geom_model.removeAllCollisionPairs()
    talos_visual_model = robot.visual_model
    if collision_balls:
        talos_geom_model = addBallsToTalos(talos_geom_model)
    for i in range(1,len(talos_model.joints)):
        talos_model.names[i] += "_"+str(id)
    for i in range(1,len(talos_model.frames)):
        talos_model.frames[i].name += "_"+str(id)
    for i in range(len(talos_geom_model.geometryObjects)):
        talos_geom_model.geometryObjects[i].name += "_"+str(id)
    for i in range(len(talos_visual_model.geometryObjects)):
        talos_visual_model.geometryObjects[i].name += "_"+str(id)
    if reduced:
        joints_to_lock = [i for i in range(2,34)]
        model_red,  geom_visual_models= pin.buildReducedModel(talos_model, [talos_geom_model, talos_visual_model], joints_to_lock, talos_model.qref)
        geom_model_red, visual_model_red = geom_visual_models[0], geom_visual_models[1]
        model_red.qref = talos_model.qref[:7].copy()
        model_red.qinit = talos_model.qinit[:7].copy()
        talos_model = model_red.copy()
        talos_model.qref, talos_model.qinit = model_red.qref, model_red.qinit
        talos_geom_model = geom_model_red.copy()
        talos_visual_model = visual_model_red.copy()
    R = np.eye(3)
    t = np.zeros(3)
    new_model, new_geom_model = pin.appendModel(
        model,
        talos_model,
        geom_model,
        talos_geom_model,
        0,
        pin.SE3(R, t),
    )
    _, new_visual_model = pin.appendModel(
        model, talos_model, visual_model, talos_visual_model, 0, pin.SE3(R, t)
    )
    new_geom_model.frictions = geom_model.frictions + [mu] * (
        talos_geom_model.ngeoms * geom_model.ngeoms
    )
    new_geom_model.compliances = geom_model.compliances + [mu] * (
        talos_geom_model.ngeoms * geom_model.ngeoms
    )
    new_geom_model.elasticities = geom_model.elasticities + [el] * (
        talos_geom_model.ngeoms * geom_model.ngeoms
    )
    # keep qref, qinit
    new_model.qinit = np.concatenate([model.qinit, talos_model.qinit])
    new_model.qref = np.concatenate([model.qref, talos_model.qref])
    data = new_model.createData()
    geom_data = new_geom_model.createData()
    for req in geom_data.collisionRequests:
        req.security_margin = 1e-3
    visual_data = new_visual_model.createData()
    talos_actuation = np.zeros((talos_model.nv, talos_model.nv - 6))
    talos_actuation[6:, :] = np.eye(talos_model.nv - 6)
    new_actuation = np.block(
        [
            [actuation, np.zeros((model.nv, talos_actuation.shape[1]))],
            [np.zeros((talos_model.nv, actuation.shape[1])), talos_actuation],
        ]
    )
    return (
        new_model,
        new_geom_model,
        new_visual_model,
        data,
        geom_data,
        visual_data,
        new_actuation,
    )


def addRandomObject(model, geom_model, visual_model, actuation):
    obj_type = np.random.randint(2)
    if obj_type == 0:
        a = 0.2 + 0.1 * np.random.rand()
        m = 1.0 + 0.1 * np.random.rand()
        mu = 0.9 + 0.1 * np.random.rand()
        el = 0.2 + 0.1 * np.random.rand()
        comp = 0.
        (
            model,
            geom_model,
            visual_model,
            data,
            geom_data,
            visual_data,
            actuation,
        ) = addBall(model, geom_model, visual_model, actuation, a, m, mu, el, comp)
    elif obj_type == 1:
        a = 0.2 + 0.1 * np.random.rand()
        m = 1.0 + 0.1 * np.random.rand()
        mu = 0.9 + 0.1 * np.random.rand()
        el = 0.2 + 0.1 * np.random.rand()
        comp = 0.
        (
            model,
            geom_model,
            visual_model,
            data,
            geom_data,
            visual_data,
            actuation,
        ) = addCube(model, geom_model, visual_model, actuation, a, m, mu, el, comp)
    elif obj_type == 2 and False:
        mu = 0.9 + 0.1 * np.random.rand()
        el = 0.0 + 0.1 * np.random.rand()
        (
            model,
            geom_model,
            visual_model,
            data,
            geom_data,
            visual_data,
            actuation,
        ) = addSolo(model, geom_model, visual_model, actuation, mu, el, comp)
    return model, geom_model, visual_model, data, geom_data, visual_data, actuation


def create_random_scene(N=1):
    """Randomly creates a scene with various objects.

    Args:
        N (int): number of objects
    """
    a = 0.2 + 0.1 * np.random.rand()
    m = 1.0 + 0.1 * np.random.rand()
    mu = 0.9 + 0.1 * np.random.rand()
    el = 0.2 + 0.1 * np.random.rand()
    (
        model,
        geom_model,
        visual_model,
        data,
        geom_data,
        visual_data,
        actuation,
    ) = create_balls([a], [m], mu, el)
    for i in range(N - 1):
        (
            model,
            geom_model,
            visual_model,
            data,
            geom_data,
            visual_data,
            actuation,
        ) = addRandomObject(model, geom_model, visual_model, actuation)
    return model, geom_model, visual_model, data, geom_data, visual_data, actuation

def build_solo_problem(dense = False, ccp_reg = False, drag= False):
    mu = 0.9  # friction parameter
    model, geom_model, visual_model, data, geom_data, visual_data, actuation = create_solo(
        mu,
        0.,
        0.,
        True,
        True
    )
    dt = 1e-3
    Kb = 1e-4 *0. # Baumgarte
    # current state
    q = model.qinit
    v = np.zeros(model.nv)
    tau = np.zeros(model.nv)
    fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
    if drag:
        fext[1].linear[1] = 2.
        fext[1] = pin.Force(data.oMi[1].actionInverse @ fext[1])
    simulator = CCPNewtonPrimalSimulator(model, geom_model, data, geom_data, regularize = ccp_reg, warm_start=False)
    simulator.setSimulation(model, data, geom_model, geom_data)
    simulator.step(
        model, data, geom_model, geom_data, q, v, tau, fext, dt, Kb, 1 , 1e-7
    )
    Del = simulator.Del
    g = simulator.g
    mus = simulator.mus
    M = simulator.M
    J = simulator.J
    vstar = simulator.vstar
    dqf = simulator.dqf
    if dense:
        Del.evaluateDel()
        prob = ContactProblem(Del.G_, g, M, J, dqf, vstar, mus)
    else:
        prob = ContactProblem(Del, g, M, J, dqf, vstar, mus)
    return prob

def build_talos_problem(dense = False, ccp_reg = False):
    mu = 0.9  # friction parameter
    model, geom_model, visual_model, data, geom_data, visual_data, actuation = create_humanoid(
        mu,
        0.,
        0.,
        True,
        True
    )
    dt = 1e-3
    Kb = 1e-4 *0. # Baumgarte
    # current state
    q = model.qinit
    v = np.zeros(model.nv)
    tau = np.zeros(model.nv)
    fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
    simulator = CCPNewtonPrimalSimulator(model, geom_model, data, geom_data, regularize = ccp_reg, warm_start=False)
    simulator.setSimulation(model, data, geom_model, geom_data)
    simulator.step(
        model, data, geom_model, geom_data, q, v, tau, fext, dt, Kb, 1 , 1e-7
    )
    Del = simulator.Del
    g = simulator.g
    mus = simulator.mus
    M = simulator.M
    J = simulator.J
    vstar = simulator.vstar
    dqf = simulator.dqf
    if dense:
        Del.evaluateDel()
        prob = ContactProblem(Del.G_, g, M, J, dqf, vstar, mus)
    else:
        prob = ContactProblem(Del, g, M, J, dqf, vstar, mus)
    return prob

def build_allegro_hand_problem(dense = False, ccp_reg = False):
    mu = 0.9  # friction parameter
    comp = 0.
    model, geom_model, visual_model, data, geom_data, visual_data, actuation = create_hand(
        mu,
        0.,
        0.,
        False,
        True
    )
    model, geom_model, visual_model, data, geom_data, visual_data, actuation = addBall(model, geom_model, visual_model, actuation,0.1,1e-3, mu, comp)
    dt = 1e-3
    Kb = 1e-4 *0. # Baumgarte
    # current state
    q = np.array([-5.14149468e-04,  5.85722938e-04,  1.95624143e-01,  7.25611247e-01,
       -1.18835920e-01,  6.69172860e-01, -1.07582646e-01, -4.12691521e-02,
        9.50254635e-01,  8.49460586e-01,  1.11450397e+00,  2.14477709e-01,
        8.87365671e-01,  8.08683286e-01,  9.83169853e-01, -4.04056637e-02,
        9.26986004e-01,  9.44336275e-01,  9.85423927e-01, -2.96813245e-02,
        9.05701600e-01,  9.67601901e-01,  1.18992234e-02, -4.05249891e-02,
       -4.42164609e-02,  2.49516901e-01,  5.20794350e-02,  1.29576429e-01,
       -3.19854604e-02,  9.89684097e-01])
    v = np.zeros(model.nv)
    des_acc = np.zeros(model.nv)
    b = pin.rnea(model, data, q, v, des_acc)
    tau, _, _, _= np.linalg.lstsq(actuation, b)
    tau = actuation @tau
    fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
    simulator = CCPNewtonPrimalSimulator(model, geom_model, data, geom_data, regularize = ccp_reg, warm_start=False)
    simulator.setSimulation(model, data, geom_model, geom_data)
    simulator.step(
        model, data, geom_model, geom_data, q, v, tau, fext, dt, Kb, 1 , 1e-7
    )
    Del = simulator.Del
    g = simulator.g
    mus = simulator.mus
    M = simulator.M
    J = simulator.J
    vstar = simulator.vstar
    dqf = simulator.dqf
    if dense:
        Del.evaluateDel()
        prob = ContactProblem(Del.G_, g, M, J, dqf, vstar, mus)
    else:
        prob = ContactProblem(Del, g, M, J, dqf, vstar, mus)
    return prob

def build_cube_problem(dense = False, drag =False, slide = False, ccp_reg = False):
    a = 0.2  # size of cube
    m = 1.0  # mass of cube
    mu = 0.95  # friction parameter
    eps = 0.0  # elasticity
    model, geom_model, visual_model, data, geom_data, visual_data, actuation = create_cubes(
        [a], [m], mu, eps
    )
    # time steps
    dt = 1e-3

    # Physical parameters of the contact problem
    Kb = 1e-4*0.  # Baumgarte

    # initial state
    q0 = model.qinit
    v0 = np.zeros(model.nv)
    if slide:
        v0[1] = 3.
    q, v = q0.copy(), v0.copy()
    tau = np.zeros(model.nv)
    fext = [pin.Force(np.zeros(6)) for i in range(model.njoints)]
    # simulator
    simulator = CCPNewtonPrimalSimulator(model, geom_model, data, geom_data, regularize = ccp_reg, warm_start=False)
    simulator.step(
        model, data, geom_model, geom_data, q, v, tau, fext, dt, Kb, 1 , 1e-7
    )
    Del = simulator.Del
    g = simulator.g
    mus = simulator.mus
    M = simulator.M
    J = simulator.J
    vstar = simulator.vstar
    dqf = simulator.dqf
    if dense:
        Del.evaluateDel()
        prob = ContactProblem(Del.G_, g, M, J, dqf, vstar, mus)
    else:
        prob = ContactProblem(Del, g, M, J, dqf, vstar, mus)
    return prob
