import numpy as np
import pinocchio as pin
from pycontact import DelassusPinocchio

def computeContacts(model, data, geom_model, geom_data, prev_contacts:dict = {}, prev_Rs : list= []):
    nc = 0
    J, Del, R, e = None, None, None, None
    mus = []
    els = []
    comps = []
    contact_models = []
    contact_datas = []
    active_col = []
    contact_points = []
    contact_joints = []
    contact_placements = []
    for i, res in enumerate(geom_data.collisionResults):
        if res.isCollision():
            active_col += [i]
            geom_id1, geom_id2 = (
                geom_model.collisionPairs[i].first,
                geom_model.collisionPairs[i].second,
            )
            mu_i = geom_model.frictions[i]
            comp_i = geom_model.compliances[i]
            el_i = geom_model.elasticities[i]
            mus += [mu_i]
            comps += [comp_i]
            els += [el_i]
            joint_id1 = geom_model.geometryObjects[geom_id1].parentJoint
            joint_id2 = geom_model.geometryObjects[geom_id2].parentJoint
            contacts = res.getContacts()
            joint_placement_1 = data.oMi[joint_id1]
            joint_placement_2 = data.oMi[joint_id2]
            for contact in contacts:
                pos_i = contact.pos
                normal_i = contact.normal
                if i in prev_contacts.keys():
                    # prev_R = prev_contacts[i][1]
                    prev_R = prev_Rs[i]
                    warm_start = True
                else:
                    warm_start=False
                if warm_start and np.allclose(normal_i,prev_R[:,2]):
                    R_i = prev_R
                else:
                    ex_i, ey_i = complete_orthonormal_basis(contact.normal, joint_placement_1)
                    ex_i = np.expand_dims(ex_i, axis=1)
                    ey_i = np.expand_dims(ey_i, axis=1)
                    normal_i = np.expand_dims(contact.normal, axis=1)
                    R_i = np.concatenate((ex_i, ey_i, normal_i), axis=1)
                R_i1 = np.dot(joint_placement_1.rotation.T, R_i)
                R_i2 = np.dot(joint_placement_2.rotation.T, R_i)
                pos_i1 = joint_placement_1.rotation.T @ (
                    pos_i - joint_placement_1.translation
                )
                pos_i2 = joint_placement_2.rotation.T @ (
                    pos_i - joint_placement_2.translation
                )
                placement_i1 = pin.SE3(R_i1, pos_i1)
                placement_i2 = pin.SE3(R_i2, pos_i2)
                contact_model_i = pin.RigidConstraintModel(
                    pin.ContactType.CONTACT_3D,
                    model,
                    joint_id2,
                    placement_i2,
                    joint_id1,
                    placement_i1,
                    pin.ReferenceFrame.LOCAL,
                )
                contact_models += [contact_model_i]
                contact_data_i = contact_model_i.createData()
                contact_datas += [contact_data_i]
                contact_points += [pos_i]
                contact_joints += [(joint_id1, joint_id2)]
                contact_placements += [(placement_i1, placement_i2)]
                e_i = np.array([0.0, 0.0, contact.penetration_depth])
                if R is None:
                    R = R_i
                    e = e_i
                else:
                    R = np.concatenate((R, R_i), axis=1)
                    e = np.concatenate((e, e_i), axis=0)
            nc += len(contacts)
    if nc > 0:
        Del = DelassusPinocchio(model, data, contact_models, contact_datas)
        J = pin.getConstraintsJacobian(model, data, contact_models, contact_datas)
    return J, Del, data.M, R, e, mus, comps, els, active_col, contact_points, contact_joints, contact_placements

def computeAllContacts(model, data, geom_model, geom_data):
    print("WARNING: going through `computeAllContacts function!!! Should pass through `computeContacts` instead!")
    # compute the contact quantities as if every body was in contact
    # this computation is done offline and can be used to set the compliance (as it is done in MuJoCo)
    nc = 0
    Del, J, R, e = None, None, None, None
    mus = []
    els = []
    contact_models = []
    contact_datas = []
    contact_points = []
    nc = len(geom_model.collisionPairs)
    for i, col_pair in enumerate(geom_model.collisionPairs):
        geom_id1, geom_id2 = (
            col_pair.first,
            col_pair.second,
        )
        mu_i = geom_model.frictions[i]
        el_i = geom_model.elasticities[i]
        mus += [mu_i]
        els += [el_i]
        joint_id1 = geom_model.geometryObjects[geom_id1].parentJoint
        joint_id2 = geom_model.geometryObjects[geom_id2].parentJoint
        contact_pen = 0.
        joint_placement_1 = data.oMi[joint_id1]
        joint_placement_2 = data.oMi[joint_id2]
        pos_i = geom_data.oMg[geom_id2].translation # TODO: where to chose the contact point to compute J ?
        normal_i = model.gravity.vector[:3]
        normal_i = normal_i/np.linalg.norm(normal_i)
        ex_i, ey_i = complete_orthonormal_basis(normal_i, joint_placement_1)
        ex_i = np.expand_dims(ex_i, axis=1)
        ey_i = np.expand_dims(ey_i, axis=1)
        normal_i = np.expand_dims(normal_i, axis=1)
        R_i = np.concatenate((ex_i, ey_i, normal_i), axis=1)
        R_i1 = np.dot(joint_placement_1.rotation.T, R_i)
        R_i2 = np.dot(joint_placement_2.rotation.T, R_i)
        pos_i1 = joint_placement_1.rotation.T @ (
            pos_i - joint_placement_1.translation
        )
        pos_i2 = joint_placement_2.rotation.T @ (
            pos_i - joint_placement_2.translation
        )
        placement_i1 = pin.SE3(R_i1, pos_i1)
        placement_i2 = pin.SE3(R_i2, pos_i2)
        contact_model_i = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_3D,
            model,
            joint_id2,
            placement_i2,
            joint_id1,
            placement_i1,
            pin.ReferenceFrame.LOCAL,
        )
        contact_models += [contact_model_i]
        contact_data_i = contact_model_i.createData()
        contact_datas += [contact_data_i]
        contact_points += [pos_i]
        e_i = np.array([0.0, 0.0, contact_pen])
        if R is None:
            R = R_i
            e = e_i
        else:
            R = np.concatenate((R, R_i), axis=1)
            e = np.concatenate((e, e_i), axis=0)
    if nc>0:
        chol = pin.ContactCholeskyDecomposition(model, contact_models)
        chol.compute(model, data, contact_models, contact_datas, 1e-9)
        Del = chol.getInverseOperationalSpaceInertiaMatrix()
        J = chol.matrix()[: 3 * nc, 3 * nc :]
    return J, Del, R, e, mus, els, contact_points


def complete_orthonormal_basis(ez, joint_placement):
    ex = joint_placement.rotation[:,0]
    ey = np.cross(ez, ex)
    if np.linalg.norm(ey) < 1e-6:
        ex = joint_placement.rotation[:,1]
        ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, ez)
    return ex, ey

def complete_orthonormal_basis_random(ez):
    ex = np.random.rand(3)
    ey = np.cross(ez, ex)
    ey /= np.linalg.norm(ey)
    ex = np.cross(ey, ez)
    return ex, ey


def computeDFreeVelocity(model, data, q, v, tau, fext, dt):
    return dt * pin.aba(model, data, q, v, tau, fext)

def inverse_kinematics_trans(model, data, frame_translation, frame_id, eps = 1e-4, max_it = 1000):
    q = pin.neutral(model)
    DT = 1e-1
    damp = 1e-12

    i = 0
    while True:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        err = data.oMf[frame_id].translation - frame_translation
        if np.linalg.norm(err) < eps:
            success = True
            break
        if i >= max_it:
            success = False
            break
        J = pin.computeFrameJacobian(model, data, q, frame_id, pin.LOCAL)
        J_trans = np.dot(data.oMf[frame_id].rotation, J[:3])
        v = -J_trans.T.dot(
            np.linalg.solve(J_trans.dot(J_trans.T) + damp * np.eye(3), err)
        )
        q = pin.integrate(model, q, v * DT)
        i += 1
    return q
