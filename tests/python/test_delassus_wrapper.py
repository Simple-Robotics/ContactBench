import example_robot_data as erd
import pinocchio as pin
import numpy as np
from pycontact import DelassusPinocchio

def create_models_and_datas():
    robot = erd.load("solo12")
    model = robot.model
    data = model.createData()
    nc = 3
    njoint = len(model.joints)
    contact_models = []
    contact_datas = []
    for i in range(nc):
        id1 = 0
        id2 = np.random.randint(1,njoint)
        placement_i1 = pin.SE3.Identity()
        placement_i2 = pin.SE3.Identity()
        contact_model_i = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_3D,
            model,
            id2,
            placement_i2,
            id1,
            placement_i1,
            pin.ReferenceFrame.LOCAL,
        )
        contact_models += [contact_model_i]
        contact_data_i = contact_model_i.createData()
        contact_datas += [contact_data_i]
    q = pin.neutral(model)
    M = pin.crba(model, data, q)
    return model, data, contact_models, contact_datas

def test_delassus_pinocchio_init():
    model, data, contact_models, contact_datas = create_models_and_datas()
    Del = DelassusPinocchio(model, data, contact_models, contact_datas)
    assert(True)

def test_delassus_pinocchio_evaluate():
    model, data, contact_models, contact_datas = create_models_and_datas()
    Del = DelassusPinocchio(model, data, contact_models, contact_datas)
    Del.computeChol(1e-6)
    Del.evaluateDel()
    G = Del.G_
    chol = pin.ContactCholeskyDecomposition(model, contact_models)
    mu = 1e-9
    chol.compute(model, data, contact_models, contact_datas, mu) #mu should be set to 0 in order to get the exact Delassus
    G_test = chol.getInverseOperationalSpaceInertiaMatrix() - mu*np.eye(3*Del.nc_)
    assert(np.allclose(G, G_test))

def test_delassus_pinocchio_applyOnTheRight():
    model, data, contact_models, contact_datas = create_models_and_datas()
    Del = DelassusPinocchio(model, data, contact_models, contact_datas)
    Del.computeChol(1e-6)
    Del.evaluateDel()
    lam = np.zeros(3*Del.nc_)
    lam_out = np.zeros(3*Del.nc_)
    Del.applyOnTheRight(lam, lam_out)
    chol = pin.ContactCholeskyDecomposition(model, contact_models)
    mu = 1e-9
    chol.compute(model, data, contact_models, contact_datas, mu) #mu should be set to 0 in order to get the exact Delassus
    G_test = chol.getInverseOperationalSpaceInertiaMatrix() - mu*np.eye(3*Del.nc_)
    lam_out2 = G_test @ lam
    assert(np.allclose(lam_out,lam_out2))

def test_delassus_pinocchio_computeChol():
    model, data, contact_models, contact_datas = create_models_and_datas()
    Del = DelassusPinocchio(model, data, contact_models, contact_datas)
    Del.computeChol(1e-9)
    assert(True)

def test_delassus_pinocchio_updateChol():
    model, data, contact_models, contact_datas = create_models_and_datas()
    Del = DelassusPinocchio(model, data, contact_models, contact_datas)
    Del.computeChol(1e-9)
    Del.updateChol(1e-7)
    assert(True)

def test_delassus_pinocchio_solve():
    model, data, contact_models, contact_datas = create_models_and_datas()
    nc = len(contact_models)
    Del = DelassusPinocchio(model, data, contact_models, contact_datas)
    Del.computeChol(1e-9)
    lam = np.zeros(3*nc)
    lam_out = np.zeros(3*nc)
    Del.solve(lam, lam_out)
    chol = pin.ContactCholeskyDecomposition(model, contact_models)
    mu = 1e-9
    chol.compute(model, data, contact_models, contact_datas, mu) #mu should be set to 0 in order to get the exact Delassus
    G_test = chol.getInverseOperationalSpaceInertiaMatrix() - mu*np.eye(3*nc)
    lam_out2 = np.linalg.inv(G_test+mu*np.eye(3*nc))@ lam
    assert(np.allclose(lam_out, lam_out2))


if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(sys.argv))
