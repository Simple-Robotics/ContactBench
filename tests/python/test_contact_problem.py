from pycontact import ContactProblem, DelassusPinocchio
import pinocchio as pin
import example_robot_data as erd
import numpy as np


def test_ContactProblem():
    nc = np.random.randint(1, 10)
    A = np.random.rand(3 * nc, 3 * nc)
    b = np.random.random(3 * nc)
    mus = [0.8] * nc
    prob = ContactProblem(A, b, mus)
    for i in range(nc):
        assert prob.contact_constraints_[i].mu_ == mus[i]


def test_ContactProblem_project():
    nc = np.random.randint(1, 10)
    A = np.random.rand(3 * nc, 3 * nc)
    b = np.random.random(3 * nc)
    mus = [0.8] * nc
    prob = ContactProblem(A, b, mus)
    lam = np.random.rand(b.shape[0])
    lam_proj = np.zeros_like(lam)
    lam_proj2 = np.zeros_like(lam)
    prob.project(lam, lam_proj)
    ContactProblem.project(mus, lam, lam_proj2)
    for i in range(nc):
        lami_proj = np.zeros(3)
        prob.contact_constraints_[i].project(lam[3*i:3*i+3], lami_proj)
        assert (lami_proj == lam_proj[3*i:3*i+3]).all()
        assert (lami_proj == lam_proj2[3*i:3*i+3]).all()

def test_ContactProblem_projectDual():
    nc = np.random.randint(1, 10)
    A = np.random.rand(3 * nc, 3 * nc)
    b = np.random.random(3 * nc)
    mus = [0.8] * nc
    prob = ContactProblem(A, b, mus)
    lam = np.random.rand(b.shape[0])
    lam_proj = np.zeros_like(lam)
    prob.projectDual(lam, lam_proj)
    for i in range(nc):
        lami_proj = np.zeros(3)
        prob.contact_constraints_[i].projectDual(lam[3*i:3*i+3], lami_proj)
        assert (lami_proj == lam_proj[3*i:3*i+3]).all()


def test_ContactProblem_computeContactComplementarity():
    nc = np.random.randint(1, 10)
    A = np.random.rand(3 * nc, 3 * nc)
    b = np.random.random(3 * nc)
    mus = [0.8] * nc
    prob = ContactProblem(A, b, mus)
    lam = np.random.random(3 * nc)
    prob.computeContactComplementarity(lam)
    assert True


def test_ContactProblem_computeContactComplementarity2():
    nc = np.random.randint(1, 10)
    A = np.random.rand(3 * nc, 3 * nc)
    b = np.random.random(3 * nc)
    mus = [0.8] * nc
    prob = ContactProblem(A, b, mus)
    lam = np.random.random(3 * nc)
    v = np.random.random(3 * nc)
    prob.computeContactComplementarity(lam, v)
    assert True


def test_ContactProblem_computeSignoriniComplementarity():
    nc = np.random.randint(1, 10)
    A = np.random.rand(3 * nc, 3 * nc)
    b = np.random.random(3 * nc)
    mus = [0.8] * nc
    prob = ContactProblem(A, b, mus)
    lam = np.random.random(3 * nc)
    prob.computeSignoriniComplementarity(lam)
    assert True


def test_ContactProblem_computeSignoriniComplementarity2():
    nc = np.random.randint(1, 10)
    A = np.random.rand(3 * nc, 3 * nc)
    b = np.random.random(3 * nc)
    mus = [0.8] * nc
    prob = ContactProblem(A, b, mus)
    lam = np.random.random(3 * nc)
    v = np.random.random(3 * nc)
    prob.computeSignoriniComplementarity(lam, v)
    assert True


def test_ContactProblem_computeConicComplementarity():
    nc = np.random.randint(1, 10)
    A = np.random.rand(3 * nc, 3 * nc)
    b = np.random.random(3 * nc)
    mus = [0.8] * nc
    prob = ContactProblem(A, b, mus)
    lam = np.random.random(3 * nc)
    prob.computeConicComplementarity(lam)
    assert True


def test_ContactProblem_computeConicComplementarity2():
    nc = np.random.randint(1, 10)
    A = np.random.rand(3 * nc, 3 * nc)
    b = np.random.random(3 * nc)
    mus = [0.8] * nc
    prob = ContactProblem(A, b, mus)
    lam = np.random.random(3 * nc)
    v = np.random.random(3 * nc)
    prob.computeConicComplementarity(lam, v)
    assert True


def test_ContactProblem_computePerContactContactComplementarity():
    nc = np.random.randint(1, 10)
    A = np.random.rand(3 * nc, 3 * nc)
    b = np.random.random(3 * nc)
    mus = [0.8] * nc
    prob = ContactProblem(A, b, mus)
    i = np.random.randint(0, nc)
    lam = np.random.random(3 * nc)
    v = A[3 * i : 3 * (i + 1)] @ lam + b[3 * i : 3 * (i + 1)]
    prob.computePerContactContactComplementarity(i, lam[3 * i : 3 * (i + 1)], v)
    assert True


def test_ContactProblem_computePerContactSignoriniComplementarity():
    nc = np.random.randint(1, 10)
    A = np.random.rand(3 * nc, 3 * nc)
    b = np.random.random(3 * nc)
    mus = [0.8] * nc
    prob = ContactProblem(A, b, mus)
    i = np.random.randint(0, nc)
    lam = np.random.random(3 * nc)
    v = A[3 * i : 3 * (i + 1)] @ lam + b[3 * i : 3 * (i + 1)]
    prob.computePerContactSignoriniComplementarity(i, lam[3 * i : 3 * (i + 1)], v)
    assert True


def test_ContactProblem_computePerContactConicComplementarity():
    nc = np.random.randint(1, 10)
    A = np.random.rand(3 * nc, 3 * nc)
    b = np.random.random(3 * nc)
    mus = [0.8] * nc
    prob = ContactProblem(A, b, mus)
    i = np.random.randint(0, nc)
    lam = np.random.random(3 * nc)
    v = A[3 * i : 3 * (i + 1)] @ lam + b[3 * i : 3 * (i + 1)]
    prob.computePerContactConicComplementarity(i, lam[3 * i : 3 * (i + 1)], v)
    assert True


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

def test_constructor_delassus():
    model, data, contact_models, contact_datas = create_models_and_datas()
    Del = DelassusPinocchio(model, data, contact_models, contact_datas)
    nc = Del.nc_
    g = np.zeros(3*nc)
    mus = [0.9]*nc
    prob = ContactProblem(Del,g,mus)


if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(sys.argv))
