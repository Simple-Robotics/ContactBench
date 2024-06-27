import hppfcl
import numpy as np
import pinocchio as pin


def test_normal_hppfcl():
    n = np.array([0, 0, 1])
    d = 0
    plane = hppfcl.Halfspace(n, d)
    T1 = hppfcl.Transform3f.Identity()

    r = 0.5
    ellipsoid = hppfcl.Ellipsoid(np.ones(3) * r)

    req = hppfcl.CollisionRequest()
    res = hppfcl.CollisionResult()
    T2 = hppfcl.Transform3f(np.eye(3), np.array([0, 0, 0.25]))

    col = hppfcl.collide(plane, T1, ellipsoid, T2, req, res)
    assert col
    if col:
        c = res.getContacts()[0]
        assert np.allclose(c.normal, n, 1e-3)


def test_normal_pin():
    n = np.array([0, 0, 1])
    d = 0
    rmodel = pin.Model()
    rdata = rmodel.createData()

    geom_model = pin.GeometryModel()
    plane_shape = hppfcl.Halfspace(n, d)
    T1 = pin.SE3(np.eye(3), np.zeros(3))
    plane = pin.GeometryObject("plane", 0, 0, T1, plane_shape)
    plane_id = geom_model.addGeometryObject(plane)
    r = 0.5
    ellipsoid_shape = hppfcl.Ellipsoid(np.ones(3) * r)
    T2 = pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.25]))
    geometry = pin.GeometryObject("ellipsoid", 0, 0, T2, ellipsoid_shape)
    geom_id = geom_model.addGeometryObject(geometry)

    ell_plane = pin.CollisionPair(plane_id, geom_id)
    geom_model.addCollisionPair(ell_plane)
    geom_data = geom_model.createData()

    q = pin.neutral(rmodel)
    pin.updateGeometryPlacements(rmodel, rdata, geom_model, geom_data, q)
    pin.computeCollisions(geom_model, geom_data, False)
    res = geom_data.collisionResults[0]
    assert res.isCollision()
    contacts = res.getContacts()
    contact = contacts[0]
    normal = contact.normal
    assert np.allclose(normal, n, 1e-3)

def test_normal_pin_inverse():
    n = np.array([0, 0, 1])
    d = 0
    rmodel = pin.Model()
    rdata = rmodel.createData()

    geom_model = pin.GeometryModel()
    plane_shape = hppfcl.Halfspace(n, d)
    T1 = pin.SE3(np.eye(3), np.zeros(3))
    plane = pin.GeometryObject("plane", 0, 0, T1, plane_shape)
    plane_id = geom_model.addGeometryObject(plane)
    r = 0.5
    ellipsoid_shape = hppfcl.Ellipsoid(np.ones(3) * r)
    T2 = pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.25]))
    geometry = pin.GeometryObject("ellipsoid", 0, 0, T2, ellipsoid_shape)
    geom_id = geom_model.addGeometryObject(geometry)

    ell_plane = pin.CollisionPair(geom_id, plane_id)
    geom_model.addCollisionPair(ell_plane)
    geom_data = geom_model.createData()

    q = pin.neutral(rmodel)
    pin.updateGeometryPlacements(rmodel, rdata, geom_model, geom_data, q)
    pin.computeCollisions(geom_model, geom_data, False)
    res = geom_data.collisionResults[0]  # get collision pair results
    assert res.isCollision()
    contacts = res.getContacts()
    contact = contacts[0]
    normal = contact.normal
    assert np.allclose(normal, -n, 1e-3)


if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(sys.argv))
