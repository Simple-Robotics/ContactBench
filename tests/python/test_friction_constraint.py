from pycontact import IceCreamCone, PyramidCone
import numpy as np

def test_ICC_init():
    cone = IceCreamCone(0.8)
    assert True


def test_ICC_project():
    cone = IceCreamCone(0.8)
    lam = np.ones(3)
    lam_out = np.zeros(3)
    cone.project(lam, lam_out)
    lam2 = lam.copy()
    lam_out2 = np.zeros(3)
    IceCreamCone.project(.8,lam2,lam_out2)
    assert np.allclose(lam_out, lam_out2)

def test_ICC_projectDual():
    mu = 0.8
    cone = IceCreamCone(mu)
    lam = np.zeros(3)
    lam[0], lam[2] = 1e3, 1.0
    lam_out = np.zeros(3)
    cone.projectDual(lam, lam_out)
    assert lam_out[0] == lam_out[2] * 1.0 / mu


def test_ICC_isInside():
    cone = IceCreamCone(0.8)
    lam = np.zeros(3)
    lam[2] = 1.
    isin = cone.isInside(lam,1e-4)
    isin2 = IceCreamCone.isInside(.8,lam, 1e-4)
    assert isin and isin2
    lam[0] = 2.
    isin = cone.isInside(lam,1e-4)
    isin2 = IceCreamCone.isInside(.8, lam,1e-4)
    assert (not isin) and (not isin2)

def test_ICC_contactComplementarity():
    cone = IceCreamCone(0.8)
    lam = np.ones(3)
    v = np.zeros(3)
    comp = cone.computeContactComplementarity(lam, v)
    assert True


def test_ICC_conicComplementarity():
    cone = IceCreamCone(0.8)
    lam = np.ones(3)
    v = np.zeros(3)
    comp = cone.computeConicComplementarity(lam, v)
    assert True


def test_ICC_signoriniComplementarity():
    cone = IceCreamCone(0.8)
    lam = np.ones(3)
    v = np.zeros(3)
    comp = cone.computeSignoriniComplementarity(lam, v)
    assert True


def test_PC_init():
    cone = PyramidCone(0.8)
    assert True


def test_PC_project_horizontal():
    cone = PyramidCone(0.8)
    lam = np.ones(3)
    lam_out = np.zeros(3)
    cone.projectHorizontal(lam, lam_out)
    assert lam_out[0] == 0.8 and lam_out[1] == 0.8


if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main(sys.argv))
