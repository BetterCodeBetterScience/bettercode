import numpy as np

from bettercode.testing.escape_velocity_v1 import escape_velocity


def test_escape_velocity():
    """
    Test the escape_velocity function with known values.
    Note: ev_expected has a typo (1186.0 instead of 11186.0) — intentionally failing demo.
    """
    mass_earth = 5.972e24  # Earth mass in kg
    radius_earth = 6.371e6  # Earth radius in meters
    ev_expected = 1186.0  # typo: should be 11186.0
    ev_computed = escape_velocity(mass_earth, radius_earth)
    assert np.allclose(ev_expected, ev_computed), "Test failed!"
