import numpy as np
import pytest

from bettercode.escape_velocity import (
    escape_velocity,
    escape_velocity_safe,
)


def test_escape_velocity():
    """
    Test the escape_velocity function with known values.
    """
    mass_earth = 5.972e24  # Earth mass in kg
    radius_earth = 6.371e6  # Earth radius in meters
    ev_expected = 11186.0  # Expected escape velocity for Earth in m/s
    ev_computed = escape_velocity(mass_earth, radius_earth)
    assert np.allclose(ev_expected, ev_computed), "Test failed!"


def test_escape_velocity_negative():
    """
    Make sure the function raises ValueError for negative mass or radius.
    """
    with pytest.raises(ValueError):
        escape_velocity_safe(-5.972e24, 6.371e6)
