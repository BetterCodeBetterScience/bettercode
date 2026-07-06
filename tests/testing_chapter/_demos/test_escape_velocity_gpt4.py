import pytest

from bettercode.escape_velocity import escape_velocity_safe as escape_velocity


def test_escape_velocity_gpt4():

    mass_earth = 5.972e24
    radius_earth = 6.371e6
    result = escape_velocity(mass_earth, radius_earth)
    assert result == pytest.approx(11186.25, rel=1e-3)

    mass_mars = 6.4171e23
    radius_mars = 3.3895e6
    result = escape_velocity(mass_mars, radius_mars)
    assert result == pytest.approx(5027.34, rel=1e-3)

    mass_jupiter = 1.8982e27
    radius_jupiter = 6.9911e7
    result = escape_velocity(mass_jupiter, radius_jupiter)
    assert result == pytest.approx(59564.97, rel=1e-3)
