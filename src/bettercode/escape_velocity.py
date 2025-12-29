import math
import numpy as np
import pytest

def escape_velocity(mass: float, radius: float, G=6.67430e-11):
    """
    Calculate the escape velocity from a celestial body, given its mass and radius.

    Args:
    mass (float): Mass of the celestial body in kg.
    radius (float): Radius of the celestial body in meters.

    Returns:
    float: Escape velocity in m/s.
    """
    if mass <= 0 or radius <= 0:
        raise ValueError("Mass and radius must be positive values.")
    return math.sqrt(2 * G * mass / radius)
