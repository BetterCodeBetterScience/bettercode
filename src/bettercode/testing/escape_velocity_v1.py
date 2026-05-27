import math


def escape_velocity(mass: float, radius: float, G=6.67430e-11):
    """
    Calculate the escape velocity from a celestial body, given its mass and radius.

    Args:
    mass (float): Mass of the celestial body in kg.
    radius (float): Radius of the celestial body in meters.

    Returns:
    float: Escape velocity in m/s.
    """

    return math.sqrt(2 * G * mass / radius)
