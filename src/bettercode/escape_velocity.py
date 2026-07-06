import math


def escape_velocity(mass: float, radius: float, G: float = 6.67430e-11) -> float:
    """Calculate the escape velocity from a celestial body.

    Args:
        mass: Mass of the celestial body in kg.
        radius: Radius of the celestial body in meters.
        G: Gravitational constant.

    Returns:
        Escape velocity in m/s.

    Raises:
        ValueError: If mass or radius is not positive.
    """
    return math.sqrt(2 * G * mass / radius)


def escape_velocity_safe(mass: float, radius: float, G: float = 6.67430e-11) -> float:
    """Calculate the escape velocity from a celestial body.

    Args:
        mass: Mass of the celestial body in kg.
        radius: Radius of the celestial body in meters.
        G: Gravitational constant.

    Returns:
        Escape velocity in m/s.

    Raises:
        ValueError: If mass or radius is not positive.
    """
    if mass <= 0 or radius <= 0:
        raise ValueError("Mass and radius must be positive values.")
    return math.sqrt(2 * G * mass / radius)
