# generate a function that calculates the distance between two points
# where each point is defined as a tuple of two numbers

import math

def distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two points.
    
    Parameters
    ----------
    p1 : tuple[float, float]
        First point as (x, y) coordinates
    p2 : tuple[float, float]
        Second point as (x, y) coordinates
    
    Returns
    -------
    float
        Euclidean distance between the two points
    """
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

