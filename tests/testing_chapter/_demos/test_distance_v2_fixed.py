import math

from bettercode.testing.distance import distance


def test_distance_zero():
    assert distance((0, 0), (0, 0)) == 0


def test_distance_positive_coordinates():
    assert distance((1, 2), (4, 6)) == 5


def test_distance_negative_coordinates():
    assert distance((-1, -2), (-4, -6)) == 5


def test_distance_mixed_coordinates():
    # Fixed expected value: sqrt(89) not sqrt(125)
    assert distance((1, -2), (-4, 6)) == math.sqrt(89)


def test_distance_same_x():
    assert distance((3, 4), (3, 8)) == 4


def test_distance_same_y():
    assert distance((3, 4), (7, 4)) == 4
