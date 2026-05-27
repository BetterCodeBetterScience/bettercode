import pytest
from time import sleep


def test_duration_3():
    sleep(3)
    assert True


def test_duration_5():
    sleep(5)
    assert True


def test_duration_1():
    sleep(1)
    assert True
