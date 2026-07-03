import time

import pytest


@pytest.mark.parametrize("x", range(10))
def test_parallel(x):
    time.sleep(1)
    assert x in range(10), f"Value {x} is not in the expected list."
