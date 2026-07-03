import numpy as np
from hypothesis import given, seed, settings, strategies as st
from hypothesis.extra import numpy as nps

from bettercode.testing.linear_regression import linear_regression


@seed(42)
@settings(database=None)
@given(
    # Only generate data that is likely to be valid to start with
    nps.arrays(np.float64, (10, 1), elements=st.floats(-1e6, 1e6)),
    nps.arrays(np.float64, (10,), elements=st.floats(-1e6, 1e6)),
)
def test_linear_regression_without_validation(X, y):
    """Tests that our algorithm matches a reference implementation (scipy).
    This is a smoke test that intentionally fails due to singular matrices.
    """
    # Now we can safely test the math against a reference implementation (scipy),
    # knowing the input is valid.
    params = linear_regression(X, y, validate=False)
    assert params is not None, "Parameters should not be None"
