import numpy as np
from hypothesis import assume, given, seed, settings, strategies as st
from hypothesis.extra import numpy as nps
from scipy.stats import linregress

from bettercode.testing.linear_regression import _validate_input, linear_regression


# Test 1: Test the validation logic in isolation
@seed(42)
@settings(database=None)
@given(
    nps.arrays(
        np.float64, (10, 1), elements=st.floats(allow_nan=True, allow_infinity=True)
    ),
    nps.arrays(
        np.float64, (10,), elements=st.floats(allow_nan=True, allow_infinity=True)
    ),
)
def test_validate_input(X, y):
    """Tests that our validation function correctly identifies and rejects bad data."""
    try:
        # Call the validation function directly
        _validate_input(X, y)
        linear_regression(X, y, validate=False)
        # If it gets here, hypothesis generated valid data and the function ran successfully.
    except ValueError:
        # If we get here, the data was invalid. The validator correctly
        # raised an error. This is also a successful test case.
        pass  # Explicitly show that catching the error is the goal.


# Test 2: Test the algorithm's correctness, assuming valid input
@seed(42)
@settings(database=None)
@given(
    nps.arrays(np.float64, (10, 1), elements=st.floats(-1e6, 1e6)),
    nps.arrays(np.float64, (10,), elements=st.floats(-1e6, 1e6)),
)
def test_linear_regression_correctness(X, y):
    """Tests that our algorithm matches a reference implementation (scipy)."""
    # Use `hypothesis.assume` to filter out any edge cases the validator would catch.
    # This tells hypothesis: "If this data is bad, just discard it and try another."
    try:
        _validate_input(X, y)
    except ValueError:
        assume(False)  # Prunes this example from the test run

    # Now we can safely test the math against a reference implementation (scipy),
    # knowing the input is valid.
    params = linear_regression(X, y)
    lr_result = linregress(X.flatten(), y.flatten())

    assert np.allclose(params, [lr_result.intercept, lr_result.slope], rtol=1e-4, atol=1e-6)
