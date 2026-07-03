import numpy as np


def linear_regression_verbose(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit ordinary least squares regression using the normal equation."""
    # Add a column of ones to the input data to account for the intercept term
    X_with_intercept = np.c_[np.ones(X.shape[0]), X]

    # Compute the parameters using the normal equation
    X_transpose = X_with_intercept.T
    X_transpose_X = X_transpose @ X_with_intercept
    X_transpose_y = X_transpose @ y
    beta = np.linalg.inv(X_transpose_X) @ X_transpose_y

    return beta


def _validate_input(X: np.ndarray, y: np.ndarray) -> None:
    """Raise ValueError if the regression inputs are degenerate or non-finite."""
    if np.isinf(X).any() or np.isinf(y).any():
        raise ValueError("Input data contains infinite values")
    if np.isnan(X).any() or np.isnan(y).any():
        raise ValueError("Input data contains NaN values")
    if len(np.unique(X)) < 2 or len(np.unique(y)) < 2:
        raise ValueError("Input data must have at least 2 unique values")

    X_with_intercept = np.c_[np.ones(X.shape[0]), X]
    if np.linalg.matrix_rank(X_with_intercept) < X_with_intercept.shape[1]:
        raise ValueError("Input data is not full rank")


def linear_regression(
    X: np.ndarray, y: np.ndarray, validate: bool = True
) -> np.ndarray:
    """Fit ordinary least squares regression, optionally validating inputs."""
    if validate:
        _validate_input(X, y)

    X = np.c_[np.ones(X.shape[0]), X]
    return np.linalg.inv(X.T @ X) @ X.T @ y
