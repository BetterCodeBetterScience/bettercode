import numpy as np


class SimpleScaler:
    """Standardize features by removing the mean and scaling to unit variance."""

    def __init__(self) -> None:
        self.transformed_ = None

    def fit(self, X: np.ndarray) -> None:
        """Compute the per-feature mean and standard deviation from ``X``."""
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize ``X`` using the fitted mean and standard deviation."""
        self.transformed_ = (X - self.mean_) / self.std_
        return self.transformed_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to ``X`` and return the standardized data."""
        self.fit(X)
        return self.transform(X)
