# custom scaler class for testing example

import numpy as np
from numpy.typing import NDArray

class SimpleScaler:
    """Simple scaler for standardizing data using z-score normalization.
    
    Attributes
    ----------
    mean_ : NDArray
        Mean values for each feature
    std_ : NDArray
        Standard deviation values for each feature
    transformed_ : NDArray or None
        Last transformed data
    """
    
    def __init__(self) -> None:
        """Initialize SimpleScaler with None values."""
        self.transformed_: NDArray | None = None
        self.mean_: NDArray | None = None
        self.std_: NDArray | None = None

    def fit(self, X: NDArray) -> None:
        """Compute the mean and standard deviation for later scaling.
        
        Parameters
        ----------
        X : NDArray
            Training data of shape (n_samples, n_features)
        """
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

    def transform(self, X: NDArray) -> NDArray:
        """Perform standardization by centering and scaling.
        
        Parameters
        ----------
        X : NDArray
            Data to transform of shape (n_samples, n_features)
        
        Returns
        -------
        NDArray
            Transformed data with zero mean and unit variance
        """
        self.transformed_ = (X - self.mean_) / self.std_
        return self.transformed_

    def fit_transform(self, X: NDArray) -> NDArray:
        """Fit to data, then transform it.
        
        Parameters
        ----------
        X : NDArray
            Training data of shape (n_samples, n_features)
        
        Returns
        -------
        NDArray
            Transformed data with zero mean and unit variance
        """
        self.fit(X)
        return self.transform(X)

