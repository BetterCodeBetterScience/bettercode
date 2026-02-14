# examples from chatgpt

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression


def multiple_linear_regression1(X, y):
    """
    Solves a multiple linear regression problem using the normal equation.
    
    Parameters:
    X (numpy array): A 2D numpy array where each row is a sample 
      and each column is a feature.
    y (numpy array): A 1D numpy array representing the target variable 
      for each sample.
    
    Returns:
    w (numpy array): The coefficients of the linear regression model.
    """
    # Add a column of ones to X for the intercept term
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Compute the coefficients using the normal equation
    w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    return w

def multiple_linear_regression2(X, y):
    """
    Computes the coefficients for a multiple linear regression 
    using the normal equation.
    
    Parameters:
    X : numpy.ndarray
        The input feature matrix (each row is a data point, and 
        each column is a feature).
    y : numpy.ndarray
        The target output vector.

    Returns:
    theta : numpy.ndarray
        The computed coefficients (including the intercept if 
        X includes a column of ones).
    """
    # Compute the normal equation: theta = (X^T X)^(-1) X^T y
    X_transpose = np.transpose(X)
    theta = np.linalg.inv(X_transpose @ X) @ X_transpose @ y
    
    return theta


# Fixtures for test data

@pytest.fixture
def simple_linear_data():
    """Simple linear relationship: y = 2 + 3x"""
    X_features = np.array([[1], [2], [3], [4], [5]])
    X = np.c_[np.ones((X_features.shape[0], 1)), X_features]  # Add intercept column
    coefs = np.array([2, 3])
    y = X.dot(coefs)  
    return X, y, coefs


# Tests
@pytest.mark.xfail(reason="Function1 does not handle intercept term correctly")
def test_simple_linear_regression_function1(simple_linear_data):
    """Test function1 with a simple linear relationship: y = 2 + 3x"""
    X, y, coefs = simple_linear_data
    
    # Fit the model
    w = multiple_linear_regression1(X, y)
    
    # Check coefficients (intercept and slope)
    assert len(w) == 2
    assert np.isclose(w[0], coefs[0], atol=1e-10)  # intercept
    assert np.isclose(w[1], coefs[1], atol=1e-10)  # slope


def test_simple_linear_regression_function2(simple_linear_data):
    """Test function2 with a simple linear relationship: y = 2 + 3x"""
    X, y, coefs = simple_linear_data
    
    # Function2 expects X with intercept (already included in fixture)
    # Fit the model
    theta = multiple_linear_regression2(X, y)
    
    # Check coefficients (intercept and slope)
    assert len(theta) == 2
    assert np.isclose(theta[0], coefs[0], atol=1e-10)  # intercept
    assert np.isclose(theta[1], coefs[1], atol=1e-10)  # slope

def test_simple_linear_regression_function1_noint(simple_linear_data):
    """Test function1 with a simple linear relationship: y = 2 + 3x"""
    X, y, coefs = simple_linear_data
    X_noint = X[:, 1:]  # Remove intercept column for function1
    # Fit the model
    w = multiple_linear_regression1(X_noint, y)
    
    # Check coefficients (intercept and slope)
    assert len(w) == 2
    assert np.isclose(w[0], coefs[0], atol=1e-10)  # intercept
    assert np.isclose(w[1], coefs[1], atol=1e-10)  # slope

@pytest.mark.xfail(reason="Function2 does not handle missing intercept term correctly")
def test_simple_linear_regression_function2_noint(simple_linear_data):
    """Test function2 with a simple linear relationship: y = 2 + 3x"""
    X, y, coefs = simple_linear_data
    X_noint = X[:, 1:]  # Remove intercept column for function1
    
    # Function2 expects X with intercept (already included in fixture)
    # Fit the model
    theta = multiple_linear_regression2(X_noint, y)
    
    # Check coefficients (intercept and slope)
    assert len(theta) == 2
    assert np.isclose(theta[0], coefs[0], atol=1e-10)  # intercept
    assert np.isclose(theta[1], coefs[1], atol=1e-10)  # slope
