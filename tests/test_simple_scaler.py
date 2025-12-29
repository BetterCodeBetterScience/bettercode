from bettercode.simpleScaler import SimpleScaler
import numpy as np


def test_simple_scaler_internals():

    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = SimpleScaler()
    _ = scaler.fit_transform(X)
    
    # Test that the transformed data is correct using the internal
    assert np.allclose(scaler.transformed_.mean(axis=0), np.array([0, 0]))
    assert np.allclose(scaler.transformed_.std(axis=0), np.array([1, 1]))


def test_simple_scaler_interface():
    X = np.array([[1, 2], [3, 4], [5, 6]])
    scaler = SimpleScaler()
    
    # Test the interface without accessing internals
    transformed_X = scaler.fit_transform(X)
    assert np.allclose(transformed_X.mean(axis=0), np.array([0, 0]))
    assert np.allclose(transformed_X.std(axis=0), np.array([1, 1]))