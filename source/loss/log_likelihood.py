"""Log-likelihood loss function - commonly used in logistic regression."""
import numpy as np
import numpy.typing as npt


def sigmoid(Z: npt.NDArray) -> npt.NDArray:
    """Calculate the sigmoid function for a given input array.
    
    Parameters:
    -----
    Z (np.array): - (m, n) an input array of the net inputs (with m features and n data points)
    - which is a linear combination of the weights and the features.

    Returns:
    -----
    np.array : (n,) an array of sigmoid outputs
    """
    return 1.0 / (1.0 + np.exp(-Z))


def log_likelihood(sigma_Z: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    """Calculate the log-likelihood for the given data.

    Parameters:
    -----
    sigma_Z (np.array): - (n,) an array of sigmoid activation function values for n data points
    y (np.array): - (n,) prediction target values (0 or 1) for n data points

    Returns:
    -----
    (np.array): - (n,) an array of log likelihood values for n data points
    """
    return (y * np.log(sigma_Z)) + ((1 - y) * (np.log(1 - sigma_Z)))
    
