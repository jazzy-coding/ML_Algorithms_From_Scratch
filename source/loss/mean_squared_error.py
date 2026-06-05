"""Mean Squared Error Loss Function

Formula
------
MSE   = (1 / n) * sum(y - y_hat)**2
OR
MSE   = (1 / n) * SSE 

where
------
MSE   = (1,): the scalar mean squared error
SSE   = (1,): the scalar sum of squared errors [sum(y - y_hat)**2]
n     = (1,): the number of data points
y     = (n,): the column vector of size n representing all target values
y_hat = (n,): the column vector of size n representing all predicted target values

Mean squared error is a common loss function in regression modelling.
It represents the average (mean) squared distance between the predicted target
values in a dataset, and the actual target values in a dataset. For this loss function,
the lower the value is, the better. A mean squared error value of 0.0 would mean
that the predicted values match the actual values. A perfect fit! Because the error
is squared, predictions that are far from the target value get heavily penalised.
"""

import numpy as np

from source.exceptions.DimensionMismatchError import DimensionMismatchError

def mean_squared_error(y: np.array, y_hat: np.array) -> np.array:
    """
    Calculate the mean squared error for vectors y and y_hat.

    Parameters:
    -----------
    y (np.array): (n,) column vector of size n representing actual prediction target values
    y_hat (np.array): (n,) column vector of size n representing predicted prediction target values

    Returns:
    -----------
    (np.array): (1,) a scalar mean squared error value

    Raises:
    -----------
    (DimensionMismatchError): if y and y_hat have different dimensions, this error gets raised
    """
    # check dims first
    if y.shape != y_hat.shape:
        raise DimensionMismatchError("y.shape and y_hat.shape must be equal")
    
    # calculate error
    n = y.shape[0]
    error = (1 / n) * sum(y - y_hat)**2

    return np.array([error])