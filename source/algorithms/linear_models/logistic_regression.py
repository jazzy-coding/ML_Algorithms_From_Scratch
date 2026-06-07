"""Logistic Regression Model"""
from typing import Optional, Callable, List

import numpy as np
import numpy.typing as npt

from source.loss.log_likelihood import log_likelihood

class LogisticRegression:
    """Class that implements the Logistic Regression model.

    Logistic Regression is typically used for binary classification 
    or probabilistic prediction problems. The model can output predicted
    classes or predicted probabilities.

    IMPORTANT: Predicted probabilities are NOT calibrated automatically!
    Perform calibration on probabilities before basing decisions on them.

    Assumptions:
    -----
    1. Logistic Regression assumes the data points are independent of each other.
    (This explains the multiplication in the MLE formula.)

    2. Logistic Regression assumes that there is a linear relationship between
    the log odds and the net input.

    Model:
    -----
    sigma(Z) = 1 / (1 + e^-Z) - the sigmoid function

    Where:
    -----
    Z = (n,) W.T @ X + b (a linear combination of the weights and the input features plus bias term)
    W = (m,) the weight vector for m features
    X = (m, n) the feature/input matrix of m features and n data points
    b = (1,) the bias term

    Decision Boundary:
    -----
    The decision boundary represents the threshold at which the model changes its class prediction.
    For example, if the decision boundary is set as 0.5 - then if P(Y = y | X = x) > 0.5, class '1'
    will be predicted. This value can be altered to maximise certain metrics like precision
    and recall if needed.

    Methods:
    -----
    net_input

    fit

    predict

    Attributes:
    -----
    coefficients_

    intercept_

    learning_rate

    epochs

    loss

    random_state
    """
    def __init__(
        self,
        learning_rate: Optional[float] = 0.01,
        epochs: Optional[int] = 200,
        loss: Callable = log_likelihood,
        random_state: Optional[int] = 42
    ):
        self.coefficients_ = np.array([])
        self.intercept_ = np.array([])

        self.learning_rate_ = learning_rate
        self.epochs_ = epochs
        self.loss_ = loss
        self.random_state_ = random_state

        self.losses_: List[float] = []


    def net_input(self, X: npt.NDArray) -> npt.NDArray:
        """Calculates the net input (W.T @ X + b) to the sigmoidal activation function.
        
        Parameters:
        -----
        X (np.array): - (m, n) a feature matrix of m features and n data points

        Returns:
        -----
        np.array: - (n,) a column vector of net inputs from n data points
        """
        return self.coefficients_.dot(X.T) + self.intercept_

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> None:
        """Fits the Logistic Regression model using standard gradient descent methods.
        
        Parameters:
        -----
        X (np.array): - (m, n) an input feature matrix of m features and n data points 
        y (np.array): - (n,) a column vector of target values of length n data points

        Returns:
        -----
        None
        """
        # reshape arrays for processing

        # FOR EACH ITERATION:

        # calculate net input
        # calculate current loss and append
        # update coefficients and intercept term
        # repeat

    def predict(self, X: npt.NDArray) -> npt.NDArray:
        pass
    