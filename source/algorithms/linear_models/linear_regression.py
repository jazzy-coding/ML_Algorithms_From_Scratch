"""Simple Linear Regression Model"""
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt

from source.loss.mean_squared_error import mean_squared_error

class LinearRegression:
    """Linear Regression Model

    Model
    -----
    y_hat    = (a * X) + b

    where
    -----
    y_hat    = (n,): column vector of predicted targets (np.array)
    a        = (m,): gradient vector of the coefficients in the linear model (np.array)
    X        = (m, n): the feature matrix - m predictors to n data points (np.array)
    b        = (1,): scalar intercept of the linear model (np.array)

    n-dimensional linear regression is used when we have n predictors
    in a regression problem. This method can only model linear relationships
    so it is best to do some exploratory data analysis first to make sure that
    the relationship between your predictor and your target is indeed approximately
    linear. This also forms the base for polynomial regression, GLMs and so on.

    Linear regression can also be thought of as predicting the expected value of y given X - E(Y|X).

    Parameters:
    -----

    learning_rate (float - defaults to 0.01): the learning rate indicates how quickly
    model parameters get updated to convergence to a local extrema in relation to the 
    selected loss function. Set the learning rate too low and model convergence will 
    be slow. Set it too high and the model may not converge to a local extrema.

    epochs (int - defaults to 200): epochs represents the maximum number of gradient 
    descent iterations that will take place before the model training process exits.

    random_state (int - defaults to 42): a random seed for experimental reproducibility

    Attributes:
    -----
    
    coefficients_: np.array (m,) - representing 'a' in the above model formula.

    intercept_: np.array (1,) - representing 'b' in the above model formula.

    epochs_: int - the number of training epochs.

    learning_rate_: float - the learning rate of the gradient descent algorithm.

    loss_function_: Callable - the loss function used to evaluate the model.

    random_state_: int - reproducibility seed
    """

    def __init__(
        self,
        learning_rate: Optional[float] = 0.01,
        epochs: int = 200,
        loss: Callable = mean_squared_error,
        random_state: Optional[int] = 42
    ) -> None:
        self.coefficients_ = None
        self.intercept_ = None
        self.epochs_ = epochs
        self.learning_rate_ = learning_rate
        self.loss_ = loss
        self.random_state_ = random_state

    def _net_input(self):
        pass

    def fit(self, X: npt.NDArray, y: npt.NDArray) -> None:
        """Fit a simple linear regression model with gradient descent.

        NOTE: This only works with the mean squared error loss function at the moment.
        Implementing derivatives soon!

        Parameters:
        -----
        X (np.array): - (m, n) feature/predictor matrix of m features and n data points
        y (np.array): - (n,) target value column vector of n data points

        Returns:
        -----
        None
        """
        print(f"DEBUG: - X.shape = {X.shape}")
        print(f"DEBUG: - y.shape = {y.shape}")

        # reshape arrays for processing
        X = X.reshape(-1, 1)
        y = y.reshape(-1, 1)

        # initialise weights and biases randomly - drawing from a standard Gaussian distribution
        self.coefficients_ = np.random.normal(loc=0.0, scale=1.0, size=X.shape[1])
        self.intercept_ = np.array([0.])

        for epoch in range(self.epochs_):
            # 1. make predictions using current weights
            preds = self.coefficients_ * X + self.intercept_
            print(f"DEBUG: - Preds shape: {preds.shape}")

            # 2. compute the loss function with current weights and bias term
            error = self.loss_(y=y, y_hat=preds)
            
            # 3. update the weights using gradient descent
            self.coefficients_ += (self.learning_rate_ * -2.0 * X.T.dot(error.T)) / X.shape[0]
            self.intercept_ += (self.learning_rate_ * -2.0 * X.T.dot(error.T))

            # 4. log the epoch
            print(f"DEBUG: - Epoch: {epoch} - Loss: {error}")

        print("INFO: - Model fitting complete.")

    def predict(self):
        pass

    