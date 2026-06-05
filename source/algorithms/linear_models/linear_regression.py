"""Simple Linear Regression Model"""
from typing import Callable, Optional
import numpy as np
from source.loss.mean_squared_error import mean_squared_error

class SimpleLinearRegression:
    """Simple Linear Regression Model

    Model
    -----
    y_hat    = (m * x) + c

    where
    -----
    y_hat    = (1,): scalar prediction target variable (float)
    m        = (1,): scalar gradient of the linear model (float)
    x        = (1,): scalar predictor variable (float)
    c        = (1,): scalar intercept of the linear model (float)

    1-dimensional linear regression is used when we have just 1 predictor
    in a regression problem. This method can only model linear relationships
    so it is best to do some exploratory data analysis first to make sure that
    the relationship between your predictor and your target is indeed approximately
    linear. This also forms the base for n-dimensional linear regression.

    Linear regression can also be thought of as the expected value of y given X - E(Y|X)

    Parameters:
    -----

    learning_rate (float - defaults to 0.01): the learning rate indicates how quickly
    model parameters get updated to convergence to a local extrema in relation to the 
    selected loss function. Set the learning rate too low and model convergence will 
    be slow. Set it too high and the model may not converge to a local extrema.

    epochs (int - defaults to 200): epochs represents the maximum number of gradient 
    descent iterations that will take place before the model training process exits.

    Attributes:
    -----
    
    coefficient_: np.array (1,) - representing 'm' in the above model formula.

    intercept_: np.array (1,) - representing 'c' in the above model formula.

    epochs_: int - the number of training epochs.

    learning_rate_: float - the learning rate of the gradient descent algorithm.

    loss_function_: Callable - the loss function used to evaluate the model.
    """

    def __init__(
        self,
        learning_rate: Optional[float] = 0.01,
        epochs: int = 200,
        loss: Callable = mean_squared_error
    ) -> None:
        self.coefficient_ = None
        self.intercept_ = None
        self.epochs_ = epochs
        self.learning_rate_ = learning_rate
        self.loss_ = loss

    def _net_input(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

    