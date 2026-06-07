"""Logistic Regression Model"""

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
    