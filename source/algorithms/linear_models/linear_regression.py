"""Simple 1-dimensional Linear Regression Model

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
"""
