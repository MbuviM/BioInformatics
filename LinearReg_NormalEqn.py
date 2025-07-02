""""
Write a Python function that performs linear regression using the normal equation. 
The function should take a matrix X (features) and a vector y (target) as input, and return the coefficients of the linear regression model. 
Round your answer to four decimal places, -0.0 is a valid result for rounding a very small number.
"""

import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
    X = np.array(X)
    y = np.array(y)
    # Calculate theta using the normal equation: theta = (X^T * X)^(-1) * X^T * y
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    # Round the coefficients to four decimal places
    theta = np.round(theta, 4).tolist()
    return theta