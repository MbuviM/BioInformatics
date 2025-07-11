"""
Linear Regression Using Gradient Descent

Write a Python function that performs linear regression using gradient descent. 
The function should take NumPy arrays X (features with a column of ones for the intercept) and y (target) as input, 
along with learning rate alpha and the number of iterations, and return the coefficients of the linear regression model as a NumPy array. 
Round your answer to four decimal places. -0.0 is a valid result for rounding a very small number.

"""

import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	# Your code here, make sure to round
	m, n = X.shape
	theta = np.zeros((n, 1))
	
	for iteration in range (iterations):
		# Find the predicted value
		# Broadcasting the shape of y
		if y.ndim == 1:
			y = y.reshape(-1, 1) # -1,is used to show that Numpy should figure out the rows
			
		# Find the predicted values
		y_hat = X @ theta
		
		# Calculate the gradient function
		gradient_function = 2/len(X) * np.dot(X.T, (y_hat - y))
		
        # Calculate the updated parameters
		theta = theta - alpha * gradient_function
		
	return np.round(theta, 4)
