"""
Write a Python function that computes the output of the sigmoid activation function given an input value z. 
The function should return the output rounded to four decimal places.
"""
from math import exp
import numpy as np
def sigmoid(z: float) -> float:
	z = np.array(z)  # Ensure z is a NumPy array for vectorized operations
	result = np.round(1/(1 + exp(-z)), 4)  # Use expit for the sigmoid function and round to four decimal places
	return result

# Example usage
if __name__ == "__main__":
    z = 0
    output = sigmoid(z)
    print(f"Sigmoid output for {z}: {output}")