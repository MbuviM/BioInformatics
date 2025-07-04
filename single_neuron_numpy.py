"""
Write a Python function that simulates a single neuron with a sigmoid activation function for binary classification, handling multidimensional input features. 
The function should take a list of feature vectors (each vector representing multiple features for an example), associated true binary labels, and the neuron's weights (one for each feature) and bias as input. 
It should return the predicted probabilities after sigmoid activation and the mean squared error between the predicted probabilities and the true labels, both rounded to four decimal places.
"""

import math
import numpy as np
from scipy.special import expit # Sigmoid function from scipy
def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Convert inputs to numpy arrays
	features = np.array(features)
	labels = np.array(labels)
	weights = np.array(weights)
	bias = np.array(bias)
	output = (features @ weights) + bias
	
    # Apply sigmoid activation function
	probabilities = np.round(expit(output), 4)  
	
	# Calculate mean squared error
	mse = np.mean((probabilities - labels) ** 2)
	mse = np.round(mse, 4)  # Round MSE to four decimal places
	
	return probabilities.tolist(), mse.item() # Convert to list and return as a tuple

# Example usage
classification = single_neuron_model(
	features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], 
	labels = [0, 1, 0], weights = [0.7, -0.4], 
	bias = -0.1
	)
print(classification)