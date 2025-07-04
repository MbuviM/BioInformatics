"""
Write a Python function that simulates a single neuron with a sigmoid activation function for binary classification, handling multidimensional input features. 
The function should take a list of feature vectors (each vector representing multiple features for an example), associated true binary labels, and the neuron's weights (one for each feature) and bias as input. 
It should return the predicted probabilities after sigmoid activation and the mean squared error between the predicted probabilities and the true labels, both rounded to four decimal places.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple

def single_neuron_model(
    features: List[List[float]],
    labels: List[float],
    weights: List[float],
    bias: float
) -> Tuple[List[float], float]:
    """
    Compute output probabilities and MSE for a single neuron.
    Uses built-in sigmoid and MSE loss.
    """
    # Convert inputs to PyTorch tensors
    features =torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    weights = torch.tensor(weights, dtype=torch.float32)
    bias = torch.tensor(bias, dtype=torch.float32)

    # Compute the linear combination of inputs
    output = torch.matmul(features, weights) + bias

    # Apply sigmoid activation function
    probabilities = torch.sigmoid(output)

    # Calculate mean squared error
    mse = F.mse_loss(probabilities, labels)

    # Convert probabilities to list and return
    return probabilities.tolist(), round(mse.item(), 4)
 
# Example usage
classification = single_neuron_model(
	features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], 
	labels = [0, 1, 0], weights = [0.7, -0.4], 
	bias = -0.1
	)
print(classification)