"""
Single Neuron with Backpropagation

Write a Python function that simulates a single neuron with sigmoid activation, and implements backpropagation to update the neuron's weights and bias. 
The function should take a list of feature vectors, associated true binary labels, initial weights, initial bias, a learning rate, and the number of epochs. 
The function should update the weights and bias using gradient descent based on the MSE loss, 
and return the updated weights, bias, and a list of MSE values for each epoch, each rounded to four decimal places.
"""

import numpy as np
from scipy.special import expit
from math import exp
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    mse_values =[]

    for epoch in range(epochs):
        # Forward Propagation
        output = np.dot(features, initial_weights) + initial_bias
        # Find probabilities (Sigmoid Activation)
        probability = expit(output)
        
        # Find MSE loss
        loss = np.mean((probability - labels) ** 2)
        mse_values.append(np.round(loss, 4))

        # Backpropagation (Compute gradients)
        dmse_dprob = 2 * (probability-labels)
        dprob_doutput = probability * (1 - probability)
        doutput = dmse_dprob * dprob_doutput
        doutput_dw = features
        doutput_db = 1
        dw = np.dot(features.T, doutput)/ len(features) # T is for transpose
        db = np.mean(doutput)

        # Partial Derivatives
        updated_weights = np.round(initial_weights - (learning_rate * dw), 4)
        updated_bias = np.round(initial_bias - (learning_rate * db), 4)
        
        initial_bias = updated_bias
        initial_weights = updated_weights

    return updated_weights.tolist(), updated_bias, mse_values

