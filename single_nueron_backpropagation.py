"""
Single Neuron with Backpropagation

Write a Python function that simulates a single neuron with sigmoid activation, and implements backpropagation to update the neuron's weights and bias. 
The function should take a list of feature vectors, associated true binary labels, initial weights, initial bias, a learning rate, and the number of epochs. 
The function should update the weights and bias using gradient descent based on the MSE loss, 
and return the updated weights, bias, and a list of MSE values for each epoch, each rounded to four decimal places.
"""

import torch
from typing import List, Tuple, Union
import torch.nn.functional as F

def train_neuron(
    features: Union[List[List[float]], torch.Tensor],
    labels:   Union[List[float],      torch.Tensor],
    initial_weights: Union[List[float], torch.Tensor],
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> Tuple[List[float], float, List[float]]:

    mse_values = []

    # Convert inputs to tensors if not already
    features = (
        features if isinstance(features, torch.Tensor)
        else torch.tensor(features, dtype=torch.float32)
    )
    labels = (
        labels if isinstance(labels, torch.Tensor)
        else torch.tensor(labels, dtype=torch.float32)
    )
    initial_weights = (
        initial_weights.detach().clone().requires_grad_(True)
        if isinstance(initial_weights, torch.Tensor)
        else torch.tensor(initial_weights, dtype=torch.float32, requires_grad=True)
    )
    initial_bias = (
        initial_bias.detach().clone().requires_grad_(True)
        if isinstance(initial_bias, torch.Tensor)
        else torch.tensor(initial_bias, dtype=torch.float32, requires_grad=True)
    )

    # Setup optimizer
    optimizer = torch.optim.SGD([initial_weights, initial_bias], lr=learning_rate)
    
    for epoch in range(epochs):
        # Forward propagation
        output = torch.matmul(features, initial_weights) + initial_bias
        probabilities = torch.sigmoid(output)

        # MSE Loss
        loss = F.mse_loss(probabilities, labels)
        mse_values.append(round(loss.item(), 4))

        # Back Propagation
        # Compute gradients
        optimizer.zero_grad() # Resets the gradients
        loss.backward() # Performs backward propagation
        optimizer.step() # Updates the weights and bias in grad

    return initial_weights, initial_bias, mse_values



    