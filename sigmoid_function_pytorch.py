"""
Write a Python function that computes the output of the sigmoid activation function given an input value z. 
The function should return the output rounded to four decimal places.
"""

import torch

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    z = torch.tensor(z)
    result = torch.sigmoid(z)
    return round(result.item(), 4)