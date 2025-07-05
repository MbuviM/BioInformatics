"""
Write a Python function that computes the transpose of a given matrix.
"""

import torch

def transpose_matrix(a) -> torch.Tensor:
    """
    Transpose a 2D matrix `a` using PyTorch.
    Inputs can be Python lists, NumPy arrays, or torch Tensors.
    Returns a transposed tensor.
    """
    a_t = torch.as_tensor(a)
    b = torch.transpose(a_t, 0, 1) # 0, 1 show dimensions rows and columns respectively
    return b

# Example usage
if __name__ == "__main__":
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    transposed = transpose_matrix(matrix)
    print(transposed)