"""
Write a Python function that computes the transpose of a given matrix.
"""

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
	import numpy as np
	a = np.array(a)
	b = np.transpose(a)
	return b.tolist()

# Example usage
if __name__ == "__main__":
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    transposed = transpose_matrix(matrix)
    print(transposed)  # Output: [[1 4 7] [2 5 8] [3 6 9]]