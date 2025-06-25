"""
Matrix-Vector Dot Product

Write a Python function that computes the dot product of a matrix and a vector. 
The function should return a list representing the resulting vector if the operation is valid, or -1 if the matrix and vector dimensions are incompatible. 
A matrix (a list of lists) can be dotted with a vector (a list) only if the number of columns in the matrix equals the length of the vector. 
For example, an n x m matrix requires a vector of length m.
a = [[1, 2], [2, 4]], b = [1, 2]
Output:
[5, 10]
"""
import numpy as np
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.

    # Convert lists to numpy arrays for easier manipulation
	a =np.array(a)
	b= np.array(b)

	# Check for shape of arrays
	a_shape = a.shape
	b_shape = b.shape

	# Check if the dimensions are compatible for dot product
	if a_shape[1] == b_shape[0]:
		c = np.dot(a,b)
		return c.tolist() # Convert numpy array to list
	else:
		return -1

matrix_dot_vector(a=[[4, 1], [3, 2]], b=[5,6])
