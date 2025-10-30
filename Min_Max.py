"""
Implement a function that performs Min-Max Normalization on a list of integers, scaling all values to the range [0, 1]. Min-Max normalization helps ensure that all features contribute equally to a model by scaling them to a common range.

Example:
Input:
min_max([1, 2, 3, 4, 5])
Output:
[0.0, 0.25, 0.5, 0.75, 1.0]
Reasoning:
The minimum value is 1 and the maximum is 5. Each value is scaled using the formula (x - min) / (max - min).
"""

def min_max(x: list[int]) -> list[float]:
    
    norm_values = []
    for val in x:
        if x[0] == x[-1]:
            norm_values.append(0.0)
        else:
            new_val = (val-x[0])/(x[-1]-x[0])
            norm_values.append(new_val)
    return norm_values

