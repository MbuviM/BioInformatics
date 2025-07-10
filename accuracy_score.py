"""
Calculate Accuracy Score

Write a Python function to calculate the accuracy score of a model's predictions. T
he function should take in two 1D numpy arrays: y_true, which contains the true labels, and y_pred, which contains the predicted labels. 
It should return the accuracy score as a float.
"""

import numpy as np

def accuracy_score(y_true, y_pred):
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	accuracy_score = np.sum(y_true == y_pred)/len(y_true)

	return accuracy_score
	