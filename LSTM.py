"""
Implement Long Short-Term Memory (LSTM) Network

Task: Implement Long Short-Term Memory (LSTM) Network
Your task is to implement an LSTM network that processes a sequence of inputs and produces the final hidden state and cell state after processing all inputs.

Write a class LSTM with the following methods:

__init__(self, input_size, hidden_size): Initializes the LSTM with random weights and zero biases.
forward(self, x, initial_hidden_state, initial_cell_state): Processes a sequence of inputs and returns the hidden states at each time step, as well as the final hidden state and cell state.
The LSTM should compute the forget gate, input gate, candidate cell state, and output gate at each time step to update the hidden state and cell state.
"""

import numpy as np
from pandas import concat
from scipy.special import expit
class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))

	def forward(self, x, initial_hidden_state, initial_cell_state):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		for h_t, x_t in initial_cell_state and initial_hidden_state:
			f_t = expit(np.dot(self.Wf, concat(h_t, x_t)) + self.bf)
			i_t = expit(np.dot(self.Wi, concat(h_t, x_t)) + self.bi)
			c_hat = np.tanh(np.dot(self.Wc, concat(h_t, x_t)) + self.bc)
			i_t.reshape(-1,1)
			c_t = f_t * h_t + i_t * c_hat
			o_t = expit(np.dot(self.Wo, concat(h_t, x_t)) + self.bo)
			final_h = o_t * np.tanh(c_t)
		
		return final_h
