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
        
        Args:
            x: Input sequence (list or array of inputs)
            initial_hidden_state: Initial hidden state
            initial_cell_state: Initial cell state
            
        Returns:
            hidden_states: List of hidden states at each time step
            final_hidden_state: Final hidden state
            final_cell_state: Final cell state
        """
        # Initialize states - ensure proper shape
        prev_h = initial_hidden_state.copy()
        if prev_h.shape[1] != 1:
            prev_h = prev_h.reshape(-1, 1)
            
        prev_c = initial_cell_state.copy()  
        if prev_c.shape[1] != 1:
            prev_c = prev_c.reshape(-1, 1)
        
        # Store all hidden states
        hidden_states = []
        
        for t in range(len(x)):
            # Current input
            x_t = x[t].reshape(-1, 1)
            
            # Concatenate input and previous hidden state
            inputs = np.concatenate((prev_h, x_t), axis=0)

            # Forget gate
            f_t = expit(np.dot(self.Wf, inputs) + self.bf)

            # Input gate
            i_t = expit(np.dot(self.Wi, inputs) + self.bi)
            
            # Candidate cell state
            c_hat_t = np.tanh(np.dot(self.Wc, inputs) + self.bc)
            
            # Update cell state
            c_t = f_t * prev_c + i_t * c_hat_t

            # Output gate
            o_t = expit(np.dot(self.Wo, inputs) + self.bo)
            
            # Update hidden state
            h_t = o_t * np.tanh(c_t)
            
            # Store current hidden state
            hidden_states.append(h_t.copy())
            
            # Update for next iteration
            prev_c = c_t
            prev_h = h_t

        return hidden_states, h_t, c_t
