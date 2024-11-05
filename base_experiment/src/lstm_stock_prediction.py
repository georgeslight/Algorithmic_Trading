import torch
import torch.nn as nn
import math


class LSTMStockPredictor(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, output_sz: int):
        """
        Custom LSTM model for stock prediction.

        Parameters:
        - input_sz (int): Number of input features (e.g., 5 for ['Open', 'High', 'Low', 'Close', 'Volume']).
        - hidden_sz (int): Number of hidden units in the LSTM cell.
        - output_sz (int): Number of output features (e.g., 5 for predicting the next day's prices).
        """
        super().__init__()
        self.input_size = input_sz
        self.hidden_size = hidden_sz
        self.output_size = output_sz

        # LSTM cell parameters; Setup
        self.W = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4)) #  A weight matrix for the input x_t, with dimensions (input_size, hidden_size * 4). LSTMs have 4 gate components, so the hidden size is multiplied by 4.
        self.U = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4)) # A weight matrix for the previous hidden state h_t, with dimensions (hidden_size, hidden_size * 4).
        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4)) # A bias term for the gates, also with size (hidden_size * 4)

        # Fully connected layer to map the hidden state to the output
        self.fc = nn.Linear(hidden_sz, output_sz)

        # Initialize weights
        self.init_weights()

    # initializes the weight matrices with small random values to break the symmetry during training.
    def init_weights(self):
        # stdv: is a small number that depends on the size of the hidden size
        stdv = 1.0 / math.sqrt(self.hidden_size) # common way to set up and balance the values.
        for weight in self.parameters(): # This loop goes through every weight (W, U, and bias) in the CustomLSTM cell
            weight.data.uniform_(-stdv, stdv) # Gives each weight a random value between -stdv and +stdv
            # Like giving it some "initial ideas" about what might be important, so it can test those ideas and improve them over time
    # If all weights started at zero or the same value, it wouldn't learn effectively because it wouldn't know what to change when it makes mistakes.
    # By starting with small, random, values it can begin learning which weight to adjust to make better rpedictions.

    # Processing Each value
    def forward(self, x, init_states=None):
        """
        Forward pass for the LSTM.

        Parameters:
        - x (Tensor): Input data of shape (batch_size, seq_len, input_size).
        - init_states (Tuple[Tensor, Tensor]): Initial hidden and cell states.

        Returns:
        - Tensor: Final output for each sequence in the batch.
        - Tensor: All hidden states for the sequence.
        """
        bs, seq_sz, _ = x.shape
        hidden_seq = []

        if init_states is None:
            # h_t: hidden state, Used to keep information for short-term use
            # c_t: cell state, long term memory that helps LSTM decide what to remember, what to forget, and what to pass on
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]

            # LSTM gates: input, forget, candidate, and output gates
            gates = x_t @ self.W + h_t @ self.U + self.bias
            # four gates: Input Gate (i_t): Decides how much of the new information should be added to the memory
            # Forget Gate (f_t): Decides how much of the old memory should be kept
            # New Candidate (g_t): Creates a new piece of information to add to the memory
            # Output Gate(o_t): Decides what to pass on as the output
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # Input gate
                torch.sigmoid(gates[:, HS:HS * 2]),  # Forget gate
                torch.tanh(gates[:, HS * 2:HS * 3]),  # New candidate
                torch.sigmoid(gates[:, HS * 3:])  # Output gate
            )

            # Cell state update
            c_t = f_t * c_t + i_t * g_t
            # Hidden state update
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        # Concatenate all hidden states and transpose to match batch-first format
        hidden_seq = torch.cat(hidden_seq, dim=0).transpose(0, 1).contiguous()

        # Pass the last hidden state through a fully connected layer to get the output
        output = self.fc(h_t)

        return output, hidden_seq