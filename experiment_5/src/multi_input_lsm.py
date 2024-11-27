import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiInputLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        """
        Initialize the MultiInputLSTM class with the given parameters.

        Parameters:
        - input_sz (int): Number of input features (e.g., 4 for ['Open', 'High', 'Low', 'Close']).
        - hidden_sz (int): Number of hidden units in each LSTM cell.
        """
        super().__init__()
        self.hidden_size = hidden_sz

        # LSTMs for each feature column
        self.open_lstm = nn.LSTM(input_size=1, hidden_size=hidden_sz, batch_first=True)
        self.high_lstm = nn.LSTM(input_size=1, hidden_size=hidden_sz, batch_first=True)
        self.low_lstm = nn.LSTM(input_size=1, hidden_size=hidden_sz, batch_first=True)
        self.close_lstm = nn.LSTM(input_size=1, hidden_size=hidden_sz, batch_first=True)

        # Attention weights
        self.Wa = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.ba = nn.Parameter(torch.Tensor(hidden_sz))

        # Fully connected layer for combining LSTM outputs and predicting multiple values
        self.fc = nn.Linear(hidden_sz * input_sz, input_sz)

    def forward(self, open_seq, high_seq, low_seq, close_seq):
        """
        Initialize the MultiInputLSTM class with the given parameters.

        Parameters:
        - input_sz (int): Number of input features (e.g., 4 for ['Open', 'High', 'Low', 'Close']).
        - hidden_sz (int): Number of hidden units in each LSTM cell.
        """
        # Process each feature through its respective LSTM
        _, (open_hidden, _) = self.open_lstm(open_seq)
        _, (high_hidden, _) = self.high_lstm(high_seq)
        _, (low_hidden, _) = self.low_lstm(low_seq)
        _, (close_hidden, _) = self.close_lstm(close_seq)

        # Calculate attention weights based on hidden states
        u_open = torch.tanh(torch.matmul(open_hidden[-1], self.Wa) + self.ba)
        u_high = torch.tanh(torch.matmul(high_hidden[-1], self.Wa) + self.ba)
        u_low = torch.tanh(torch.matmul(low_hidden[-1], self.Wa) + self.ba)
        u_close = torch.tanh(torch.matmul(close_hidden[-1], self.Wa) + self.ba)

        # Normalize attention weights
        alpha_open = F.softmax(u_open, dim=1)
        alpha_high = F.softmax(u_high, dim=1)
        alpha_low = F.softmax(u_low, dim=1)
        alpha_close = F.softmax(u_close, dim=1)

        # Weighted sum of hidden states
        open_weighted = alpha_open * open_hidden[-1]
        high_weighted = alpha_high * high_hidden[-1]
        low_weighted = alpha_low * low_hidden[-1]
        close_weighted = alpha_close * close_hidden[-1]

        # Concatenate the weighted hidden states
        L_t = torch.cat([open_weighted, high_weighted, low_weighted, close_weighted], dim=1)

        # Predict values
        return self.fc(L_t)
