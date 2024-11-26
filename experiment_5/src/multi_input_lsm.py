import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiInputLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        """
        Multi-input LSTM model for predicting multiple future days.

        Parameters:
        - input_sz (int): Number of input features (e.g., 1 for each column).
        - hidden_sz (int): Number of hidden units in each LSTM cell.
        """
        super().__init__()
        self.hidden_size = hidden_sz

        # LSTMs for each column
        self.open_lstm = nn.LSTM(input_size=1, hidden_size=hidden_sz, batch_first=True)
        self.high_lstm = nn.LSTM(input_size=1, hidden_size=hidden_sz, batch_first=True)
        self.low_lstm = nn.LSTM(input_size=1, hidden_size=hidden_sz, batch_first=True)
        self.close_lstm = nn.LSTM(input_size=1, hidden_size=hidden_sz, batch_first=True)

        # Attention weights
        self.Wa = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        self.ba = nn.Parameter(torch.Tensor(hidden_sz))

        # Fully connected layer for combining LSTM outputs and predicting multiple days
        self.fc = nn.Linear(hidden_sz * input_sz, input_sz)

    def forward(self, open_seq, high_seq, low_seq, close_seq):
        """
        Forward pass for the model.

        Parameters:
        - open_seq, high_seq, low_seq, close_seq (Tensor): Input sequences for each feature with shape (batch_size, seq_len, 1).

        Returns:
        - Tensor: Predicted values for the next `output_days` days.
        """
        # Process each feature through its LSTM
        _, (open_hidden, _) = self.open_lstm(open_seq)
        _, (high_hidden, _) = self.high_lstm(high_seq)
        _, (low_hidden, _) = self.low_lstm(low_seq)
        _, (close_hidden, _) = self.close_lstm(close_seq)

        # Concatenate the final hidden states from all LSTMs
        combined_features = torch.cat(
            (open_hidden[-1], high_hidden[-1], low_hidden[-1], close_hidden[-1]), dim=1
        )

        # Calculate attention weights
        u_open = torch.tanh(torch.matmul(open_hidden[-1], self.Wa) + self.ba)
        u_high = torch.tanh(torch.matmul(high_hidden[-1], self.Wa) + self.ba)
        u_low = torch.tanh(torch.matmul(low_hidden[-1], self.Wa) + self.ba)
        u_close = torch.tanh(torch.matmul(close_hidden[-1], self.Wa) + self.ba)

        # Normalize attention weights
        alpha_open = F.softmax(u_open, dim=1)
        alpha_high = F.softmax(u_high, dim=1)
        alpha_low = F.softmax(u_low, dim=1)
        alpha_close = F.softmax(u_close, dim=1)

        # Weighted sum of cell inputs
        L_t = alpha_open * open_hidden[-1] + alpha_high * high_hidden[-1] + alpha_low * low_hidden[-1]
