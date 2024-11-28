import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiInputLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        """
        Initialize the MultiInputLSTM class with the given parameters.

        Parameters:
        - input_sz (int): Number of input features (e.g., 5 for ['Open', 'High', 'Low', 'Close', 'Volume']).
        - hidden_sz (int): Number of hidden units in each LSTM cell.
        - dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__()
        # LSTMs for each feature column
        self.open_lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.high_lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.low_lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.close_lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.volume_lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)

        # # Attention weights
        # self.Wa = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # self.ba = nn.Parameter(torch.Tensor(hidden_size))

        # Fully connected layer for combining LSTM outputs and predicting multiple values
        self.fc = nn.Linear(hidden_size * input_size, input_size)

    def forward(self, open_seq, high_seq, low_seq, close_seq, volume_seq):
        """
        Forward pass for the MultiInputLSTM model.

        Parameters:
        - open_seq (Tensor): Input sequence for 'Open' prices.
        - high_seq (Tensor): Input sequence for 'High' prices.
        - low_seq (Tensor): Input sequence for 'Low' prices.
        - close_seq (Tensor): Input sequence for 'Close' prices.

        Returns:
        - Tensor: Predicted values.
        """
        # Process each feature through its respective LSTM
        _, (open_hidden, _) = self.open_lstm(open_seq)
        _, (high_hidden, _) = self.high_lstm(high_seq)
        _, (low_hidden, _) = self.low_lstm(low_seq)
        _, (close_hidden, _) = self.close_lstm(close_seq)
        _, (volume_hidden, _) = self.volume_lstm(volume_seq)

        # # Calculate attention weights based on hidden states
        # u_open = torch.tanh(torch.matmul(open_hidden[-1], self.Wa) + self.ba)
        # u_high = torch.tanh(torch.matmul(high_hidden[-1], self.Wa) + self.ba)
        # u_low = torch.tanh(torch.matmul(low_hidden[-1], self.Wa) + self.ba)
        # u_close = torch.tanh(torch.matmul(close_hidden[-1], self.Wa) + self.ba)
        # u_volume = torch.tanh(torch.matmul(volume_hidden[-1], self.Wa) + self.ba)

        # # Normalize attention weights
        # alpha_open = F.softmax(u_open, dim=1)
        # alpha_high = F.softmax(u_high, dim=1)
        # alpha_low = F.softmax(u_low, dim=1)
        # alpha_close = F.softmax(u_close, dim=1)
        # alpha_volume = F.softmax(u_volume, dim=1)

        # # Weighted sum of hidden states
        # open_weighted = alpha_open * open_hidden[-1]
        # high_weighted = alpha_high * high_hidden[-1]
        # low_weighted = alpha_low * low_hidden[-1]
        # close_weighted = alpha_close * close_hidden[-1]
        # volume_weighted = alpha_volume * volume_hidden[-1]

        # Concatenate hidden states
        L_t = torch.cat([open_hidden[-1], high_hidden[-1], low_hidden[-1], close_hidden[-1], volume_hidden[-1]], dim=1)

        # Predict values
        return self.fc(L_t)