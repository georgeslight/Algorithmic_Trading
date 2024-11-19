import torch
import torch.nn as nn


class MultiInputLSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        """
        Multi-input LSTM model for predicting multiple future days.

        Parameters:
        - input_sz (int): Number of input features (e.g., 1 for each column).
        - hidden_sz (int): Number of hidden units in each LSTM cell.
        - output_days (int): Number of future days to predict.
        """
        super().__init__()
        self.hidden_size = hidden_sz

        # LSTMs for each column
        self.open_lstm = nn.LSTM(input_size=1, hidden_size=hidden_sz, batch_first=True)
        self.high_lstm = nn.LSTM(input_size=1, hidden_size=hidden_sz, batch_first=True)
        self.low_lstm = nn.LSTM(input_size=1, hidden_size=hidden_sz, batch_first=True)
        self.close_lstm = nn.LSTM(input_size=1, hidden_size=hidden_sz, batch_first=True)
        # self.volume_lstm = nn.LSTM(input_size=1, hidden_size=hidden_sz, batch_first=True)

        # Fully connected layer for combining LSTM outputs and predicting multiple days
        self.fc = nn.Linear(hidden_sz * input_sz, input_sz)

    def forward(self, open_seq, high_seq, low_seq, close_seq):
        """
        Forward pass for the model.

        Parameters:
        - open_seq, high_seq, low_seq, close_seq, volume_seq (Tensor):
          Input sequences for each feature with shape (batch_size, seq_len, 1).

        Returns:
        - Tensor: Predicted values for the next `output_days` days.
        """
        # Process each feature through its LSTM
        _, (open_hidden, _) = self.open_lstm(open_seq)
        _, (high_hidden, _) = self.high_lstm(high_seq)
        _, (low_hidden, _) = self.low_lstm(low_seq)
        _, (close_hidden, _) = self.close_lstm(close_seq)
        # _, (volume_hidden, _) = self.volume_lstm(volume_seq)

        # Concatenate the final hidden states from all LSTMs
        combined_features = torch.cat(
            (open_hidden[-1], high_hidden[-1], low_hidden[-1], close_hidden[-1]), dim=1
        )

        # Pass combined features through fully connected layer to predict multiple days
        prediction = self.fc(combined_features)

        return prediction
