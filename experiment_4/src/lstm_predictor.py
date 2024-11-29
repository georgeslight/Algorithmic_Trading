import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, output_sz: int, num_layers: int, dropout: float):
        """
        Initializes the LSTMPredictor model.

        Parameters:
        - input_sz (int): The number of input features.
        - hidden_sz (int): The number of features in the hidden state.
        - output_sz (int): The number of output features.
        - num_layers (int): The number of recurrent layers.
        - dropout (float): The dropout probability.
        """
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_sz, hidden_size=hidden_sz, batch_first=True, dropout=dropout, bias=True, num_layers=num_layers, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=hidden_sz * 2, hidden_size=hidden_sz, batch_first=True, dropout=dropout, bias=True, num_layers=num_layers, bidirectional=True)
        self.lstm3 = nn.LSTM(input_size=hidden_sz * 2, hidden_size=hidden_sz, batch_first=True, dropout=dropout, bias=True, num_layers=num_layers, bidirectional=True)
        self.lstm4 = nn.LSTM(input_size=hidden_sz * 2, hidden_size=hidden_sz, batch_first=True, dropout=dropout, bias=True, num_layers=num_layers, bidirectional=True)
        self.attention = nn.Linear(hidden_sz * 2, 1)
        self.fc = nn.Linear(hidden_sz * 2, output_sz)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Parameters:
        - x (Tensor): The input tensor of shape (batch_size, sequence_length, input_sz).

        Returns:
        - output (Tensor): The output tensor of shape (batch_size, output_sz).
        """
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out3, _ = self.lstm3(lstm_out2)
        lstm_out4, _ = self.lstm4(lstm_out3)

        # Apply attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out4), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out4, dim=1)

        # Pass the context vector through the fully connected layer
        output = self.fc(context_vector)

        return output
