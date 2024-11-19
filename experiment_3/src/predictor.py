import torch
import numpy as np

class Predictor:
    def __init__(self, model, scaler, volume_scaler, sequence_length, device):
        """
        Initialize the Predictor class.

        Parameters:
        - model: The trained LSTM model.
        - scaler: The MinMaxScaler used for data preprocessing.
        - sequence_length: Length of the input sequence.
        - device: Device to run the model on ('cpu' or 'cuda').
        """
        self.model = model
        self.scaler = scaler
        self.volume_scaler = volume_scaler
        self.sequence_length = sequence_length
        self.device = device

    def predict_future(self, dataset, start_date, days_to_predict):
        """
        Predict stock prices for a given number of days starting from a specific date.

        Parameters:
        - dataset: The complete dataset as a pandas DataFrame.
        - start_date: The date from which predictions start.
        - days_to_predict: Number of days to predict iteratively.

        Returns:
        - iterative_predictions_original: Predicted stock prices in the original scale.
        """
        # Locate the starting sequence
        start_index = dataset.index[dataset['Date'] == start_date][0]
        end_index = start_index - self.sequence_length

        if end_index < 0:
            raise ValueError("Insufficient data before the start date to form a sequence.")

        # Extract the initial sequence
        initial_sequence = dataset.iloc[end_index:start_index + 1][['Open', 'High', 'Low', 'Close', 'Volume']].values
        # Scale
        current_sequence = self.scale(initial_sequence)

        open_seq = current_sequence[:, :, 0].unsqueeze(2)
        high_seq = current_sequence[:, :, 1].unsqueeze(2)
        low_seq = current_sequence[:, :, 2].unsqueeze(2)
        close_seq = current_sequence[:, :, 3].unsqueeze(2)
        volume_seq = current_sequence[:, :, 4].unsqueeze(2)

        # Iterative prediction
        predictions = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(days_to_predict):
                output = self.model(open_seq, high_seq, low_seq, close_seq, volume_seq)  # Predict the next day
                predicted_day = output.squeeze(0).cpu().numpy()  # Remove batch dimension
                predictions.append(predicted_day)

                # Update the sequence
                next_day_tensor = torch.tensor(predicted_day, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
                current_sequence = torch.cat((current_sequence[:, 1:], next_day_tensor), dim=1)

        # Convert predictions back to original scale
        # predictions_original = self.scaler.inverse_transform(np.array(predictions))
        predictions_scaled = np.array(predictions)

        return predictions_scaled

    def scale(self, values):
        ohlc = values[:, :4]  # Extract OHLC columns
        volume = values[:, [4]]  # Extract Volume column

        # Scale OHLC and Volume separately
        scaled_ohlc = self.scaler.transform(ohlc)
        scaled_volume = self.volume_scaler.transform(volume)

        # Combine scaled OHLC and Volume
        combined_values_scaled = np.hstack([scaled_ohlc, scaled_volume])

        # Convert to tensor
        return torch.tensor(combined_values_scaled, dtype=torch.float32).unsqueeze(0).to(self.device)
