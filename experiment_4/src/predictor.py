import torch
import numpy as np

class Predictor:
    def __init__(self, model, scaler, sequence_length, device):
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
        # self.volume_scaler = volume_scaler
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
        start_index = dataset.index[dataset['Date'] == start_date][0] + 1
        end_index = start_index - self.sequence_length

        if end_index < 0:
            raise ValueError("Insufficient data before the start date to form a sequence.")

        # Extract the initial sequence
        initial_sequence = dataset.iloc[end_index:start_index][['Open', 'High', 'Low', 'Close']].values
        # Scale
        scaled_seq = self.scaler.transform(initial_sequence)

        # Split the scaled sequence into separate tensors for each feature
        open_seq = torch.tensor(scaled_seq[:, [0]], dtype=torch.float32, device=self.device).unsqueeze(0)
        high_seq = torch.tensor(scaled_seq[:, [1]], dtype=torch.float32, device=self.device).unsqueeze(0)
        low_seq = torch.tensor(scaled_seq[:, [2]], dtype=torch.float32, device=self.device).unsqueeze(0)
        close_seq = torch.tensor(scaled_seq[:, [3]], dtype=torch.float32, device=self.device).unsqueeze(0)

        # Iterative prediction
        predictions_scaled = []
        self.model.eval()
        with torch.no_grad():
            for _ in range(days_to_predict):
                output = self.model(open_seq, high_seq, low_seq, close_seq)  # Predict the next day
                predicted_day = output.squeeze(0).cpu().numpy()  # Remove batch dimension
                predictions_scaled.append(predicted_day)

                # Update sequences with the predicted day
                next_day_tensor = torch.tensor(predicted_day, dtype=torch.float32, device=self.device).unsqueeze(0)
                open_seq = torch.cat((open_seq[:, 1:], next_day_tensor[:, 0:1].unsqueeze(-1)), dim=1)
                high_seq = torch.cat((high_seq[:, 1:], next_day_tensor[:, 1:2].unsqueeze(-1)), dim=1)
                low_seq = torch.cat((low_seq[:, 1:], next_day_tensor[:, 2:3].unsqueeze(-1)), dim=1)
                close_seq = torch.cat((close_seq[:, 1:], next_day_tensor[:, 3:4].unsqueeze(-1)), dim=1)

            # Convert scaled predictions back to the original scale
            predictions_scaled_np = np.array(predictions_scaled)
            predictions_original = self.scaler.inverse_transform(predictions_scaled_np)

            return predictions_original, predictions_scaled_np

