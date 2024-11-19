import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Preprocessing:
    def __init__(self, output_days, folder_path="data", split_ratio=0.8, sequence_length=100):
        """
        Initialize the Preprocessing class.

        Parameters:
        - folder_path (str): Path to the folder where CSV data files are located.
        - split_ratio (float): Proportion of data to use as training data.
        - sequence_length (int): Number of past days to use as input features.
        """
        self.output_days = output_days
        self.folder_path = folder_path
        self.split_ratio = split_ratio
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.volume_scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self):
        """
        Load and combine all CSV files in the folder into a single DataFrame.

        Returns:
        - DataFrame: Combined data from all CSV files.
        """
        all_data = []
        for file in os.listdir(self.folder_path):
            if file.endswith('.csv'):
                ticker_data = pd.read_csv(os.path.join(self.folder_path, file))
                all_data.append(ticker_data)
        dataset = pd.concat(all_data, ignore_index=True)
        dataset['Date'] = pd.to_datetime(dataset['Date']).dt.date
        dataset = dataset.sort_values('Date').reset_index(drop=True)
        return dataset

    def create_sequences_multi_input(self, data):
        """
        Create sequences of `sequence_length` days with the next day's 'Close' value as the label.

        Parameters:
        - data (DataFrame): Scaled DataFrame of stock data.

        Returns:
        - x_open, x_high, x_low, x_close, x_volume (arrays): Sequences for each feature.
        - y (array): Corresponding 'Close' values as labels.
        """
        x_open, x_high, x_low, x_close, x_volume, y = [], [], [], [], [], []
        y_dates = []
        for i in range(len(data) - self.sequence_length- self.output_days):
            # Input sequences for each feature
            x_open.append(data.iloc[i:i + self.sequence_length]['Open'].values)
            x_high.append(data.iloc[i:i + self.sequence_length]['High'].values)
            x_low.append(data.iloc[i:i + self.sequence_length]['Low'].values)
            x_close.append(data.iloc[i:i + self.sequence_length]['Close'].values)
            x_volume.append(data.iloc[i:i + self.sequence_length]['Volume'].values)
            # Label is the next day's 'Close' price
            y.append(data.iloc[i + self.sequence_length:i + self.sequence_length + self.output_days]['Close'].values.flatten())
            y_dates.append(data['Date'].iloc[i + self.sequence_length:i + self.sequence_length + self.output_days].values)

        return (
            np.array(x_open), np.array(x_high), np.array(x_low), np.array(x_close), np.array(x_volume), np.array(y), np.array(y_dates)
        )

    def split_data(self, dataset):
        """
        Split the dataset chronologically into training and testing sets.

        Parameters:
        - dataset (DataFrame): The full dataset to split.

        Returns:
        - train_data, test_data (DataFrames): Unscaled training and testing datasets.
        """
        split_point = int(len(dataset) * self.split_ratio)
        train_data = dataset[:split_point]
        test_data = dataset[split_point:]
        return train_data, test_data

    def scale_data(self, train_data, test_data):
        """
        Scale the training and testing datasets.

        Parameters:
        - train_data (DataFrame): Unscaled training dataset.
        - test_data (DataFrame): Unscaled testing dataset.

        Returns:
        - train_scaled, test_scaled (DataFrames): Scaled training and testing datasets with additional columns.
        """
        # Scale Open, High, Low, Close
        train_scaled = pd.DataFrame(
            self.scaler.fit_transform(train_data.iloc[:, 1:5].values),
            columns=train_data.iloc[:, 1:5].columns
        )
        test_scaled = pd.DataFrame(
            self.scaler.transform(test_data.iloc[:, 1:5].values),
            columns=test_data.iloc[:, 1:5].columns
        )

        # Scale Volume separately
        vol_train_scaled = pd.DataFrame(
            self.volume_scaler.fit_transform(train_data.iloc[:, [5]].values),
            columns=train_data.iloc[:, [5]].columns
        )
        vol_test_scaled = pd.DataFrame(
            self.volume_scaler.transform(test_data.iloc[:, [5]].values),
            columns=test_data.iloc[:, [5]].columns
        )

        # Add Volume column to scaled DataFrame
        train_scaled['Volume'] = vol_train_scaled['Volume']
        test_scaled['Volume'] = vol_test_scaled['Volume']

        # Add Date column to scaled DataFrame
        train_scaled.insert(loc=0, column='Date', value=train_data['Date'].values)
        test_scaled.insert(loc=0, column='Date', value=test_data['Date'].values)

        return train_scaled, test_scaled

    def preprocess_pipeline(self):
        """
        Full preprocessing pipeline that loads, combines, splits, and scales data.

        Returns:
        - x_open_train, x_high_train, x_low_train, x_close_train, x_volume_train (arrays): Training input sequences.
        - x_open_test, x_high_test, x_low_test, x_close_test, x_volume_test (arrays): Testing input sequences.
        - y_train, y_test (arrays): Training and testing labels.
        """
        dataset = self.load_data()

        # Split data
        train_data, test_data = self.split_data(dataset)

        # Scale data
        train_scaled, test_scaled = self.scale_data(train_data, test_data)

        # Create sequences for multi-input LSTM
        x_open_train, x_high_train, x_low_train, x_close_train, x_volume_train, y_train, y_train_dates = self.create_sequences_multi_input(
            train_scaled)
        x_open_test, x_high_test, x_low_test, x_close_test, x_volume_test, y_test, y_test_dates = self.create_sequences_multi_input(
            test_scaled)

        return (
            x_open_train, x_high_train, x_low_train, x_close_train, x_volume_train, y_train, y_train_dates,
            x_open_test, x_high_test, x_low_test, x_close_test, x_volume_test, y_test, y_test_dates
        )

