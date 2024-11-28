import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Preprocessing:
    def __init__(self, folder_path: str, split_ratio: float, sequence_length: int):
        """
        Parameters:
        - folder_path (str): Path to the folder containing CSV data files.
        - split_ratio (float): Proportion of data to use for training.
        - sequence_length (int): Number of past days to use as input features..
        """
        self.folder_path = folder_path
        self.split_ratio = split_ratio
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self):
        """
        Load and sort the dataset from the specified folder path.

        Returns:
        - dataset (DataFrame): Loaded and sorted dataset.
        """
        file = os.listdir(self.folder_path)
        dataset = pd.read_csv(os.path.join(self.folder_path, file[0]))
        dataset['Date'] = pd.to_datetime(dataset['Date']).dt.date
        dataset = dataset.sort_values('Date').reset_index(drop=True)
        return dataset

    def create_sequences_multi_input(self, data):
        """
        Create sequences of `sequence_length` days with the next day's parameters as the label.

        Parameters:
        - data (DataFrame): Scaled DataFrame of stock data.

        Returns:
        - x_open, x_high, x_low, x_close (arrays): Sequences for each feature.
        - y (array): Corresponding next day's parameters as labels.
        - y_dates (array): Dates corresponding to the labels.
        """
        x_open, x_high, x_low, x_close, x_volume, y = [], [], [], [], [], []
        y_dates = []
        for i in range(len(data) - self.sequence_length):
            # Input sequences for each feature
            x_open.append(data.iloc[i:i + self.sequence_length]['Open'].values)
            x_high.append(data.iloc[i:i + self.sequence_length]['High'].values)
            x_low.append(data.iloc[i:i + self.sequence_length]['Low'].values)
            x_close.append(data.iloc[i:i + self.sequence_length]['Close'].values)
            x_volume.append(data.iloc[i:i + self.sequence_length]['Volume'].values)

            # Label is the next day's 'Close' price
            y.append(np.array(data.iloc[i + self.sequence_length][['Open', 'High', 'Low', 'Close', 'Volume']].values, dtype=float))
            y_dates.append(data['Date'].iloc[i + self.sequence_length])
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
        Scale the training and testing datasets using MinMaxScaler.

        Parameters:
        - train_data (DataFrame): Unscaled training dataset.
        - test_data (DataFrame): Unscaled testing dataset.

        Returns:
        - train_scaled, test_scaled (DataFrames): Scaled training and testing datasets with additional columns.
        """
        # Scale Open, High, Low, Close, Volume
        train_scaled = pd.DataFrame(
            self.scaler.fit_transform(train_data.iloc[:, 1:6].values),
            columns=train_data.iloc[:, 1:6].columns
        )
        test_scaled = pd.DataFrame(
            self.scaler.transform(test_data.iloc[:, 1:6].values),
            columns=test_data.iloc[:, 1:6].columns
        )

        # Add Date column to scaled DataFrame
        train_scaled.insert(loc=0, column='Date', value=train_data['Date'].values)
        test_scaled.insert(loc=0, column='Date', value=test_data['Date'].values)

        return train_scaled, test_scaled

    def preprocess_pipeline(self):
        """
        Execute the full preprocessing pipeline: load, split, scale, and create sequences.

        Returns:
        - x_open_train, x_high_train, x_low_train, x_close_train (arrays): Training input sequences.
        - y_train (array): Training labels.
        - y_train_dates (array): Dates corresponding to the training labels.
        - x_open_test, x_high_test, x_low_test, x_close_test (arrays): Testing input sequences.
        - y_test (array): Testing labels.
        - ly_test_dates (array): Dates corresponding to the testing labels.
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

