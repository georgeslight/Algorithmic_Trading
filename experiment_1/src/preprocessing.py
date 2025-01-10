import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Preprocessing:
    def __init__(self, folder_path: str, split_ratio: float, sequence_length: int):
        """
        Initialize the Preprocessing class.

        Parameters:
        - folder_path (str): Path to the folder where CSV data files are located.
        - split_ratio (float): Proportion of data to use as training data.
        - sequence_length (int): Number of past days to use as input features.
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

    def create_sequences(self, data):
        """
        Create sequences of `sequence_length` days with the next day's data as the label.

        Parameters:
        - data (DataFrame): Scaled DataFrame of stock data.

        Returns:
        - x (array): Sequences of past `sequence_length` days.
        - y (array): Corresponding next day's data as labels.
        - x_dates (array): Dates for the input sequences.
        - y_dates (array): Dates for the labels.
        """
        x, y = [], []
        x_dates, y_dates = [], []  # To store dates for input and label
        for i in range(len(data) - self.sequence_length):
            # Input sequence of past `sequence_length` days
            x.append(data.iloc[i:i + self.sequence_length][['Open', 'High', 'Low', 'Close', 'Volume']].values)
            x_dates.append(data.iloc[i:i + self.sequence_length]['Date'].values)  # Store dates for the input sequence
            # Label is the next day's price data
            y.append(data.iloc[i + self.sequence_length][['Open', 'High', 'Low', 'Close', 'Volume']].values)
            y_dates.append(data.iloc[i + self.sequence_length]['Date'])  # Store date for the label

        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32), np.array(x_dates), np.array(y_dates)

    def split_data(self, dataset):
        """
        Split the dataset chronologically into training and testing sets.

        Parameters:
        - dataset (DataFrame): The full dataset to split.

        Returns:
        - train_data (DataFrame): Unscaled training dataset.
        - test_data (DataFrame): Unscaled testing dataset.
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
        - train_scaled (DataFrame): Scaled training dataset with additional columns.
        - test_scaled (DataFrame): Scaled testing dataset with additional columns.
        """
        # Scale the data
        train_scaled = pd.DataFrame(self.scaler.fit_transform(train_data[['Open', 'High', 'Low', 'Close', 'Volume']]),
                                    columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        test_scaled = pd.DataFrame(self.scaler.transform(test_data[['Open', 'High', 'Low', 'Close', 'Volume']]),
                                   columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Add Date column to scaled DataFrame
        train_scaled.insert(loc=0, column='Date', value=train_data['Date'].values)
        test_scaled.insert(loc=0, column='Date', value=test_data['Date'].values)

        return train_scaled, test_scaled

    def preprocess_pipeline(self):
        """
        Execute the full preprocessing pipeline: load, split, scale, and create sequences.

        Returns:
        - x_train (array): Training input sequences.
        - x_test (array): Testing input sequences.
        - y_train (array): Training labels.
        - y_test (array): Testing labels.
        - x_train_dates (array): Dates for the training input sequences.
        - x_test_dates (array): Dates for the testing input sequences.
        - y_train_dates (array): Dates for the training labels.
        - y_test_dates (array): Dates for the testing labels.
        """
        dataset = self.load_data()

        # Split data
        train_data, test_data = self.split_data(dataset)

        # Scale data
        train_scaled, test_scaled = self.scale_data(train_data, test_data)

        # Create sequences for multi-input LSTM
        x_train, y_train, x_train_dates, y_train_dates = self.create_sequences(train_scaled)
        x_test, y_test, x_test_dates, y_test_dates = self.create_sequences(test_scaled)

        return x_train, x_test, y_train, y_test, x_train_dates, x_test_dates, y_train_dates, y_test_dates
