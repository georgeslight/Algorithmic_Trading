import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Preprocessing:
    def __init__(self, folder_path="data", split_ratio=0.8, sequence_length=100):
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

    def load_and_combine_data(self):
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
        dataset['Date'] = pd.to_datetime(dataset['Date'])
        dataset = dataset.sort_values('Date').reset_index(drop=True)
        return dataset

    def create_sequences(self, data):
        """
        Create sequences of `sequence_length` days with the next day's data as the label.

        Parameters:
        - data (DataFrame): Scaled DataFrame of stock data.

        Returns:
        - x, y (arrays): Sequences and corresponding next-day labels.
        """
        x, y = [], []
        for i in range(len(data) - self.sequence_length):
            # Input sequence of past `sequence_length` days
            x.append(data.iloc[i:i + self.sequence_length][['Open', 'High', 'Low', 'Close', 'Volume']].values)
            # Label is the next day's price data
            y.append(data.iloc[i + self.sequence_length][['Open', 'High', 'Low', 'Close', 'Volume']].values)

        return np.array(x), np.array(y)

    def split_data(self, dataset):
        """
        Split the dataset chronologically into training and testing sets, then create sequences.

        Parameters:
        - dataset (DataFrame): The full dataset to split.

        Returns:
        - x_train, x_test, y_train, y_test (arrays): Training and testing sets for inputs and labels.
        """
        split_point = int(len(dataset) * self.split_ratio)
        train_data = dataset[:split_point]
        test_data = dataset[split_point:]

        # Scale the data
        train_scaled = pd.DataFrame(self.scaler.fit_transform(train_data[['Open', 'High', 'Low', 'Close', 'Volume']]),
                                    columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        test_scaled = pd.DataFrame(self.scaler.transform(test_data[['Open', 'High', 'Low', 'Close', 'Volume']]),
                                   columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Create sequences for training and testing
        x_train, y_train = self.create_sequences(train_scaled)
        x_test, y_test = self.create_sequences(test_scaled)

        return x_train, x_test, y_train, y_test

    def preprocess_pipeline(self):
        """
        Full preprocessing pipeline that loads, combines, splits, and scales data.

        Returns:
        - x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled (arrays): Preprocessed and scaled data.
        """
        dataset = self.load_and_combine_data()
        x_train, x_test, y_train, y_test = self.split_data(dataset)

        return x_train, x_test, y_train, y_test
