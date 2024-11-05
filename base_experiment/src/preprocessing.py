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
        dataset['Date'] = pd.to_datetime(dataset['Date'])
        dataset = dataset.sort_values('Date').reset_index(drop=True)
        return dataset

    def create_sequences(self, data, dates):
        """
        Create sequences of `sequence_length` days with the next day's data as the label.

        Parameters:
        - data (DataFrame): Scaled DataFrame of stock data.

        Returns:
        - x, y (arrays): Sequences and corresponding next-day labels.
        """
        x, y = [], []
        x_dates, y_dates = [], []  # To store dates for input and label
        for i in range(len(data) - self.sequence_length):
            # Input sequence of past `sequence_length` days
            x.append(data.iloc[i:i + self.sequence_length][['Open', 'High', 'Low', 'Close', 'Volume']].values)
            x_dates.append(dates[i:i + self.sequence_length].values)  # Store dates for the input sequence
            # Label is the next day's price data
            y.append(data.iloc[i + self.sequence_length][['Open', 'High', 'Low', 'Close', 'Volume']].values)
            y_dates.append(dates.iloc[i + self.sequence_length])  # Store date for the label

        # return np.array(x), np.array(y)
        return np.array(x), np.array(y), np.array(x_dates), np.array(y_dates)

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

        # Extract dates for training and testing sets
        train_dates = train_data['Date']
        test_dates = test_data['Date']

        # Scale the data
        train_scaled = pd.DataFrame(self.scaler.fit_transform(train_data[['Open', 'High', 'Low', 'Close', 'Volume']]),
                                    columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        test_scaled = pd.DataFrame(self.scaler.transform(test_data[['Open', 'High', 'Low', 'Close', 'Volume']]),
                                   columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # # Create sequences for training and testing
        # x_train, y_train = self.create_sequences(train_scaled)
        # x_test, y_test = self.create_sequences(test_scaled)
        #
        # return x_train, x_test, y_train, y_test
        x_train, y_train, x_train_dates, y_train_dates = self.create_sequences(train_scaled, train_dates)
        x_test, y_test, x_test_dates, y_test_dates = self.create_sequences(test_scaled, test_dates)

        return x_train, x_test, y_train, y_test, x_train_dates, x_test_dates, y_train_dates, y_test_dates

    def preprocess_pipeline(self):
        """
        Full preprocessing pipeline that loads, combines, splits, and scales data.

        Returns:
        - x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled (arrays): Preprocessed and scaled data.
        """
        dataset = self.load_data()
        x_train, x_test, y_train, y_test, x_train_dates, x_test_dates, y_train_dates, y_test_dates = self.split_data(
            dataset)

        return x_train, x_test, y_train, y_test, x_train_dates, x_test_dates, y_train_dates, y_test_dates
