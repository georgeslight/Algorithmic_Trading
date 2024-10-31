import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta


class Preprocessing:
    def __init__(self, folder_path="data", split_ratio=0.8, shift_days=1):
        """
        Initialize the Preprocessing class.

        Parameters:
        - folder_path (str): Path to the folder where CSV data files are located.
        - split_ratio (float): Proportion of data to use as training data.
        - shift_days (int): Number of days to shift labels for the prediction target.
        """
        self.folder_path = folder_path
        self.split_ratio = split_ratio
        self.shift_days = shift_days
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

    def split_data(self, dataset):
        """
        Split the dataset chronologically into training and testing sets.

        Parameters:
        - dataset (DataFrame): The full dataset to split.

        Returns:
        - x_train, x_test, y_train, y_test (DataFrames): Training and testing sets for inputs and labels.
        """
        split_point = int(len(dataset) * self.split_ratio)
        train_data = dataset[:split_point]
        test_data = dataset[split_point:]

        x_train = train_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        y_train = train_data[['Open', 'High', 'Low', 'Close', 'Volume']].shift(
            -self.shift_days)  # Next day's data as labels
        x_test = test_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        y_test = test_data[['Open', 'High', 'Low', 'Close', 'Volume']].shift(-self.shift_days)

        # Drop NaN values in y_train and y_test created by shifting
        y_train = y_train.dropna()
        y_test = y_test.dropna()

        return x_train, x_test, y_train, y_test

    def scale_data(self, x_train, x_test, y_train, y_test):
        """
        Scale the training and testing data using MinMaxScaler.

        Parameters:
        - x_train, x_test, y_train, y_test (DataFrames): Training and testing sets for inputs and labels.

        Returns:
        - x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled (arrays): Scaled data for model input.
        """
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        y_train_scaled = self.scaler.transform(y_train)
        y_test_scaled = self.scaler.transform(y_test)

        return x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled

    def preprocess_pipeline(self):
        """
        Full preprocessing pipeline that loads, combines, splits, and scales data.

        Returns:
        - x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled (arrays): Preprocessed and scaled data.
        """
        dataset = self.load_and_combine_data()
        x_train, x_test, y_train, y_test = self.split_data(dataset)
        x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = self.scale_data(x_train, x_test, y_train, y_test)

        return x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled
