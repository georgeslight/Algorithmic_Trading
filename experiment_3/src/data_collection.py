import os

import pandas as pd
import yfinance as yf

class DataCollection:
    def __init__(self, tickers, start_date, end_date, interval="1d", folder_path="../data"):
        """
        Parameters:
        - tickers (list): List of stock tickers to fetch data for.
        - start_date (str): Start date for data collection in 'YYYY-MM-DD' format.
        - end_date (str): End date for data collection in 'YYYY-MM-DD' format.
        - interval (str): Data interval (e.g., '1d' for daily data).
        - folder_path (str): Path to the folder where data will be saved.
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.folder_path = folder_path

    def fetch_data(self, ticker):
        """
        Fetch historical stock data for a given ticker.

        Parameters:
        - ticker (str): Stock ticker symbol.

        Returns:
        - data (DataFrame): DataFrame containing the historical stock data.
        """
        data = yf.download(ticker, start=self.start_date, end=self.end_date, interval=self.interval)
        return data

    def save_data_to_csv(self, data, ticker):
        """
        Save the fetched data to a CSV file.

        Parameters:
        - data (DataFrame): DataFrame containing the historical stock data.
        - ticker (str): Stock ticker symbol.
        """
        data = data.reset_index()
        data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        data['Ticker'] = ticker

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        os.makedirs(self.folder_path, exist_ok=True)
        file_path = os.path.join(self.folder_path, f"{ticker}_data.csv")

        data.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")

    def fetch_and_save_all(self):
        """
        Fetch and save historical stock data for all tickers.

        Iterates over the list of tickers, fetches the data, and saves it to CSV files.
        """
        for ticker in self.tickers:
            data = self.fetch_data(ticker)
            if not data.empty:
                self.save_data_to_csv(data, ticker)
