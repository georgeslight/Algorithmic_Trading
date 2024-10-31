import os
from datetime import datetime

import pandas as pd
import yfinance as yf

class DataCollection:
    def __init__(self, tickers, start_date, end_date, interval="1d", folder_path="data"):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.folder_path = folder_path

    def fetch_data(self, ticker):
        data = yf.download(ticker, start=self.start_date, end=self.end_date, interval=self.interval)
        return data

    def save_data_to_csv(self, data, ticker):
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
        for ticker in self.tickers:
            data = self.fetch_data(ticker)
            if not data.empty:
                self.save_data_to_csv(data, ticker)
