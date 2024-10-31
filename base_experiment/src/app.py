from datetime import datetime, timedelta

from data_collection import DataCollection
from preprocessing import Preprocessing

##### DataCollection #####
predicted_days = 1

tickers = ["AAPL", "GOOGL", "BTC-USD"]
start_date = "2000-01-01"
end_date = datetime.today().date() - timedelta(days=predicted_days)

data_collector = DataCollection(tickers, start_date, end_date, folder_path="data")
data_collector.fetch_and_save_all()

# ##### Preprocessing #####
# Initialize Preprocessing with a 1-day shift for next-day prediction
pp = Preprocessing(folder_path="data", split_ratio=0.8, shift_days=1)
x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = pp.preprocess_pipeline()

print("x_train_scaled sample:")
print(x_train_scaled[:2])
print("x_test_scaled sample:")
print(x_test_scaled[:2])
print("y_train_scaled sample:")
print(y_train_scaled[:2])
print("y_test_scaled sample:")
print(y_test_scaled[:2])