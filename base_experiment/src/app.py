from datetime import datetime, timedelta

from data_collection import DataCollection
from preprocessing import Preprocessing

##### DataCollection #####
predicted_days = 1

tickers = ["GOOGL"]
start_date = "2000-01-01"
end_date = datetime.today().date() - timedelta(days=predicted_days)

data_collector = DataCollection(tickers, start_date, end_date, folder_path="data")
data_collector.fetch_and_save_all()

##### Preprocessing #####
# Initialize Preprocessing with a 1-day shift for next-day prediction and a sequence length of 100 days
sequence_length = 100
pp = Preprocessing(folder_path="data", split_ratio=0.8, sequence_length=sequence_length)
x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = pp.preprocess_pipeline()

# Print samples from x_train and y_train
print("Sample input sequence (x_train):")
print(x_train_scaled[0])  # Print the first sequence (shape: (100, 5))

print("Sample corresponding label (y_train):")
print(y_train_scaled[0])  # Print the corresponding next day’s values (shape: (5,))

print('Train: x=%s, y=%s' % (x_train_scaled.shape, y_train_scaled.shape))
print('Test: x=%s, y=%s' % (x_test_scaled.shape, y_test_scaled.shape))