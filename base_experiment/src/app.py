from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data_collection import DataCollection
from preprocessing import Preprocessing
from lstm_stock_prediction import LSTMStockPredictor as lstm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##### DATA PREPARATION #####

# Parameters
predicted_days = 1
tickers = ["GOOGL"]
start_date = "2000-01-01"
end_date = datetime.today().date() - timedelta(days=predicted_days)
sequence_length = 100 # todo 30, 50, 100; longer sequence length provides more context but may also introduce more noise
batch_size = 32 # todo 16, 32, 64

# Data collection
data_collector = DataCollection(tickers, start_date, end_date, folder_path="data")
data_collector.fetch_and_save_all()

# Preprocessing
pp = Preprocessing(folder_path="data", split_ratio=0.8, sequence_length=sequence_length)
x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled, x_train_dates, x_test_dates, y_train_dates, y_test_dates = pp.preprocess_pipeline()

# Convert data to PyTorch tensors
x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

# Create DataLoader for batching
train_dataset = TensorDataset(x_train_tensor, y_train_tensor) # TensorDataset is a PyTorch utility that pairs input and target tensors together
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Print samples from x_train and y_train
print("Sample input sequence (x_train):")
print(x_train_scaled[0])  # Print the first sequence (shape: (100, 5))
##### MODEL SETUP #####

input_size = 5  # ['Open', 'High', 'Low', 'Close', 'Volume']
hidden_size = 50  # Number of hidden units in LSTM # todo 32, 50, 64, or 128
output_size = 5  # Predicting 5 values for the next day
num_layers = 2  # Two LSTM layers # todo 1 to 3
dropout = 0.2  # Dropout rate for regularization # todo 0.1 to 0.5
learning_rate = 0.001 # todo 0.001, 0.0005, 0.0001

# Instantiate the model
model = lstm(input_sz=input_size, hidden_sz=hidden_size, output_sz=output_size).to(device) # todo Bidirectional LSTM, Gated Recurrent Unit (GRU)

# Loss and optimizer
criterion = nn.MSELoss() # todo Mean Absolute Error (MAE)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


print("Sample corresponding label (y_train):")
print(y_train_scaled[0])  # Print the corresponding next day’s values (shape: (5,))

print('Train: x=%s, y=%s' % (x_train_scaled.shape, y_train_scaled.shape))
print('Test: x=%s, y=%s' % (x_test_scaled.shape, y_test_scaled.shape))