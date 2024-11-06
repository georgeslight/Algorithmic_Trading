import os
from datetime import datetime, timedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from algorithmic_trading.experiment_1.src.data_plotter import DataPlotter
from data_collection import DataCollection
from lstm_stock_prediction import LSTMStockPredictor as lstm
from preprocessing import Preprocessing

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
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# Data collection
data_collector = DataCollection(tickers, start_date, end_date, folder_path=data_path)
data_collector.fetch_and_save_all()

# Preprocessing
pp = Preprocessing(folder_path=data_path, split_ratio=0.8, sequence_length=sequence_length)
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

##### MODEL SETUP #####

input_size = 5  # ['Open', 'High', 'Low', 'Close', 'Volume']
hidden_size = 50  # Number of hidden units in LSTM # todo 32, 50, 64, or 128
output_size = 5  # Predicting 5 values for the next day
num_layers = 2  # LSTM layers # todo 1 to 3
dropout = 0.2  # Dropout rate for regularization # todo 0.1 to 0.5
learning_rate = 0.001 # todo 0.001, 0.0005, 0.0001

# Instantiate the model
model = lstm(input_sz=input_size, hidden_sz=hidden_size, output_sz=output_size).to(device) # todo Bidirectional LSTM, Gated Recurrent Unit (GRU)

# Loss and optimizer
criterion = nn.MSELoss() # todo Mean Absolute Error (MAE = L1Loss()), Mean Squared Error (MSE)
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # todo Adam, AdamW, Adamax, ASGD

##### TRAINING THE MODEL #####

num_epochs = 50 # todo 50, 200
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.6f}")

##### EVALUATING THE MODEL #####
model.eval()
test_loss = 0.0
predictions = []
actuals = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        output, _ = model(x_batch)
        loss = criterion(output, y_batch)
        test_loss += loss.item()
        predictions.append(output.cpu())
        actuals.append(y_batch.cpu())

print(f"Test Loss: {test_loss / len(test_loader):.4f}")

# Post-processing predictions and actuals for inverse scaling
predictions = torch.cat(predictions).numpy()
actuals = torch.cat(actuals).numpy()
predictions_original = pp.scaler.inverse_transform(predictions)
actuals_original = pp.scaler.inverse_transform(actuals)

for i in range(5):
    print(f"Date: {y_test_dates[i]}")
    prediction_str = ", ".join([f"{x:.2f}" for x in predictions_original[i]])
    actual_str = ", ".join([f"{x:.2f}" for x in actuals_original[i]])
    print(f"Sample prediction (original scale): {prediction_str}")
    print(f"Actual values (original scale): {actual_str}")
    print("-" * 50)

# Plotting results
plotter = DataPlotter()
df_results = plotter.create_results_dataframe(y_test_dates, actuals_original, predictions_original)
plotter.plot_results(df_results)
