from datetime import datetime, timedelta

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

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
criterion = nn.MSELoss() # todo Mean Absolute Error (MAE = L1Loss()), Mean Squared Error (MSE)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

# Generate a sample dataframe representing stock data for illustration
# Create a DataFrame with the test dates and the original-scaled predictions and actuals
df_results = pd.DataFrame({
    "Date": y_test_dates,
    "Actual_Open": actuals_original[:, 0],
    "Predicted_Open": predictions_original[:, 0],
    "Actual_High": actuals_original[:, 1],
    "Predicted_High": predictions_original[:, 1],
    "Actual_Low": actuals_original[:, 2],
    "Predicted_Low": predictions_original[:, 2],
    "Actual_Close": actuals_original[:, 3],
    "Predicted_Close": predictions_original[:, 3],
    "Actual_Volume": actuals_original[:, 4],
    "Predicted_Volume": predictions_original[:, 4]
})

# Plotting the actual vs predicted values for Open, High, Low, Close, and Volume
features = ["Open", "High", "Low", "Close", "Volume"]
fig, axes = plt.subplots(len(features), 1, figsize=(12, 15), sharex=True)

for i, feature in enumerate(features):
    axes[i].plot(df_results["Date"], df_results[f"Actual_{feature}"], label=f"Actual {feature}", linestyle="-")
    axes[i].plot(df_results["Date"], df_results[f"Predicted_{feature}"], label=f"Predicted {feature}", linestyle="--")
    axes[i].set_title(f"{feature} Price Over Time" if feature != "Volume" else "Volume Over Time")
    axes[i].legend()
    axes[i].set_ylabel(feature)
    axes[i].grid(True)

# Display date on x-axis for the last subplot only
axes[-1].set_xlabel("Date")
fig.tight_layout()
plt.show()
