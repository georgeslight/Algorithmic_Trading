# Experiment 1

## Short Description
This experiment utilizes a Long Short-Term Memory (LSTM) model to predict stock prices one day ahead based on historical OHLCV (Open, High, Low, Close, Volume) data. 
Multiple configurations are tested to identify an optimal setup for minimizing prediction error.

## Data Acquisition
- Data source: Historical daily stock data from yfinance (Yahoo Finance) for the ticker "GOOGL".
- Data Range: 2000-01-01 to the current date, excluding the predicted day.
- Target: The experiment uses OHLCV data from past stock metrics to predict the OHLCV for the next day.


## Features
The following features are calculated and normalized as input for the LSTM model:
- **OHLCV**: 'Open', 'High', 'Low', 'Close', 'Volume'.
- Each sequence consists of data from the past 30, 50, 100, or 365 days, depending on the configuration.

## Target
Predict the values of 'Open', 'High', 'Low', 'Close', and 'Volume' for the next day.

## Modeling Architecture
- **LSTM Architecture** model with configurations varied across hyperparameters:
  - Input size: 5 (OHLCV)
  - Hidden size: Tested values include 32, 50, 64, and 128
  - Number of layers: Configurations range from 2 to 3 layers
  - Dropout: 0.2 or 0.5
  - Sequence Length: Configurations tested include 30, 50, 100, and 365 days
  - Optimizer: Adam and AdamW optimizers with optional weight decay
  - Loss Function: Both Mean Squared Error (MSE) and Mean Absolute Error (MAE) were tested
 
## Performance Criteria
Performance is evaluated based on the following metrics:
- Epoch Loss: Average loss over each training epoch.
- Test Loss: Average loss over the test set.

## Baseline
Comparison between Mean Squared Error (MSE) and Mean Absolute Error (MAE) as loss functions:
- **MSE**: Generally yields lower epoch and test loss values, indicating effective minimization of error during training.
- **MAE**: Provides closer alignment with actual values across most stock metrics (Open, High, Low, and Close) when analyzing the plots, despite showing slightly higher loss values. Both metrics, however, struggled with accurately predicting volume trends.

## Results
The results are saved in the 'outputs' directory:
- 'results.csv': Contains details of each configuration tested, including hyperparameters, epoch loss, and test loss for each setup.
- Plots Directory: Each experiment’s actual vs. predicted results are visualized in the outputs/plots/ directory. Plot files are named based on the plot_id column in results.csv. For instance, results_1.png corresponds to the tuple with 'plot_id = 1' in the CSV file.

## Key Findings
- **Initial Best Configuration**:
  - Looking at the loss values, the first configuration achieved the best initial results for epoch and test loss, with an epoch loss of 0.000492 and test loss of 0.0067. This configuration provided a baseline understanding of model performance.

| plot_id | sequence_length | batch_size | hidden_size | num_layers | dropout | learning_rate | criterion | optimizer | num_epochs | epoch_loss | test_loss |
|---------|-----------------|------------|-------------|------------|---------|---------------|-----------|-----------|------------|------------|-----------|
| 1       | 100             | 32         | 50          | 2          | 0.2     | 0.001         | MSE       | Adam      | 50         | 0.000492   | 0.0067    |

- **Overfitting and Best Overall Configuration**:
  - Despite the promising results from configuration 7, this setup demonstrated signs of overfitting, as the significant drop in test loss did not align with consistent prediction accuracy across the dataset.

| plot_id | sequence_length | batch_size | hidden_size | num_layers | dropout | learning_rate | criterion | optimizer                      | num_epochs | epoch_loss | test_loss |
|---------|-----------------|------------|-------------|------------|---------|---------------|-----------|--------------------------------|------------|------------|-----------|
| 7       | 365             | 64         | 32          | 3          | 0.5     | 0.001         | MSE       | AdamW (with weight_decay=0.01) | 50         | 0.000345   | 0.0239    |

- **Best Balanced Configuration**
  - This configuration provided a better balance between underfitting and overfitting, delivering consistent predictions and maintaining low loss metrics across the board. It was found to be the most effective for achieving reliable, real-world predictions.

| plot_id | sequence_length | batch_size | hidden_size | num_layers | dropout | learning_rate | criterion | optimizer | num_epochs | epoch_loss | test_loss |
|---------|-----------------|------------|-------------|------------|---------|---------------|-----------|-----------|------------|------------|-----------|
| 13      | 100             | 32         | 128         | 2          | 0.2     | 0.0001        | MSE       | Adam      | 200        | 0.000478   | 0.0025    |

- **MSE vs. MAE**:
  - Although MSE configurations yielded lower numerical loss values, visual inspection of prediction plots showed that MAE configurations produced closer alignment with actual values across most stock metrics (Open, High, Low, and Close).
  - Volume Prediction: Both MSE and MAE configurations struggled with accurately predicting Volume, with MAE showing greater variance between actual and predicted values. This suggests that MAE might not be ideal for capturing volume trends.