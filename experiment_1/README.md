# Experiment 1

## Short Description
Using a simple LSTM model to predict the next day's stock prices (Open, High, Low, Close, Volume) for a given stock.

## Data Acquisition
Daily stock prices for Google (GOOGL) from 2000-01-01 to the present day. The data is split into training and testing sets with an 80-20 split.

## Features

The following features are used for each stock:  
- Open price
- High price
- Low price
- Close price
- Volume

These features are normalized over the last 30, 50, 100, or 365 days, depending on the configuration.

## Target
The target is to predict the next day's stock prices (Open, High, Low, Close, Volume)

## Modeling Architecture
- **LSTM Architecture** model with configurations varied across hyperparameters:
  - Input size: 5 (OHLCV)
  - Hidden size: Tested values include 32, 50, 64, and 128
  - Sequence Length: Configurations tested include 30, 50, 100, and 365 days
  - Batch Size: Tested values include 32 and 64
  - Learning Rate: Tested values include 0.001 and 0.0001
  - Optimizer: Adam and AdamW optimizers with optional weight decay
  - Loss Function: Both Mean Squared Error (MSE) and Mean Absolute Error (MAE) were tested
  - The model was trained with 50 to 200 epochs, depending on the configuration
 
## Performance Criteria
Mean Squared Error (MSE) or Mean Absolute Error (MAE) of true vs. predicted stock prices for the test data.

## Baseline
The baseline for this experiment is the Mean Squared Error (MSE) and Mean Absolute Error (MAE) of the true stock prices compared to the average stock prices over the training period for the test data. This provides a reference point to evaluate the performance of the LSTM model.

## Results

In this experiment, we explored the use of a simple LSTM model to predict the next day's stock prices for Google (GOOGL). Various configurations of the LSTM model were tested, and their performance was evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE) as loss functions. The results indicated that while the LSTM model could capture some trends in the stock prices, further improvements and more complex models may be needed to achieve better predictions. The findings from this experiment provide a foundation for future work in developing more accurate stock price prediction models.

The results are saved in the 'outputs' directory:
- 'results.csv': Contains details of each configuration tested, including hyperparameters, epoch loss, and test loss for each setup.
- Plots Directory: Each experiment’s actual vs. predicted results are visualized in the outputs/plots/ directory. Plot files are named based on the plot_id column in results.csv. For instance, results_1.png corresponds to the tuple with 'plot_id = 1' in the CSV file.

## Key Findings
- **Initial Best Configuration**:
  - Looking at the loss values, the first configuration achieved the best initial results for epoch and test loss, with an epoch loss of 0.000492 and test loss of 0.0067. This configuration provided a baseline understanding of model performance.

| plot_id | sequence_length | batch_size | hidden_size | learning_rate | criterion | optimizer | num_epochs | epoch_loss | test_loss |
|---------|-----------------|------------|-------------|---------------|-----------|-----------|------------|------------|-----------|
| 1       | 100             | 32         | 50          | 0.001         | MSE       | Adam      | 50         | 0.000492   | 0.0067    |

- **Overfitting and Best Overall Configuration**:
  - Despite the promising results from configuration 7, this setup demonstrated signs of overfitting, as the significant drop in test loss did not align with consistent prediction accuracy across the dataset.

| plot_id | sequence_length | batch_size | hidden_size | learning_rate | criterion | optimizer                      | num_epochs | epoch_loss | test_loss |
|---------|-----------------|------------|-------------|---------------|-----------|--------------------------------|------------|------------|-----------|
| 7       | 365             | 64         | 32          | 0.001         | MSE       | AdamW (with weight_decay=0.01) | 50         | 0.000345   | 0.0239    |

- **Best Balanced Configuration**
  - This configuration provided a better balance between underfitting and overfitting, delivering consistent predictions and maintaining low loss metrics across the board. It was found to be the most effective for achieving reliable, real-world predictions.

| plot_id | sequence_length | batch_size | hidden_size | learning_rate | criterion | optimizer | num_epochs | epoch_loss | test_loss |
|---------|-----------------|------------|-------------|---------------|-----------|-----------|------------|------------|-----------|
| 13      | 100             | 32         | 128         | 0.0001        | MSE       | Adam      | 200        | 0.000478   | 0.0025    |

