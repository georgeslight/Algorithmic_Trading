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

#### Overall Test Loss

 - Test Loss: 0.00120

#### Comparison of MSE and Variance for Each Feature

  - Open: MSE = 0.03589, Variance = 0.10168 
  
    The model performs better than a mean-based prediction for Open.

  - High: MSE = 0.05428, Variance = 0.09563
    
    The model performs better than a mean-based prediction for High.

  - Low: MSE = 0.03833, Variance = 0.10446
    
    The model performs better than a mean-based prediction for Low.

  - Close: MSE = 0.05300, Variance = 0.10110

    The model performs better than a mean-based prediction for Close.

  - Volume: MSE = 0.00498, Variance = 0.00157
    
    The model does not outperform a mean-based prediction for Volume.

These results indicate that the model performs well for predicting the 'Open', 'High', 'Low', and 'Close' prices, but it does not perform as well for predicting the 'Volume' feature.

##### Detailed Results

The results for different configurations are saved in the 'outputs' directory:
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

- **Feature Scaling and Variance**
  - A notable discrepancy was observed in the variance of the Volume feature during evaluation, which consistently demonstrated much lower variance compared to the other features. While this issue may be attributed to the use of MinMax scaling—compressing the range of features to [0, 1] and disproportionately reducing the variance of the significantly larger Volume values—it is also possible that the behavior arises from intrinsic properties of the data or the model's treatment of this feature. To address this imbalance and investigate its origins, future experiments should explore alternative scaling methods.