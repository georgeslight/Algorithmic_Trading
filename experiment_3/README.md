# Experiment 3

## Short Description
This experiment utilizes a Multi-Input LSTM model to predict the next day's stock prices (Open, High, Low, Close, Volume) for a given stock. Each feature is processed independently through separate LSTM layers, aiming to leverage specialized representations for each feature while predicting all target values simultaneously.

## Data Acquisition
Daily stock prices for Google (GOOGL) from 2000-01-01 to the present day. The data is split into training and testing sets with an 80-20 split.

## Features

The following features are used for each stock:  
- Open price
- High price
- Low price
- Close price
- Volume

In this experiment, all features were scaled together using the MinMaxScaler to retain their interdependencies. Each feature was then passed independently into its corresponding LSTM layer for specialized processing.

## Target
The target is to predict the next day's stock prices (Open, High, Low, Close, Volume).

## Modeling Architecture
The Multi-Input LSTM architecture is configured as follows:
- Separate LSTM layers for each feature (Open, High, Low, Close, Volume).
  - Input Size: 1 (Single feature per LSTM layer).
  - Hidden Size: 50 (Number of hidden units in each LSTM).
  - Number of Layers: 4.
  - Dropout: 0.2.
- Fully Connected Layer: Concatenates hidden states from all LSTM layers and maps to the output.
  - Input Size: Hidden Size × 5 (for five features).
  - Output Size: 5 (Predicting next day's Open, High, Low, Close, Volume values).
- Loss Function: Mean Squared Error (MSE).
- Optimizer: Adam (optim.Adam with lr=0.001).
- Sequence Length: 100 days.
- Batch Size: 32.
- Number of Epochs: 50.

## Performance Criteria
The model's performance was evaluated by comparing the Mean Squared Error (MSE) of the predicted values with the Variance of the actual values for each feature. A lower MSE than the Variance indicates the model outperforms a mean-based prediction for that feature.

## Baseline
The Mean Squared Error (MSE) of the predicted stock prices is compared with the Variance of the actual stock prices for each feature. This provides a benchmark to determine if the model outperforms a simple mean-based prediction.

## Results

### Overall Test Loss

 - Test Loss: 0.13976

### Comparison of MSE and Variance for Each Feature
The performance of the model was evaluated by comparing the Mean Squared Error (MSE) of the predicted values with the Variance of the actual values for each feature:

  - Open: MSE = 5.30164, Variance = 0.10168
  
    The model does not outperform a mean-based prediction for Open.

  - High: MSE = 4.98063, Variance = 0.09563
    
    The model does not outperform a mean-based prediction for High.

  - Low: MSE = 5.76715, Variance = 0.10446
    
    The model does not outperform a mean-based prediction for Low.

  - Close: MSE = 5.80307, Variance = 0.10110

    The model does not outperform a mean-based prediction for Close.

  - Volume: MSE = 0.00313, Variance = 0.00157
    
    The model does not outperform a mean-based prediction for Volume.

#### Interpretation
The results indicate that the Multi-Input LSTM architecture fails to outperform a mean-based prediction for any of the features. The significantly higher MSE values compared to variance suggest that the model struggles to capture the underlying patterns in the data.

This poor performance may stem from:

  - Feature Independence: Processing each feature in separate LSTM layers may disrupt inter-feature dependencies critical for stock price predictions.
  - Complexity: The added complexity of multiple LSTM layers could lead to overfitting on training data, especially with limited data points per feature.
  - Data Scaling: While MinMaxScaler retained interdependencies between features, alternative scaling methods may yield better results.