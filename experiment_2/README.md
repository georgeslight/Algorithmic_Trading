# Experiment 2

## Short Description
Using a simple LSTM model to predict the next day's stock prices (Open, High, Low, Close, Volume) for a given stock. In this experiment, a distinct scaling method 

## Data Acquisition
Daily stock prices for Google (GOOGL) from 2000-01-01 to the present day. The data is split into training and testing sets with an 80-20 split.

## Features

The following features are used for each stock:  
- Open price
- High price
- Low price
- Close price
- Volume

In this experiment, the Open, High, Low, and Close features were scaled together using the MinMaxScaler, while the Volume feature was scaled independently with its own MinMaxScaler. This differs from the first experiment, where all features were scaled together.

## Target
The target is to predict the next day's stock prices (Open, High, Low, Close, Volume).

## Modeling Architecture
- **LSTM Architecture** model architecture builds upon the base configurations from Experiment 1 that performed very well compared to others:
  - Input Size: 5 (OHLCV)
  - Output Size: 5 (Predicting 5 values for the next day)
  - Hidden Size: 50 (Number of hidden units in the LSTM)
  - Sequence Length: 100 (Number of past days used as input)
  - Split Ratio: 80-20 (Training and testing split)
  - Batch Size: 32
  - Learning Rate: 0.001
  - Optimizer: Adam (optim.Adam with lr=0.001)
  - Loss Function: Mean Squared Error (MSE)
  - The model was trained with 50 epochs.

These base configurations were carried over from Experiment 1 due to their strong performance in terms of both epoch and test loss. The model was trained to predict the next day's stock prices (Open, High, Low, Close, Volume) using the LSTM architecture.
 
## Performance Criteria
The model's performance was evaluated by comparing the Mean Squared Error (MSE) of the predicted values with the Variance of the actual values for each feature (Open, High, Low, Close, Volume). A lower MSE than the Variance indicates the model outperforms a mean-based prediction for that feature.

## Baseline
The Mean Squared Error (MSE) of the predicted stock prices is compared with the Variance of the actual stock prices for each feature. This provides a reference point to determine if the model outperforms a simple mean-based prediction. A feature is considered well-predicted if its MSE is lower than its Variance.

## Results

### Overall Test Loss

 - Test Loss: 0.00279

### Comparison of MSE and Variance for Each Feature
The performance of the model was evaluated by comparing the Mean Squared Error (MSE) of the predicted values with the Variance of the actual values for each feature:

  - Open: MSE = 0.09564, Variance = 0.10168
  
    The model performs better than a mean-based prediction for Open.

  - High: MSE = 0.06550, Variance = 0.09563
    
    The model performs better than a mean-based prediction for High.

  - Low: MSE = 0.12649, Variance = 0.10446
    
    The model does not outperform a mean-based prediction for Low.

  - Close: MSE = 0.13148, Variance = 0.10110

    The model does not outperform a mean-based prediction for Close.

  - Volume: MSE = 0.01375, Variance = 0.00157
    
    The model does not outperform a mean-based prediction for Volume.

#### Interpretation
These findings indicate that further enhancements, such as improved feature scaling methods, adjustments to the model architecture, or more advanced data preprocessing techniques, may be necessary to achieve better performance for these features. While scaling the Volume feature separately in the 0-1 range using a different scaler was intended to address variance imbalances, it did not yield better results for Volume. Moreover, it led to worse predictions for the Low and Close features. This outcome suggests that scaling features independently might disrupt the relationships between them, particularly in time-series data where feature interdependence is critical. Future experiments should explore alternative scaling techniques that preserve these relationships while addressing variance discrepancies.