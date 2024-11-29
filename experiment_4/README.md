# Experiment 4

## Short Description
In this experiment, a bidirectional LSTM architecture with attention mechanism was implemented to predict the next day's stock price for the "Close" value using historical data. Four bidirectional LSTM layers were stacked, each followed by an attention mechanism to focus on the most relevant features within the input sequence.

## Data Acquisition
Historical stock data for Google (GOOGL) was fetched using the yfinance library. The data spans from 2000-01-01 to the present day and includes daily Open, High, Low, Close, and Volume values. The dataset was split into training and testing sets with an 80-20 split.

## Features

The following features are used for each stock:  
- Open price
- High price
- Low price
- Close price
- Volume

## Target
The model aims to predict the "Close" price for the next day.

## Modeling Architecture
The model architecture consisted of:
- Bidirectional LSTM Layers:
  - Layer 1: Input size = 5 (OHLCV features), hidden size = 50, bidirectional, dropout = 0.2
  - Layer 2: Input size = 100 (bidirectional output from Layer 1), hidden size = 50, bidirectional, dropout = 0.2
  - Layer 3: Input size = 100 (bidirectional output from Layer 2), hidden size = 50, bidirectional, dropout = 0.2
  - Layer 4: Input size = 100 (bidirectional output from Layer 3), hidden size = 50, bidirectional, dropout = 0.2
- Attention Mechanism: Applied to the output of the fourth LSTM layer to assign weights to sequence features.  - Input Size: Hidden Size × 5 (for five features).
- Fully Connected Layer: A linear layer mapping the attention context vector (dimension = 100) to the output size of 1 (predicted "Close" value).

### Hyperparameters
  - Input size: 5 (OHLCV)
  - Hidden size: 50
  - Number of LSTM layers: 2 (bidirectional)
  - Sequence length: 100 (days)
  - Batch size: 32
  - Learning rate: 0.001
  - Dropout: 0.2
  - Loss function: Mean Squared Error (MSE)
  - Optimizer: Adam
  - Number of epochs: 50
## Performance Criteria
The model's performance was evaluated based on:
  - Mean Squared Error (MSE): Comparing predicted "Close" values with actual values.
  - Variance: The variance of the actual values was used as a baseline. The model was considered effective if the MSE was smaller than the variance.

## Baseline
The baseline for this experiment was the variance of the "Close" price in the test dataset. A model that achieves an MSE lower than the variance can be considered better than a mean-based prediction.

## Results

### Overall Test Loss

 - Test Loss: 1.19728

### Comparison of MSE and Variance for Each Feature
The performance of the model was evaluated by comparing the Mean Squared Error (MSE) of the predicted values with the Variance of the actual values for the 'Close' feature:

  - Close: MSE = 37.44753, Variance = 0.40440
  
  The model's performance was significantly worse compared to the baseline. Despite using a sophisticated architecture with bidirectional LSTMs and an attention mechanism, the predictions were less accurate than a simple mean-based approach.

 #### Key Observations

  1. Overfitting Issues:
     - The model may have overfitted the training data due to the complex architecture and insufficient generalization to unseen data.
  2. Attention Mechanism:
     - While the attention mechanism was expected to improve focus on relevant sequence features, its integration might not have effectively captured the necessary relationships in the stock price data.
  3. Hyperparameter Challenges:
     - The selected hyperparameters, such as learning rate, sequence length, and dropout, might need further optimization to balance model complexity and performance.axScaler retained interdependencies between features, alternative scaling methods may yield better results.

### Conclusion

The bidirectional LSTM with an attention mechanism failed to achieve satisfactory results for stock price prediction. Despite the complexity of the model, it was unable to outperform a simple mean-based prediction baseline, indicating potential issues with the data, architecture, or training process. The test loss and MSE were significantly higher than expected, suggesting that the model struggled to generalize to unseen data. Possible causes for these shortcomings include overfitting or errors in the model configuration. Future experiments with adapted configurations and hyperparameter tuning should be conducted to evaluate whether these adjustments can improve the outputs and enhance the model's predictive capabilities.