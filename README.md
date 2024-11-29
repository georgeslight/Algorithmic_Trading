# Algorithmic Trading with LSTM Networks
## General Description and Goal of the Project

This project explores various machine learning models based on Long Short-Term Memory (LSTM) networks for predicting daily stock prices. The primary objective is to develop models capable of forecasting key stock metrics—Open, High, Low, Close, and Volume—for the next day using historical data. By testing different architectures, scaling methods, and configurations, the project seeks to identify the best approaches for improving prediction accuracy across stock features.

The dataset used in this project consists of historical stock prices for Google (GOOGL) spanning from its IPO on 2004-08-19 to the present, fetched using the yfinance library. The data includes the following features: Open, High, Low, Close, and Volume (OHLCV). The models aim to predict these metrics or a subset of them, with results evaluated using Mean Squared Error (MSE) as the primary performance metric.

The project includes multiple experiments, each designed to test a specific hypothesis or address issues observed in previous experiments.

## Summary and Results Across Experiments
### Experiment 1: Baseline LSTM Model

- **Goal:** Predict all five features (OHLCV) using a simple LSTM model.
- **Key Findings:**
    - The model outperformed a mean-based baseline for most features except Volume.
    - Variance in the Volume feature posed challenges, possibly due to the scaling method.
- **Best Configuration:** Sequence length of 100 days, hidden size of 128, MSE loss, Adam optimizer, learning rate of 0.0001.
- **Overall Test Loss:** 0.0025

### Experiment 2: LSTM with Independent Volume Scaling

- **Goal:** Address variance imbalance in the Volume feature by scaling it separately from the other features.
- **Key Findings:**
    - The Volume predictions did not improve significantly.
    - Independent scaling disrupted interdependencies among features, leading to worse predictions for the Low and Close prices.
- **Overall Test Loss:** 0.00279

### Experiment 3: Multi-Input LSTM

- **Goal:** Process each feature independently through separate LSTM layers for specialized representation.
- **Key Findings:**
    - The model failed to capture inter-feature dependencies, resulting in poor performance across all features.
    - Complexity of the architecture likely contributed to overfitting.
    - The configuration used may have led to a highly decoupled representation of the features, disrupting the natural relationships essential for accurate stock price predictions.
- **Overall Test Loss:** 0.13976

### Experiment 4: Bidirectional LSTM with Attention Mechanism

- **Goal:** Enhance model performance with bidirectional LSTM layers and an attention mechanism to focus on relevant sequence features.
- **Key Findings:**
    - The model failed to outperform the baseline for the 'Close' price.
    - Possible causes include overfitting, errors in model configuration, and challenges in integrating the attention mechanism effectively.
- **Overall Test Loss:** 0.31830

## Overall Insights and Conclusions

1. **Baseline LSTM (Experiment 1)** provided the most consistent results, with strong predictions for Open, High, Low, and Close prices. However, Volume remains a challenging feature to predict due to its variance and range.
2. **Feature Scaling (Experiment 2)** highlights the importance of maintaining inter-feature dependencies in time-series data.
3. **Specialized Representations (Experiment 3)** are not always beneficial, as separate processing of features disrupted critical relationships between them.
4. **Advanced Architectures (Experiment 4)** like bidirectional LSTMs with attention mechanisms require careful configuration and may not always yield better performance, especially for highly noisy data.
