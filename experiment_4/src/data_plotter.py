import pandas as pd
from matplotlib import pyplot as plt


class DataPlotter:
    def create_results_dataframe(self, dates, actuals_original, predictions_original):
        """
        Create a DataFrame with dates, actual values, and predicted values.

        Parameters:
        - dates (list): List of dates.
        - actuals_original (numpy array): Array of actual values.
        - predictions_original (numpy array): Array of predicted values.

        Returns:
        - DataFrame: DataFrame containing dates, actual values, and predicted values.
        """
        flattened_dates = []
        flattened_actuals = []
        flattened_predictions = []

        for i in range(len(dates)):
            flattened_dates.append(dates[i])  # Append date for each day
            flattened_actuals.append(actuals_original[i])  # Append actual value
            flattened_predictions.append(predictions_original[i])  # Append predicted value

        df_results = pd.DataFrame({
            'Date': flattened_dates,
            'Actual': flattened_actuals,
            'Predicted': flattened_predictions
        })

        return df_results

    def plot_results(self, df_results):
        """
        Plot the actual vs. predicted values.

        Parameters:
        - df_results (DataFrame): DataFrame containing dates, actual values, and predicted values.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(df_results['Date'], df_results['Actual'], label='Actual')
        plt.plot(df_results['Date'], df_results['Predicted'], label='Predicted')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Actual vs. Predicted Values')
        plt.legend()
        plt.show()