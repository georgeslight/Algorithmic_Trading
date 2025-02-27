import pandas as pd
import matplotlib.pyplot as plt

class DataPlotter:
    def __init__(self):
        pass

    def create_results_dataframe(self, dates, actuals_original, predictions_original):
        """
        Create a DataFrame with the given dates and the original-scaled predictions and actuals.

        Parameters:
        - dates (array-like): Array of dates corresponding to the given data.
        - actuals_original (array-like): Array of actual values in the original scale.
        - predictions_original (array-like): Array of predicted values in the original scale.

        Returns:
        - DataFrame: DataFrame containing the dates, actual values, and predicted values.
        """
        df_results = pd.DataFrame({
            "Date": dates,
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
        return df_results

    def plot_results(self, df_results):
        """
        Plot the actual vs. predicted values for Open, High, Low, and Close prices.

        Parameters:
        - df_results (DataFrame): DataFrame containing the dates, actual values, and predicted values.
        """
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
