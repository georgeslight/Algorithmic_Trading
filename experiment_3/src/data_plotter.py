import pandas as pd
import matplotlib.pyplot as plt

class DataPlotter:
    def __init__(self):
        pass

    def create_results_dataframe(self, dates, actuals_original, predictions_original):
        """
        Create a DataFrame with test dates and corresponding actual and predicted 'Close' values for multiple days.
        """
        flattened_dates = []
        flattened_actuals = []
        flattened_predictions = []

        # Flatten predictions and actuals with corresponding dates
        for i in range(len(dates)):
            for j in range(actuals_original.shape[1]):  # Iterate over output days
                flattened_dates.append(dates[i][j])  # Append date for each day
                flattened_actuals.append(actuals_original[i][j])  # Append actual value
                flattened_predictions.append(predictions_original[i][j])  # Append predicted value

        # Create the DataFrame
        df_results = pd.DataFrame({
            "Date": flattened_dates,
            "Actual_Close": flattened_actuals,
            "Predicted_Close": flattened_predictions,
        })
        return df_results

    def plot_results(self, df_results):
        """
        Plot the actual vs. predicted 'Close' values for multiple days.
        """
        df_results = df_results.sort_values("Date")  # Ensure dates are sorted
        plt.figure(figsize=(12, 6))
        plt.plot(df_results["Date"], df_results["Actual_Close"], label="Actual Close", linestyle="-")
        plt.plot(df_results["Date"], df_results["Predicted_Close"], label="Predicted Close", linestyle="--")
        plt.title("Close Price Over Time")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        # """
        # Plot the actual vs. predicted values for Open, High, Low, Close, and Volume.
        # """
        # features = ["Open", "High", "Low", "Close", "Volume"]
        # fig, axes = plt.subplots(len(features), 1, figsize=(12, 15), sharex=True)
        #
        # for i, feature in enumerate(features):
        #     axes[i].plot(df_results["Date"], df_results[f"Actual_{feature}"], label=f"Actual {feature}", linestyle="-")
        #     axes[i].plot(df_results["Date"], df_results[f"Predicted_{feature}"], label=f"Predicted {feature}", linestyle="--")
        #     axes[i].set_title(f"{feature} Price Over Time" if feature != "Volume" else "Volume Over Time")
        #     axes[i].legend()
        #     axes[i].set_ylabel(feature)
        #     axes[i].grid(True)
        #
        # # Display date on x-axis for the last subplot only
        # axes[-1].set_xlabel("Date")
        # fig.tight_layout()
        # plt.show()
