import math


class InvestmentSimulator:
    def __init__(self, prices, initial_capital, profit_threshold, loss_threshold):
        """
        Initialize the InvestmentSimulator class.

        Parameters:
        - prices: np.array, stock prices (assumes 'Close' prices are in column index 3).
        - initial_capital: float, initial investment amount.
        - profit_threshold: float, target profit percentage (e.g., 0.10 for 10%).
        - loss_threshold: float, target loss percentage (e.g., -0.05 for 5%).
        """
        self.prices = prices  # Array of stock prices
        self.initial_capital = initial_capital
        self.profit_threshold = profit_threshold
        self.loss_threshold = loss_threshold

    def simulate(self):
        """
        Simulate the investment.

        Parameters:
        - starting_day: int, index of the day to start the simulation.
        - num_days: int, number of days to simulate.

        Returns:
        - result: dict, with keys:
            - "days_to_threshold": Number of days it took to reach the threshold.
            - "final_profit_loss": Profit or loss achieved at the threshold.
            - "status": 'profit', 'loss', or 'end' (if no threshold was reached).
        """
        # Initialize variables
        stock_price = self.prices[0, 3]  # Buy at the 'Close' price on the starting day
        total_stock = math.floor(self.initial_capital / stock_price)  # Number of shares bought
        starting_date = 0
        num_days = len(self.prices)
        buy_price = stock_price * total_stock
        profit_loss = 0

        # Simulate daily price changes
        for day in range(0, num_days):
            today_price = self.prices[day, 3]  # Today's 'Close' price
            sell_price = today_price * total_stock   # Value of stock today
            profit_loss = (sell_price - buy_price) / buy_price
            print(f"Day {day}: Profit/Loss: {profit_loss:.2%}")

            # Check profit or loss thresholds
            if profit_loss >= self.profit_threshold:
                return {
                    "total_stock": total_stock,
                    "stock_price": stock_price,
                    "days_to_threshold": day - starting_date,
                    "final_profit_loss": sell_price - self.initial_capital,
                    "status": "profit"
                }
            elif profit_loss <= self.loss_threshold:
                return {
                    "total_stock": total_stock,
                    "stock_price": stock_price,
                    "days_to_threshold": day - starting_date,
                    "final_profit_loss": sell_price - self.initial_capital,
                    "status": "loss"
                }
        return {
            "status": "end"
        }