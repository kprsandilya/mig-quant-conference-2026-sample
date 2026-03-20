import numpy as np

def get_actions(prices: np.ndarray) -> np.ndarray:
    """
    Arguments:
        prices  -- np.ndarray of shape (num_stocks, num_days)
                   Contains the Open price for each stock on each day.
                   Rows = stocks (sorted by ticker), Columns = days.

    Returns:
        actions -- np.ndarray of the same shape (num_stocks, num_days).
                   Each value is the number of shares TRADED on that day.
                   +N = buy N shares
                   -N = sell / open short N shares
                    0 = hold (no trade)
    """
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)

    window = 20
    entry_z = -1.5   # buy when price is 1.5 std below mean
    exit_z = 0.0     # sell when z-score crosses back above 0
    shares = 10

    for i in range(num_stocks):
        position = 0
        for t in range(window, num_days):
            window_prices = prices[i, t - window : t]
            ma = window_prices.mean()
            std = window_prices.std()
            if std < 1e-12:
                std = 1e-12
            z = (prices[i, t] - ma) / std

            if z < entry_z and position == 0:
                actions[i, t] = shares
                position = 1
            elif z >= exit_z and position > 0:
                actions[i, t] = -shares
                position = 0

    return actions