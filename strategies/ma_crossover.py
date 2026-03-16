"""
MA(5/20) moving-average crossover strategy.
Buy 1 share when 5-day MA crosses above 20-day MA; sell when it crosses below.
"""

import numpy as np


def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)
    short_window, long_window = 5, 20

    for i in range(num_stocks):
        position = 0
        for t in range(long_window, num_days):
            short_ma = prices[i, t - short_window : t].mean()
            long_ma = prices[i, t - long_window : t].mean()

            if short_ma > long_ma and position == 0:
                actions[i, t] = 1
                position = 1
            elif short_ma <= long_ma and position == 1:
                actions[i, t] = -1
                position = 0

    return actions
