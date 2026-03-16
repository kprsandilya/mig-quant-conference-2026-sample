import numpy as np


def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)
    window = 20

    for i in range(num_stocks):
        position = 0
        for t in range(window, num_days):
            ma = prices[i, t - window : t].mean()
            price = prices[i, t]

            if price < ma * 0.98 and position == 0:
                actions[i, t] = 100
                position = 1
            elif price > ma * 1.02 and position > 0:
                actions[i, t] = -100
                position = 0

    return actions
