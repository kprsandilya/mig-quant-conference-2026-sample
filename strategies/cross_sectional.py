import numpy as np

def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)

    window = 20
    top_k = max(1, num_stocks // 10)  # top/bottom 10%
    capital_per_stock = 1000  # controls aggressiveness

    positions = np.zeros(num_stocks)

    for t in range(window, num_days):
        window_prices = prices[:, t-window:t]
        ma = window_prices.mean(axis=1)
        std = window_prices.std(axis=1)
        std[std < 1e-8] = 1e-8

        z = (prices[:, t] - ma) / std

        # Rank stocks
        sorted_idx = np.argsort(z)

        long_idx = sorted_idx[:top_k]     # most negative z
        short_idx = sorted_idx[-top_k:]   # most positive z

        target_positions = np.zeros(num_stocks)

        # Allocate capital evenly
        for i in long_idx:
            shares = int(capital_per_stock / prices[i, t])
            target_positions[i] = shares

        for i in short_idx:
            shares = int(capital_per_stock / prices[i, t])
            target_positions[i] = -shares

        # Convert target positions → actions
        actions[:, t] = target_positions - positions
        positions = target_positions.copy()

    return actions