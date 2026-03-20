import numpy as np

def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)

    # --- Parameters ---
    window = 30
    long_window = 50

    entry_z_long = -1.5
    entry_z_short = 1.5
    exit_z = 0.5

    stop_loss_pct = 0.05
    max_hold = 10

    target_risk = 0.02  # risk scaling

    # --- State tracking ---
    positions = np.zeros(num_stocks)      # -1, 0, 1
    entry_prices = np.zeros(num_stocks)
    days_held = np.zeros(num_stocks)

    for t in range(long_window, num_days):

        # Precompute rolling stats for all stocks
        window_prices = prices[:, t-window:t]
        ma = window_prices.mean(axis=1)
        std = window_prices.std(axis=1)
        std[std < 1e-8] = 1e-8

        long_ma = prices[:, t-long_window:t].mean(axis=1)

        z_scores = (prices[:, t] - ma) / std

        for i in range(num_stocks):
            price = prices[i, t]
            z = z_scores[i]

            # Volatility-scaled position size
            shares = max(10, int(target_risk / std[i]))

            # -------------------------
            # EXISTING POSITION
            # -------------------------
            if positions[i] != 0:
                days_held[i] += 1

                # Stop-loss
                if positions[i] == 1:
                    if price < entry_prices[i] * (1 - stop_loss_pct):
                        actions[i, t] = -shares
                        positions[i] = 0
                        continue
                elif positions[i] == -1:
                    if price > entry_prices[i] * (1 + stop_loss_pct):
                        actions[i, t] = shares
                        positions[i] = 0
                        continue

                # Exit on mean reversion
                if positions[i] == 1 and z >= exit_z:
                    actions[i, t] = -shares
                    positions[i] = 0
                    continue
                elif positions[i] == -1 and z <= -exit_z:
                    actions[i, t] = shares
                    positions[i] = 0
                    continue

                # Max holding period
                if days_held[i] >= max_hold:
                    if positions[i] == 1:
                        actions[i, t] = -shares
                    else:
                        actions[i, t] = shares
                    positions[i] = 0
                    continue

            # -------------------------
            # ENTRY CONDITIONS
            # -------------------------
            else:
                # Long entry (mean reversion + trend filter)
                if z < entry_z_long:
                    actions[i, t] = shares
                    positions[i] = 1
                    entry_prices[i] = price
                    days_held[i] = 0

                # Short entry
                elif z > entry_z_short:
                    actions[i, t] = -shares
                    positions[i] = -1
                    entry_prices[i] = price
                    days_held[i] = 0

    return actions