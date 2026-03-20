"""
Cross-sectional momentum: hold the single strongest 20d momentum name at a time.
Two-day cycle (liquidate → buy) so sells are not blocked by the backtester's
buy-before-sell order within a day.
"""

import numpy as np


def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)

    lookback = 20
    # Single share so buys virtually always fill; avoids state/backtester mismatch → accidental shorts.
    sh = 1

    held_i = -1

    for t in range(lookback, num_days):
        if t % 2 == 0:
            # Liquidation: only sells (no buys this day)
            if held_i >= 0:
                actions[held_i, t] = -sh
                held_i = -1
            continue

        mom = prices[:, t] / np.maximum(prices[:, t - lookback], 1e-12) - 1.0
        i = int(np.argmax(mom))
        actions[i, t] = sh
        held_i = i

    return actions
