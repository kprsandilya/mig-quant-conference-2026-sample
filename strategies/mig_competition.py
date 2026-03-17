"""
MIG Quant Competition — Strategy for submission.
Uses only Open prices. Optimized for: 1st PnL, 2nd Sharpe, 3rd lower Max Drawdown.
No ta-lib dependency so it runs locally and in sandbox; can be extended with ta-lib if desired.
Ref: https://github.com/AryamanGoenka0910/mig-quant-conference-2026-sample
"""

import numpy as np


def get_actions(prices: np.ndarray) -> np.ndarray:
    """
    Arguments:
        prices  -- np.ndarray of shape (num_stocks, num_days)
                   Open price for each stock on each day.
                   Rows = stocks (sorted by ticker), Columns = days.

    Returns:
        actions -- np.ndarray of the same shape (num_stocks, num_days).
                   +N = buy N shares, -N = sell/short N shares, 0 = hold.
    """
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)

    # Parameters tuned for PnL / Sharpe / drawdown
    window = 20
    trend_window = 50
    entry_z = -1.6       # oversold entry (stricter = fewer, higher-quality)
    exit_z = -0.2        # take profit before full reversion
    exit_z_stop = -2.6   # stop-loss if more oversold
    max_hold_days = 18
    trend_min_ratio = 0.91  # only long when price >= trend_ma * this (avoid falling knives)

    dollar_target = 450   # target $ per position
    max_shares = 90       # cap concentration

    for i in range(num_stocks):
        position = 0
        entry_day = -1
        for t in range(max(window, trend_window), num_days):
            window_prices = prices[i, t - window : t]
            ma = window_prices.mean()
            std = window_prices.std()
            if std < 1e-12:
                std = 1e-12
            z = (prices[i, t] - ma) / std

            trend_prices = prices[i, t - trend_window : t]
            trend_ma = trend_prices.mean()
            above_trend = prices[i, t] >= trend_ma * trend_min_ratio

            shares = min(max_shares, max(1, int(dollar_target / (prices[i, t] + 1e-12))))

            if position > 0:
                days_held = t - entry_day
                if z >= exit_z:
                    actions[i, t] = -position
                    position = 0
                elif z <= exit_z_stop or days_held >= max_hold_days:
                    actions[i, t] = -position
                    position = 0

            if position == 0 and z < entry_z and above_trend:
                actions[i, t] = shares
                position = shares
                entry_day = t

    return actions
