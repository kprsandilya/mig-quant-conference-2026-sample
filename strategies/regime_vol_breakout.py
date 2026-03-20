"""
Regime: low realized vol → buy 20d dip (z < -threshold); high vol → 5d momentum.
Single-stock exposure + two-day liquidate / enter cycle (backtester buys before sells).
"""

import numpy as np


def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)

    w = 20
    w_short = 5
    vol_hi = 0.017
    entry_z = 1.55
    mom_th = 0.008
    sh = 1  # one share: reliable fills, no phantom positions vs backtester

    held_i = -1

    for t in range(w, num_days):
        if t % 2 == 0:
            if held_i >= 0:
                actions[held_i, t] = -sh
                held_i = -1
            continue

        best_i = -1
        best_score = -np.inf

        for i in range(num_stocks):
            hist = prices[i, t - w : t]
            p0 = prices[i, t]
            ma = hist.mean()
            sd_p = hist.std()
            if sd_p < 1e-12:
                sd_p = 1e-12
            z = (p0 - ma) / sd_p
            rets = np.diff(hist) / (hist[:-1] + 1e-12)
            vol = float(rets.std()) if len(rets) > 1 else 0.0
            r5 = p0 / max(prices[i, t - w_short], 1e-12) - 1.0

            if vol <= vol_hi:
                if z < -entry_z:
                    score = -z
                else:
                    score = -np.inf
            else:
                if r5 > mom_th:
                    score = r5
                else:
                    score = -np.inf

            if score > best_score:
                best_score = score
                best_i = i

        if best_i < 0 or not np.isfinite(best_score):
            continue

        actions[best_i, t] = sh
        held_i = best_i

    return actions
