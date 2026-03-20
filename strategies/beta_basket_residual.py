"""
Co-linear / basket residual: regress each stock's log Open on the equal-weight
log basket over a rolling window; mean-revert the standardized residual (stat-arb style).
"""

import numpy as np


def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)

    window = 50
    entry_z = 1.35
    exit_z = 0.4
    dollar = 500
    max_sh = 90

    logp = np.log(np.maximum(prices, 1e-8))
    basket = logp.mean(axis=0)

    positions = np.zeros(num_stocks)

    for t in range(window, num_days):
        target = np.zeros(num_stocks)
        xw = basket[t - window : t + 1]

        for i in range(num_stocks):
            yw = logp[i, t - window : t + 1]
            xm, ym = xw.mean(), yw.mean()
            vx = np.var(xw) + 1e-12
            cov = np.mean((xw - xm) * (yw - ym))
            beta = cov / vx
            alpha = ym - beta * xm
            e_hist = yw - (alpha + beta * xw)
            e_now = e_hist[-1]
            sd = e_hist.std()
            if sd < 1e-8:
                sd = 1e-8
            z = (e_now - e_hist.mean()) / sd

            sh = min(max_sh, max(1, int(dollar / (prices[i, t] + 1e-12))))
            pos = positions[i]

            if pos == 0:
                if z < -entry_z:
                    target[i] = sh
                elif z > entry_z:
                    target[i] = -sh
                else:
                    target[i] = 0
            elif pos > 0:
                if z > -exit_z:
                    target[i] = 0
                else:
                    target[i] = pos  # hold size
            else:
                if z < exit_z:
                    target[i] = 0
                else:
                    target[i] = pos  # hold short size

        actions[:, t] = target - positions
        positions = target.copy()

    return actions
