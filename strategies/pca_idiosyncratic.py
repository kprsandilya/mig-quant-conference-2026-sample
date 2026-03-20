"""
PCA / factor residual: estimate first principal component of return covariance over a window;
strip systematic component from each name and mean-revert cross-sectional idiosyncratic z-scores.
"""

import numpy as np


def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)

    window = 40
    top_k = max(2, num_stocks // 5)
    entry_z = 1.1
    exit_z = 0.35
    dollar = 450
    max_sh = 85

    logp = np.log(np.maximum(prices, 1e-8))
    rets = np.diff(logp, axis=1)

    positions = np.zeros(num_stocks)

    for t in range(window, num_days):
        # Returns from rets[:, t-window : t] → price moves into day t
        R = rets[:, t - window : t]
        if R.shape[1] < 5:
            continue
        Rc = R - R.mean(axis=1, keepdims=True)
        C = np.cov(Rc)
        eigvals, eigvecs = np.linalg.eigh(C)
        v = eigvecs[:, -1]
        v = v / (np.linalg.norm(v) + 1e-12)

        r_last = R[:, -1]
        f = float(np.dot(v, r_last))
        idio = r_last - v * f

        z = (idio - idio.mean()) / (idio.std() + 1e-12)

        sorted_idx = np.argsort(z)
        long_set = set(sorted_idx[:top_k])
        short_set = set(sorted_idx[-top_k:])

        target = np.zeros(num_stocks)
        for i in range(num_stocks):
            sh = min(max_sh, max(1, int(dollar / (prices[i, t] + 1e-12))))
            pos = positions[i]
            zi = z[i]

            if pos == 0:
                if i in long_set and zi < -entry_z:
                    target[i] = sh
                elif i in short_set and zi > entry_z:
                    target[i] = -sh
            elif pos > 0:
                if zi > -exit_z or i not in long_set:
                    target[i] = 0
                else:
                    target[i] = pos
            else:
                if zi < exit_z or i not in short_set:
                    target[i] = 0
                else:
                    target[i] = pos

        actions[:, t] = target - positions
        positions = target.copy()

    return actions
