"""
Graph trading (refined): static Laplacian eigenvectors + residual z-score.

Single-name trading with conservative sizing and both long/short.
Includes take-profit + stop-loss to reduce drawdowns and avoid negative portfolio values.
"""

import numpy as np


def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)

    eps = 1e-8
    logp = np.log(np.maximum(prices, eps))

    # Static graph (uses full series correlation)
    k_eigs = 6
    returns = np.diff(prices, axis=1) / np.maximum(prices[:, :-1], eps)
    corr = np.corrcoef(returns)
    A = np.maximum(corr, 0.0)
    np.fill_diagonal(A, 0.0)

    D = np.diag(A.sum(axis=1))
    L = D - A
    _, eigvecs = np.linalg.eigh(L)

    start_col = 1
    stop_col = min(num_stocks, start_col + k_eigs)
    U = eigvecs[:, start_col:stop_col]

    # Signal settings
    window = 20
    entry_abs_z = 1.05
    exit_abs_z = 0.55
    top_percentile = 82

    # Conservative single-position sizing
    dollar_target = 5_000.0
    max_sh = 220
    max_hold_days = 25
    stop_abs_z = 2.2

    held_i = -1
    held_qty = 0
    held_dir = 0
    entry_day = -1

    for t in range(window, num_days):
        if U is None or U.shape[1] < 1:
            continue

        p = logp[:, t]
        smooth = U @ (U.T @ p)
        residual = p - smooth
        std = residual.std()
        if std < 1e-8:
            continue

        z = residual / (std + 1e-12)
        absz = np.abs(z)
        extreme_thresh = np.percentile(absz, top_percentile)

        if held_i >= 0:
            if held_dir > 0:
                exit_tp = z[held_i] > -exit_abs_z
                exit_sl = z[held_i] <= -stop_abs_z
                if exit_tp or exit_sl or (t - entry_day) >= max_hold_days:
                    actions[held_i, t] = -held_qty
                    held_i = -1
                    held_qty = 0
                    held_dir = 0
                    entry_day = -1
            else:
                exit_tp = z[held_i] < exit_abs_z
                exit_sl = z[held_i] >= stop_abs_z
                if exit_tp or exit_sl or (t - entry_day) >= max_hold_days:
                    actions[held_i, t] = held_qty
                    held_i = -1
                    held_qty = 0
                    held_dir = 0
                    entry_day = -1
            continue

        entry_mask = (absz > extreme_thresh) & (absz > entry_abs_z)
        if not np.any(entry_mask):
            continue

        cand = np.where(entry_mask)[0]
        best_i = cand[np.argmax(absz[cand])]

        px = float(prices[best_i, t])
        qty = int(dollar_target / (px + 1e-12))
        qty = max(1, min(max_sh, qty))

        if z[best_i] < 0:
            actions[best_i, t] = qty
            held_dir = 1
        else:
            actions[best_i, t] = -qty
            held_dir = -1

        held_i = best_i
        held_qty = qty
        entry_day = t

    return actions

