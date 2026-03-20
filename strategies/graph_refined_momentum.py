"""
Graph trading (refined): mean-reversion residual z-score + momentum disagreement filter.

Single-name trading with conservative sizing and both long/short,
with take-profit + stop-loss to prevent negative portfolio values.
"""

import numpy as np


def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)

    eps = 1e-8
    logp = np.log(np.maximum(prices, eps))

    # Static graph build
    k_eigs = 7
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
    exit_abs_z = 0.60
    top_percentile = 83

    # Momentum filter (avoid entering longs when momentum is strongly positive)
    mom_w = 5
    mom_avoid_long = 0.020

    dollar_target = 5_000.0
    max_sh = 220
    max_hold_days = 25
    stop_abs_z = 2.2

    held_i = -1
    held_qty = 0
    held_dir = 0
    entry_day = -1

    for t in range(window + mom_w, num_days):
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
                    actions[held_i, t] = held_qty  # cover short
                    held_i = -1
                    held_qty = 0
                    held_dir = 0
                    entry_day = -1
            continue

        # Candidate names (extreme residuals). Direction depends on sign(z).
        entry_mask = (absz > extreme_thresh) & (absz > entry_abs_z)
        if not np.any(entry_mask):
            continue

        mom = prices[:, t] / np.maximum(prices[:, t - mom_w], eps) - 1.0

        cand = np.where(entry_mask)[0]
        # Momentum disagreement filter:
        # - for long candidates (z < 0), avoid when momentum is strongly positive
        # - for short candidates (z > 0), avoid when momentum is strongly negative
        long_cand = cand[z[cand] < 0]
        short_cand = cand[z[cand] > 0]
        if len(long_cand) > 0:
            long_cand = long_cand[mom[long_cand] <= mom_avoid_long]
        if len(short_cand) > 0:
            short_cand = short_cand[mom[short_cand] >= -mom_avoid_long]
        if len(long_cand) == 0 and len(short_cand) == 0:
            continue

        # Choose best by abs(z)
        if len(long_cand) > 0 and len(short_cand) > 0:
            best_long = long_cand[np.argmax(absz[long_cand])]
            best_short = short_cand[np.argmax(absz[short_cand])]
            if absz[best_long] >= absz[best_short]:
                best_i = best_long
            else:
                best_i = best_short
        elif len(long_cand) > 0:
            best_i = long_cand[np.argmax(absz[long_cand])]
        else:
            best_i = short_cand[np.argmax(absz[short_cand])]

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

