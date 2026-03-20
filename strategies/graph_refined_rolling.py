"""
Graph trading (refined): rolling correlation Laplacian eigenvectors + residual z-score.

The backtester in this repo does not credit short-sale proceeds to cash and can
drive the portfolio negative for large/uncontrolled shorts.
This version trades in a *single-name* manner with conservative sizing and
hard exits (take-profit + stop-loss), supporting both long and short.
"""

import numpy as np


def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)

    eps = 1e-8
    logp = np.log(np.maximum(prices, eps))

    # Rolling graph / factor residual settings
    corr_window = 80
    rebuild_every = 25
    k_eigs = 6

    # Signal settings
    entry_abs_z = 1.15
    exit_abs_z = 0.65
    top_percentile = 80  # only trade the most extreme residuals

    # Conservative long sizing (single position → avoids multi-stock cash-order issues)
    dollar_target = 5_000.0
    max_sh = 220
    max_hold_days = 25
    stop_abs_z = 2.2  # stop when z-score moves further away

    held_i = -1
    held_qty = 0  # always positive share count in state
    held_dir = 0   # +1 long, -1 short
    entry_day = -1
    U = None

    # Precompute returns for rolling correlation
    rets = np.diff(prices, axis=1) / np.maximum(prices[:, :-1], eps)  # (n, T-1)

    for t in range(corr_window, num_days):
        if t % rebuild_every == 0 or U is None:
            start = t - corr_window
            end = t
            if end - start < 30:
                continue

            corr = np.corrcoef(rets[:, start:end])
            A = np.maximum(corr, 0.0)
            np.fill_diagonal(A, 0.0)

            D = np.diag(A.sum(axis=1))
            L = D - A
            _, eigvecs = np.linalg.eigh(L)

            start_col = 1
            stop_col = min(num_stocks, start_col + k_eigs)
            U = eigvecs[:, start_col:stop_col]

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
                # Long: z is negative; take profit as it mean-reverts toward 0+
                # Stop-loss if z becomes too negative (position continues to move away).
                exit_tp = z[held_i] > -exit_abs_z
                exit_sl = z[held_i] <= -stop_abs_z
                if exit_tp or exit_sl or (t - entry_day) >= max_hold_days:
                    actions[held_i, t] = -held_qty  # sell/close long
                    held_i = -1
                    held_qty = 0
                    held_dir = 0
                    entry_day = -1
            else:
                # Short: z is positive; take profit when it mean-reverts back down.
                exit_tp = z[held_i] < exit_abs_z
                exit_sl = z[held_i] >= stop_abs_z
                if exit_tp or exit_sl or (t - entry_day) >= max_hold_days:
                    actions[held_i, t] = held_qty  # cover short
                    held_i = -1
                    held_qty = 0
                    held_dir = 0
                    entry_day = -1
            continue

        # Entry: choose the single most extreme z among the percentile tail
        entry_mask = (absz > extreme_thresh) & (absz > entry_abs_z)
        if not np.any(entry_mask):
            continue

        cand = np.where(entry_mask)[0]
        best_i = cand[np.argmax(absz[cand])]

        px = float(prices[best_i, t])
        qty = int(dollar_target / (px + 1e-12))
        qty = max(1, min(max_sh, qty))

        if z[best_i] < 0:
            # Oversold → long
            actions[best_i, t] = qty
            held_dir = 1
        else:
            # Overvalued → short
            actions[best_i, t] = -qty
            held_dir = -1

        held_i = best_i
        held_qty = qty
        entry_day = t

    return actions

