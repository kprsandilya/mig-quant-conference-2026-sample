import numpy as np

def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)

    window = 20
    k_eigs = 5
    entry_z = 1.0

    total_capital = 25000  # FIXED capital budget

    positions = np.zeros(num_stocks)

    # -----------------------------
    # Build graph once
    # -----------------------------
    returns = np.diff(prices) / prices[:, :-1]
    corr = np.corrcoef(returns)

    A = np.maximum(corr, 0)
    np.fill_diagonal(A, 0)

    D = np.diag(A.sum(axis=1))
    L = D - A

    eigvals, eigvecs = np.linalg.eigh(L)
    U = eigvecs[:, 1:k_eigs+1]

    # -----------------------------
    # Trading loop
    # -----------------------------
    for t in range(window, num_days):
        p = np.log(prices[:, t])

        # Graph smoothing
        smooth = U @ (U.T @ p)
        residual = p - smooth

        std = residual.std()
        if std < 1e-8:
            continue

        z = residual / std

        # -------------------------
        # SIGNAL → WEIGHTS
        # -------------------------
        signal = np.zeros(num_stocks)

        # only trade strongest signals (reduces noise + risk)
        threshold = np.percentile(np.abs(z), 80)

        for i in range(num_stocks):
            if abs(z[i]) > threshold:
                signal[i] = -z[i]  # mean reversion

        # -------------------------
        # NORMALIZATION (CRITICAL)
        # -------------------------
        if np.sum(np.abs(signal)) < 1e-8:
            continue

        # dollar neutral
        signal -= signal.mean()

        weights = signal / (np.sum(np.abs(signal)) + 1e-8)

        # scale to capital
        dollar_position = weights * total_capital

        # convert to shares
        target = dollar_position / prices[:, t]

        # -------------------------
        # ACTIONS
        # -------------------------
        actions[:, t] = target - positions
        positions = target.copy()

    return actions