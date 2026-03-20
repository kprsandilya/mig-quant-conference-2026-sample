import numpy as np

def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)

    window = 30
    entry_z = 1.5
    exit_z = 0.3

    capital_per_pair = 1000

    # -----------------------------
    # STEP 1: Find pairs (once)
    # -----------------------------
    returns = np.diff(prices) / prices[:, :-1]
    corr = np.corrcoef(returns)

    pairs = []
    used = set()

    for i in range(num_stocks):
        if i in used:
            continue

        # ignore self, find best partner
        corr_i = corr[i].copy()
        corr_i[i] = -1

        j = np.argmax(corr_i)

        if corr_i[j] > 0.6 and j not in used:
            pairs.append((i, j))
            used.add(i)
            used.add(j)
    # -----------------------------
    # STATE
    # -----------------------------
    positions = np.zeros((num_stocks,))
    pair_positions = {pair: 0 for pair in pairs}  # -1, 0, 1

    # -----------------------------
    # STEP 2: Trade spreads
    # -----------------------------
    for t in range(window, num_days):
        target_positions = np.zeros(num_stocks)

        for (i, j) in pairs:
            spread = np.log(prices[i, t-window:t]) - np.log(prices[j, t-window:t])
            current_spread = np.log(prices[i, t]) - np.log(prices[j, t])
            mean = spread.mean()
            std = spread.std()
            if std < 1e-8:
                continue

            current_spread = prices[i, t] - prices[j, t]
            z = (current_spread - mean) / std

            pos = pair_positions[(i, j)]

            # -------------------------
            # ENTRY
            # -------------------------
            if pos == 0:
                if z > entry_z:
                    # i expensive, j cheap → short i, long j
                    pair_positions[(i, j)] = -1
                elif z < -entry_z:
                    # i cheap, j expensive → long i, short j
                    pair_positions[(i, j)] = 1

            # -------------------------
            # EXIT
            # -------------------------
            elif abs(z) < exit_z:
                pair_positions[(i, j)] = 0

            # -------------------------
            # POSITION SIZING
            # -------------------------
            pos = pair_positions[(i, j)]
            if pos != 0:
                shares_i = int(capital_per_pair / prices[i, t])
                shares_j = int(capital_per_pair / prices[j, t])

                if pos == 1:
                    target_positions[i] += shares_i
                    target_positions[j] -= shares_j
                elif pos == -1:
                    target_positions[i] -= shares_i
                    target_positions[j] += shares_j

        # Convert to actions
        actions[:, t] = target_positions - positions
        positions = target_positions.copy()

    return actions