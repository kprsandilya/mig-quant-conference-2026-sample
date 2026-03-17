import numpy as np


def get_actions(prices: np.ndarray) -> np.ndarray:
    """
    Cross-sectional momentum strategy with volatility scaling.
    """

    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)

    # strategy parameters
    momentum_short = 20
    momentum_long = 60
    vol_window = 20
    rebalance = 5

    positions = np.zeros(num_stocks)

    for t in range(momentum_long, num_days):

        # rebalance periodically
        if t % rebalance != 0:
            continue

        # compute momentum
        short_ret = prices[:, t] / prices[:, t - momentum_short] - 1
        long_ret = prices[:, t] / prices[:, t - momentum_long] - 1
        momentum = short_ret - long_ret

        # volatility estimate
        returns = prices[:, t - vol_window + 1:t + 1] / prices[:, t - vol_window:t] - 1
        vol = returns.std(axis=1) + 1e-6

        # risk adjusted score
        score = momentum / vol

        # rank stocks
        ranks = np.argsort(score)

        n = num_stocks // 5  # top/bottom 20%

        longs = ranks[-n:]
        shorts = ranks[:n]

        target = np.zeros(num_stocks)

        # position sizing
        target[longs] = 5 / vol[longs]
        target[shorts] = -5 / vol[shorts]

        # convert to trade actions
        trade = target - positions
        actions[:, t] = trade.astype(int)

        positions = target

    return actions