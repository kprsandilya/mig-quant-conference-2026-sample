"""
Machine learning: logistic regression on pooled past-only features to predict next-day up move.
Train once on the first half of the timeline (no peeking at second half); trade the second half.
First half uses a simple long-only momentum overlay so the backtest is not idle.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _features_at(prices: np.ndarray, i: int, t: int, w: int = 20) -> np.ndarray | None:
    if t < w + 5:
        return None
    p = prices[i, :]
    p0 = p[t]
    if p0 < 1e-12:
        return None
    r1 = p[t] / p[t - 1] - 1.0
    r5 = p[t] / p[t - 5] - 1.0
    r20 = p[t] / p[t - 20] - 1.0
    hist = p[t - w : t]
    ma = hist.mean()
    std = hist.std()
    if std < 1e-12:
        std = 1e-12
    z = (p0 - ma) / std
    rets = np.diff(hist) / (hist[:-1] + 1e-12)
    vol = float(rets.std()) if len(rets) > 1 else 0.0
    return np.array([r1, r5, r20, z, vol], dtype=np.float64)


def get_actions(prices: np.ndarray) -> np.ndarray:
    num_stocks, num_days = prices.shape
    actions = np.zeros_like(prices)

    w = 20
    min_t = w + 5
    train_end = max(min_t + 50, num_days // 2)

    # --- Build training set (only days where t+1 exists and t < train_end) ---
    X_list, y_list = [], []
    for t in range(min_t, min(train_end, num_days - 1)):
        for i in range(num_stocks):
            feat = _features_at(prices, i, t, w)
            if feat is None:
                continue
            y = 1 if prices[i, t + 1] > prices[i, t] else 0
            X_list.append(feat)
            y_list.append(y)

    model = None
    if len(X_list) >= 200:
        X = np.vstack(X_list)
        y = np.array(y_list, dtype=np.int32)
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=300,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(X, y)

    dollar = 400
    max_sh = 80
    prev_target = np.zeros(num_stocks)

    for t in range(min_t, num_days):
        target = np.zeros(num_stocks)
        for i in range(num_stocks):
            feat = _features_at(prices, i, t, w)
            if feat is None:
                continue
            p = prices[i, t]
            sh = min(max_sh, max(1, int(dollar / (p + 1e-12))))

            if t < train_end or model is None:
                # First half / no model: mild long-only momentum
                if feat[2] > 0.02:  # r20 > 2%
                    target[i] = sh
            else:
                proba = float(model.predict_proba(feat.reshape(1, -1))[0, 1])
                if proba > 0.56:
                    target[i] = sh
                elif proba < 0.44:
                    target[i] = -sh

        actions[:, t] = target - prev_target
        prev_target = target.copy()

    return actions
