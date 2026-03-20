"""
Run all strategies in this folder against the backtester and compare results.
Usage (from repo root): python strategies/run_all.py
       or (from strategies/):  python run_all.py
"""

import os
import sys

import numpy as np
import pandas as pd

# Run from repo root so backtester and dev_data are found
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from backtester import Backtester


def load_prices(data_path: str) -> np.ndarray:
    """Load dev_data.csv and return prices matrix (num_stocks, num_days)."""
    df = pd.read_csv(
        data_path,
        parse_dates=["Date"],
        date_format="%Y-%m-%d",
        thousands=",",
    )
    prices_df = df.pivot(index="Ticker", columns="Date", values="Open").sort_index()
    prices_df = prices_df.ffill(axis=1).dropna(axis=1)
    return prices_df.values


def backtest_quiet(strategy_fn, prices: np.ndarray, cash: float = 25_000) -> dict | None:
    """Run strategy through backtester; return metrics dict or None if failed."""
    actions = strategy_fn(prices)
    bt = Backtester(prices, actions, cash=cash)
    # Suppress backtester prints by redirecting stdout temporarily
    import io
    import contextlib
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        port_values, pnl = bt.eval_actions()
    if port_values is None:
        return None
    pv = np.array(port_values)
    # Daily returns for Sharpe and drawdown
    daily_ret = np.diff(pv) / (pv[:-1] + 1e-12)
    daily_ret = daily_ret[~np.isnan(daily_ret)]
    sharpe = 0.0
    if len(daily_ret) > 0 and np.std(daily_ret) > 0:
        sharpe = np.mean(daily_ret) / np.std(daily_ret) * np.sqrt(252)
    running_max = np.maximum.accumulate(pv)
    drawdown = (running_max - pv) / (running_max + 1e-12)
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
    return {
        "pnl": pnl,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "final_value": float(pv[-1]),
    }


def main():
    data_path = os.path.join(REPO_ROOT, "dev_data_30.csv")
    if not os.path.isfile(data_path):
        print(f"Data not found: {data_path}")
        sys.exit(1)

    prices = load_prices(data_path)
    print(f"Loaded prices: {prices.shape[0]} stocks, {prices.shape[1]} days")
    print()

    # Import strategy modules (must be run with cwd = repo root)
    from strategies import (
        ma_crossover,
        ma_crossover_slow,
        mean_reversion,
        zscore_reversion,
        chat_vol,
        gemini_mean_reversion,
        mig_competition,
        chat_mean_reversion,
        cross_sectional,
        pairs_trading,
        graph
    )
    strategies = [
        ("MA(5/20)", ma_crossover.get_actions),
        ("MA(10/50)", ma_crossover_slow.get_actions),
        ("Mean reversion (20d)", mean_reversion.get_actions),
        ("Z-score reversion (20d)", zscore_reversion.get_actions),
        ("ChatGPT Volatility (20d)", chat_vol.get_actions),
        ("Gemini Mean Reversion (20d)", gemini_mean_reversion.get_actions),
        ("MIG Competition", mig_competition.get_actions),
        ("ChatGPT Mean Reversion", chat_mean_reversion.get_actions),
        ("Cross Sectional Volatity", cross_sectional.get_actions),
        ("Pairs Trading", pairs_trading.get_actions),
        ("Graph Trading", graph.get_actions)
    ]

    results = []
    for name, fn in strategies:
        print(f"Running: {name} ...")
        r = backtest_quiet(fn, prices)
        if r is None:
            print(f"  FAILED (e.g. portfolio went negative)")
            results.append({"Strategy": name, "PnL ($)": "FAILED", "Sharpe": "-", "Max DD (%)": "-", "Final value": "-"})
        else:
            results.append({
                "Strategy": name,
                "PnL ($)": f"{r['pnl']:,.2f}",
                "Sharpe": f"{r['sharpe']:.3f}",
                "Max DD (%)": f"{r['max_drawdown_pct']:.2f}",
                "Final value": f"{r['final_value']:,.2f}",
            })
        print()

    df = pd.DataFrame(results).set_index("Strategy")
    print("=" * 60)
    print("Strategy comparison (best by PnL)")
    print("=" * 60)
    print(df.to_string())
    print()

    # Rank by PnL (parse back for ordering if needed)
    pnls = []
    for r in results:
        pnl_str = r["PnL ($)"]
        if pnl_str == "FAILED":
            pnls.append((r["Strategy"], float("-inf")))
        else:
            pnls.append((r["Strategy"], float(pnl_str.replace(",", ""))))
    best = max(pnls, key=lambda x: x[1])
    print(f"Best strategy by PnL: {best[0]} (PnL = ${best[1]:,.2f})")


if __name__ == "__main__":
    main()
