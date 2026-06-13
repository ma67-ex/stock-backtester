"""
Moving-Average Crossover Backtester
-----------------------------------
A simple, honest backtest of a long/flat SMA-crossover strategy.

Strategy: go long when the short moving average is above the long
moving average, otherwise stay in cash. We compare it against a plain
buy-and-hold of the same stock.

Usage:
    python backtester.py --ticker AAPL --start 2018-01-01 --short 50 --long 200
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


TRADING_DAYS = 252  # ~ number of trading days in a year, used to annualize


def load_prices(ticker: str, start: str, end: str | None) -> pd.DataFrame:
    """Download daily prices and keep the adjusted close."""
    data = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if data.empty:
        raise SystemExit(f"No data returned for '{ticker}'. Check the symbol or dates.")
    # yfinance can return multi-index columns for a single ticker; flatten it.
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.to_frame("price")


def run_backtest(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    """Add moving averages, positions, and daily returns to the frame."""
    df = df.copy()
    df["sma_short"] = df["price"].rolling(short_window).mean()
    df["sma_long"] = df["price"].rolling(long_window).mean()

    # Signal: 1 when short MA is above long MA, else 0 (long or flat).
    df["signal"] = (df["sma_short"] > df["sma_long"]).astype(int)

    # Trade on the NEXT day's open-to-close, so we shift the signal by one
    # day. This avoids look-ahead bias (acting on info we couldn't have had).
    df["position"] = df["signal"].shift(1).fillna(0)

    df["market_return"] = df["price"].pct_change().fillna(0)
    df["strategy_return"] = df["position"] * df["market_return"]

    # Growth of $1 invested in each approach.
    df["buy_hold_equity"] = (1 + df["market_return"]).cumprod()
    df["strategy_equity"] = (1 + df["strategy_return"]).cumprod()
    return df


def performance(returns: pd.Series, equity: pd.Series) -> dict:
    """Compute headline stats for a return stream."""
    total_return = equity.iloc[-1] - 1
    years = len(returns) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan

    # Annualized Sharpe (risk-free rate assumed 0 for simplicity).
    std = returns.std()
    sharpe = (returns.mean() / std) * np.sqrt(TRADING_DAYS) if std > 0 else np.nan

    # Max drawdown: worst peak-to-trough drop on the equity curve.
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_drawdown = drawdown.min()

    return {
        "Total return": f"{total_return:6.1%}",
        "CAGR":         f"{cagr:6.1%}",
        "Sharpe":       f"{sharpe:6.2f}",
        "Max drawdown": f"{max_drawdown:6.1%}",
    }


def report(df: pd.DataFrame, ticker: str, short_window: int, long_window: int) -> None:
    """Print a side-by-side comparison table to the terminal."""
    strat = performance(df["strategy_return"], df["strategy_equity"])
    hold = performance(df["market_return"], df["buy_hold_equity"])

    print(f"\n{ticker}  |  SMA {short_window}/{long_window}  |  "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    print("-" * 48)
    print(f"{'Metric':<16}{'Strategy':>14}{'Buy & Hold':>14}")
    print("-" * 48)
    for key in strat:
        print(f"{key:<16}{strat[key]:>14}{hold[key]:>14}")
    print("-" * 48)


def plot(df: pd.DataFrame, ticker: str, short_window: int, long_window: int,
         outfile: str) -> None:
    """Save a two-panel chart: price+signals on top, equity curves below."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(df.index, df["price"], label="Price", linewidth=1, color="#333")
    ax1.plot(df.index, df["sma_short"], label=f"SMA {short_window}", linewidth=1)
    ax1.plot(df.index, df["sma_long"], label=f"SMA {long_window}", linewidth=1)

    # Mark where we enter (buy) and exit (sell).
    trades = df["position"].diff()
    buys = df[trades == 1]
    sells = df[trades == -1]
    ax1.scatter(buys.index, buys["price"], marker="^", color="green",
                s=60, label="Buy", zorder=5)
    ax1.scatter(sells.index, sells["price"], marker="v", color="red",
                s=60, label="Sell", zorder=5)

    ax1.set_title(f"{ticker} — SMA {short_window}/{long_window} Crossover Strategy")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    ax2.plot(df.index, df["strategy_equity"], label="Strategy", linewidth=1.5)
    ax2.plot(df.index, df["buy_hold_equity"], label="Buy & Hold",
             linewidth=1.5, linestyle="--")
    ax2.set_ylabel("Growth of $1")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    print(f"\nChart saved to {outfile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SMA crossover backtester")
    parser.add_argument("--ticker", default="AAPL", help="Stock symbol (e.g. AAPL)")
    parser.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--short", type=int, default=50, help="Short SMA window")
    parser.add_argument("--long", type=int, default=200, help="Long SMA window")
    parser.add_argument("--out", default="backtest.png", help="Output chart filename")
    args = parser.parse_args()

    if args.short >= args.long:
        raise SystemExit("--short window must be smaller than --long window.")

    prices = load_prices(args.ticker, args.start, args.end)
    df = run_backtest(prices, args.short, args.long)
    report(df, args.ticker, args.short, args.long)
    plot(df, args.ticker, args.short, args.long, args.out)


if __name__ == "__main__":
    main()
