import numpy as np
import pandas as pd
import time

def calculate_pnl(df, fee=0.0006):
    df["returns"] = df["close"].pct_change().fillna(0)
    df["pnl"] = df["returns"] * df["position"].shift(1).fillna(0) - fee * df["trade"]
    df["cumpnl"] = df['pnl'].cumsum()
    return df

def calculate_positions(df, logic_fn, long_threshold, short_threshold):
    signals = df['signal'].to_numpy()
    n = len(signals)
    positions = np.zeros(n, dtype=int)
    trades    = np.zeros(n, dtype=int)
    t0 = time.time()

    for i in range(1, n):
        prev = positions[i-1]
        newp = logic_fn(signals[i], long_threshold, short_threshold, prev)
        positions[i] = newp
        trades[i]    = abs(newp - prev)
    elapsed = time.time() - t0
    it_s    = n / elapsed if elapsed > 0 else float('inf')

    print(f"[Backtest] Completed {n} rows in {elapsed:.8f}s — {it_s:,.0f} it/s")

    df['position'] = positions
    df['trade']    = trades
    return df
