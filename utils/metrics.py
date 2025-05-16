import numpy as np
import pandas as pd
import re

def get_periods_per_year(interval):
    m = re.match(r"(\d+)\s*(min|h|d)", interval, re.IGNORECASE)
    if not m:
        raise ValueError(f"Invalid interval format: {interval}")
    num, unit = m.groups()
    num = int(num)
    unit = unit.lower()
    if unit == "min":
        return 525600 / num
    if unit == "h":
        return 8760 / num
    if unit == "d":
        return 365 / num
    return 1

def calculate_metrics(df, interval, annualization_factor=None):
    """
    Compute performance metrics (no per-trade stats).
    Returns a dict suitable for JSON serialization.
    """
    # Ensure datetime column exists
    df = df.copy()
    if df.index.name == "datetime" or np.issubdtype(df.index.dtype, np.datetime64):
        df = df.reset_index()
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Basic returns
    returns = df.get("pnl", pd.Series(dtype=float))
    periods_per_year = get_periods_per_year(interval)
    if annualization_factor is None:
        annualization_factor = np.sqrt(periods_per_year)

    # Sharpe Ratio
    pnl_std = returns.std() if len(returns) else 0
    mean_return = returns.mean() if len(returns) else 0
    sharpe = (mean_return / pnl_std * annualization_factor) if pnl_std else 0

    # Cumulative and drawdowns
    cumu = returns.cumsum()
    running_max = cumu.cummax()
    drawdowns = cumu - running_max
    max_dd = drawdowns.min() if len(drawdowns) else 0

    # Drawdown dates
    if len(drawdowns):
        dd_end_idx = drawdowns.idxmin()
        dd_end_loc = drawdowns.index.get_loc(dd_end_idx)
        dd_end = df.at[dd_end_loc, "datetime"]
    else:
        dd_end = df.at[0, "datetime"] if len(df) else None
        dd_end_loc = 0

    if dd_end_loc > 0:
        pre = running_max.iloc[:dd_end_loc] == cumu.iloc[:dd_end_loc]
        if pre.any():
            start_label = pre[pre].index[-1]
            start_loc = drawdowns.index.get_loc(start_label)
            dd_start = df.at[start_loc, "datetime"]
        else:
            dd_start = df.at[0, "datetime"]
    else:
        dd_start = dd_end

    post = cumu.iloc[dd_end_loc:]
    recover = post[post >= running_max.iloc[dd_end_loc]]
    if len(recover):
        rec_loc = drawdowns.index.get_loc(recover.index[0])
        dd_recover = df.at[rec_loc, "datetime"]
    else:
        dd_recover = dd_end

    mdd_days = (dd_recover - dd_start).total_seconds() / 86400 if dd_start and dd_recover else 0

    # Sortino Ratio
    downside = returns[returns < 0]
    ds_std = downside.std() if len(downside) else 0
    sortino = (mean_return / ds_std * annualization_factor) if ds_std else 0
    calmar = (mean_return * periods_per_year / abs(max_dd)) if max_dd else 0
    ann_return = mean_return * periods_per_year
    num_trades = int(df.get("trade", pd.Series()).sum())
    total_ret = float(returns.sum())
    tpi = num_trades / len(df) if len(df) else 0
    sr_cr = sharpe / calmar if calmar else 0

    return {
        "SR": round(sharpe, 4),
        "CR": round(calmar, 4),
        "MDD": round(max_dd, 4),
        "sortino_ratio": round(sortino, 4),
        "AR": round(ann_return, 4),
        "num_of_trades": float(num_trades),
        "TR": round(total_ret, 4),
        "trades_per_interval": round(tpi, 4),
        "MDD_MAX_DURATION_IN_DAY": round(mdd_days, 4),
        "SR_CR": round(sr_cr, 4),
        "backtest_start_date": df["datetime"].min().isoformat() if "datetime" in df.columns else None,
        "backtest_end_date": df["datetime"].max().isoformat() if "datetime" in df.columns else None,
    }
