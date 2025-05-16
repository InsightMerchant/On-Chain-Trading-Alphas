# utils/transformation.py

import numpy as np
import pandas as pd
from scipy.stats import boxcox, yeojohnson, mstats
from sklearn.preprocessing import QuantileTransformer
from typing import Optional, Callable

def with_rolling(fn: Callable[..., pd.Series]) -> Callable[..., pd.Series]:
    def wrapper(series: pd.Series, window: Optional[int] = None, *args, **kwargs) -> pd.Series:
        if window is None:
            return fn(series, *args, **kwargs)
        return series.rolling(window).apply(
            lambda arr: fn(pd.Series(arr), *args, **kwargs).iloc[-1],
            raw=False
        )
    return wrapper


# === 1) Stateless (pointwise) transforms – no look‐ahead risk ===

@with_rolling
def direct(series: pd.Series) -> pd.Series:
    """Identity transform."""
    return series

@with_rolling
def diff(series: pd.Series) -> pd.Series:
    """First difference."""
    return series.diff()

@with_rolling
def pct_change(series: pd.Series) -> pd.Series:
    """Percentage change."""
    return series.pct_change()

@with_rolling
def log(series: pd.Series) -> pd.Series:
    """Natural logarithm."""
    return np.log(series)

@with_rolling
def log1p(series: pd.Series) -> pd.Series:
    """Log(1 + x)."""
    return np.log1p(series)

@with_rolling
def log10(series: pd.Series) -> pd.Series:
    """Base-10 logarithm."""
    return np.log10(series)

@with_rolling
def sqrt(series: pd.Series) -> pd.Series:
    """Square root."""
    return np.sqrt(series)

@with_rolling
def cbrt(series: pd.Series) -> pd.Series:
    """Cube root."""
    return np.cbrt(series)

@with_rolling
def square(series: pd.Series) -> pd.Series:
    """Square."""
    return series ** 2

@with_rolling
def cube(series: pd.Series) -> pd.Series:
    """Cube."""
    return series ** 3

@with_rolling
def cube_root(series: pd.Series) -> pd.Series:
    return np.sign(series) * np.cbrt(np.abs(series))

@with_rolling
def sign_log(series: pd.Series) -> pd.Series:
    """Signed log1p: preserves sign, compresses magnitude."""
    return np.sign(series) * np.log1p(np.abs(series))

@with_rolling
def inverse_sinh(series: pd.Series) -> pd.Series:
    """Inverse hyperbolic sine (arcsinh)."""
    return np.arcsinh(series)

@with_rolling
def modulus(series: pd.Series, lambda_value: float = 0.5) -> pd.Series:
    """Signed power transform (modulus)."""
    if abs(lambda_value) < 1e-6:
        return np.sign(series) * np.log1p(np.abs(series))
    return np.sign(series) * ((np.abs(series) + 1) ** lambda_value - 1) / lambda_value

@with_rolling
def reciprocal(series: pd.Series) -> pd.Series:
    """Reciprocal with tiny offset to avoid div by zero."""
    return 1 / (series + 1e-9)

@with_rolling
def abs_val(series: pd.Series) -> pd.Series:
    """Absolute value."""
    return series.abs()

@with_rolling
def tanh(series: pd.Series) -> pd.Series:
    """Hyperbolic tangent."""
    return np.tanh(series)

@with_rolling
def arctan(series: pd.Series) -> pd.Series:
    """Arctangent."""
    return np.arctan(series)

@with_rolling
def sigmoid(series: pd.Series) -> pd.Series:
    """Logistic sigmoid."""
    return 1 / (1 + np.exp(-series))

@with_rolling
def softplus(series: pd.Series) -> pd.Series:
    """Softplus: log(1 + exp(x))."""
    return np.log1p(np.exp(series))

@with_rolling
def cumsum(series: pd.Series) -> pd.Series:
    """Cumulative sum."""
    return series.cumsum()

@with_rolling
def ema(series: pd.Series, span: int = 10) -> pd.Series:
    """Exponential moving average with fixed span."""
    return series.ewm(span=span, adjust=False).mean()

@with_rolling
def ema_diff(series: pd.Series, span: int = 10) -> pd.Series:
    """Difference between series and its EMA."""
    return series - series.ewm(span=span, adjust=False).mean()

@with_rolling
def sin_enc(series: pd.Series, period: int) -> pd.Series:
    """Sine encoding for cyclic features."""
    return np.sin(2 * np.pi * series / period)

@with_rolling
def cos_enc(series: pd.Series, period: int) -> pd.Series:
    """Cosine encoding for cyclic features."""
    return np.cos(2 * np.pi * series / period)

@with_rolling
def yeo_johnson_transform(series: pd.Series,lambda_value: float = 0.5) -> pd.Series:
    arr = series.values.astype(float).reshape(-1, 1)
    result = yeojohnson(arr, lmbda=lambda_value)
    if isinstance(result, tuple):
        transformed = result[0]
    else:
        transformed = result
    return pd.Series(transformed.flatten(), index=series.index)


# === 2) Parameterized transforms – MUST be fitted rolling or pre-fit on past only ===

@with_rolling
def winsorize(series: pd.Series, limits: tuple = (0.01, 0.01)) -> pd.Series:
    """Cap at given lower/upper quantiles."""
    return pd.Series(mstats.winsorize(series, limits=limits), index=series.index)

@with_rolling
def boxcox_transform(series: pd.Series, lmbda: Optional[float] = None) -> pd.Series:
    """Box–Cox transform (strictly positive input)."""
    s = series.copy().astype(float)
    if s.min() <= 0:
        s += -s.min() + 1e-6
    transformed, _ = boxcox(s, lmbda=lmbda)
    return pd.Series(transformed, index=series.index)

def minmax(series: pd.Series, window: int) -> pd.Series:
    """Rolling min–max scale into [–1, 1]."""
    rmin = series.rolling(window).min()
    rmax = series.rolling(window).max()
    return 2 * (series - rmin) / (rmax - rmin) - 1

def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score (center & scale)."""
    mu = series.rolling(window).mean()
    sd = series.rolling(window).std()
    return (series - mu) / sd

@with_rolling
def quantile_uniform(series: pd.Series, n_quantiles: int = 1000) -> pd.Series:
    """Map empirical CDF to uniform [0,1]."""
    qt = QuantileTransformer(n_quantiles=min(n_quantiles, len(series)),
                             output_distribution='uniform',
                             random_state=0)
    return pd.Series(qt.fit_transform(series.values.reshape(-1, 1)).flatten(),
                     index=series.index)

@with_rolling
def quantile_gaussian(series: pd.Series, n_quantiles: int = 1000) -> pd.Series:
    """Map empirical CDF to standard Gaussian."""
    qt = QuantileTransformer(n_quantiles=min(n_quantiles, len(series)),
                             output_distribution='normal',
                             random_state=0)
    return pd.Series(qt.fit_transform(series.values.reshape(-1, 1)).flatten(),
                     index=series.index)

# === 3) Registry ===

TRANSFORMATIONS = {
    # Stateless
    "direct":           direct,        
    "diff":             diff,                 #Non-Monotonic
    "pct_change":       pct_change,           #MNon-Monotonic
    "log":              log,                  #Monotonic
    "log1p":            log1p,                #Monotonic
    "log10":            log10,                #Monotonic
    "sqrt":             sqrt,                 #Monotonic
    "cbrt":             cbrt,                 #Monotonic
    "square":           square,               #Monotonic
    "cube":             cube,                 #Monotonic
    "cube_root":        cube_root,            #Monotonic
    "sign_log":         sign_log,             #Monotonic
    "inverse_sinh":     inverse_sinh,         #Monotonic
    "modulus":          modulus,              #Monotonic
    "reciprocal":       reciprocal,           #Monotonic
    "abs":              abs_val,              #Monotonic
    "tanh":             tanh,                 #Monotonic
    "arctan":           arctan,               #Monotonic
    "sigmoid":          sigmoid,              #Monotonic
    "softplus":         softplus,             #Monotonic
    "cumsum":           cumsum,               #Non-Monotonic
    "ema":              ema,                  #Monotonic
    "ema_diff":         ema_diff,             #Monotonic
    "sin":              sin_enc,              #Non-Monotonic
    "cos":              cos_enc,              #Non-Monotonic

    # Parameterized (use window or pre-fit on past only)
    "winsorize":        winsorize,             #Monotonic
    "boxcox":           boxcox_transform,      #Monotonic
    "yeo_johnson":      yeo_johnson_transform, #Monotonic
    "minmax":           minmax,                #Monotonic
    "zscore":           zscore,                #Monotonic
    "quantile_uniform": quantile_uniform,      #Monotonic
    "quantile_gaussian":quantile_gaussian,     #Monotonic
}
