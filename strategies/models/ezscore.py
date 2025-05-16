# ===== ezscorev1.py =====
import pandas as pd
from utils.interfaces import ModelInterface

class EZScore(ModelInterface):
    MODEL_CONFIG = {
        "rolling_window": {
            "default": 20,
            "type": int,
            "description": "Span for EWM and window size for rolling std in ez-score computation"
        }
    }
    
    def calculate_signal(self, df, rolling_window, factor_column):
        df = df.copy()
        # Exponential moving average
        df['ema'] = (
            df[factor_column]
            .ewm(span=rolling_window, adjust=False, min_periods=rolling_window)
            .mean()
        )
        # Rolling standard deviation
        df['std'] = df[factor_column].rolling(window=rolling_window).std()
        # Calculate z-score
        df['zscore'] = (df[factor_column] - df['ema']) / df['std']
        # Expose as the strategy’s signal
        df['signal'] = df['zscore']
        return df
