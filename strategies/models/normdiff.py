# ===== normdiff.py =====
import pandas as pd
from utils.interfaces import ModelInterface

class NormalizedDiff(ModelInterface):
    MODEL_CONFIG = {
        "rolling_window": {
            "default": 20,
            "type": int,
            "description": "Period for difference and rolling std normalization"
        }
    }

    def calculate_signal(self, df, rolling_window, factor_column):
        """
        Normalized Difference:
        diff = x_t - x_{t - rolling_window}
        normalized_diff = diff / STD(x over past rolling_window)
        """
        df = df.copy()
        df['diff'] = df[factor_column].diff(periods=rolling_window)
        df['normalized_diff'] = (
            df['diff'] /
            df[factor_column].rolling(window=rolling_window).std()
        )
        df['signal'] = df['normalized_diff']
        return df
