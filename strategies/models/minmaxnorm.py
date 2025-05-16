# ===== minmaxnorm.py =====
import pandas as pd
from utils.interfaces import ModelInterface

class MinMaxNorm(ModelInterface):
    MODEL_CONFIG = {
        "rolling_window": {
            "default": 20,
            "type": int,
            "description": "Window size for rolling min/max normalization"
        }
    }

    def calculate_signal(self, df, rolling_window, factor_column):
        df = df.copy()
        df['rolling_min'] = df[factor_column].rolling(window=rolling_window).min()
        df['rolling_max'] = df[factor_column].rolling(window=rolling_window).max()
        df['min_max_scaling'] = (
            (df[factor_column] - df['rolling_min']) /
            (df['rolling_max'] - df['rolling_min'])
        )
        df['signal'] = df['min_max_scaling']
        return df
