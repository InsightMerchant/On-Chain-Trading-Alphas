import pandas as pd
from utils.interfaces import ModelInterface

class MeanNormal(ModelInterface):
    MODEL_CONFIG = {
        "rolling_window": {
            "default": 173,
            "type": int,
            "description": "Window size for mean/min/max calculations"
        }
    }
    
    def calculate_signal(self, df, rolling_window, factor_column):
        df = df.copy()
        df['mean'] = df[factor_column].rolling(window=rolling_window).mean()
        df['rolling_min'] = df[factor_column].rolling(window=rolling_window).min()
        df['rolling_max'] = df[factor_column].rolling(window=rolling_window).max()
        df['scaled'] = (df[factor_column] - df['mean']) / (df['rolling_max'] - df['rolling_min'])
        df['signal'] = df['scaled']
        return df
