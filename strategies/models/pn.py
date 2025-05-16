import pandas as pd
from utils.interfaces import ModelInterface

class PercentileNorm(ModelInterface):
    MODEL_CONFIG = {
        "rolling_window": {
            "default": 173,
            "type": int,
            "description": "Window size for mean/min/max calculations"
        }
    }
    
    def calculate_signal(self, df, rolling_window, factor_column):
        df = df.copy()
        df['25_percentile'] = df[factor_column].rolling(window=rolling_window).quantile(0.25)
        df['75_percentile'] = df[factor_column].rolling(window=rolling_window).quantile(0.75)
        df['scale'] = (df[factor_column] - df['25_percentile'])/(df['75_percentile'] - df['25_percentile'])
        df['signal'] = df['scale']
        return df
