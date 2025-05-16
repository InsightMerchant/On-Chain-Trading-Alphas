import pandas as pd
from utils.interfaces import ModelInterface

class RobustScaler(ModelInterface):
    MODEL_CONFIG = {
        "rolling_window": {
            "default": 173,
            "type": int,
            "description": "Window size for calculating rolling median and IQR"
        }
    }
    
    def calculate_signal(self, df, rolling_window, factor_column):
        df = df.copy()
        df['rolling_median'] = df[factor_column].rolling(window=rolling_window).median()
        df['rolling_q1'] = df[factor_column].rolling(window=rolling_window).quantile(0.25)
        df['rolling_q3'] = df[factor_column].rolling(window=rolling_window).quantile(0.75)
        df['rolling_iqr'] = df['rolling_q3'] - df['rolling_q1']
        epsilon = 1e-8
        df['scaled'] = (df[factor_column] - df['rolling_median']) / (df['rolling_iqr'] + epsilon)
        df['signal'] = df['scaled']
        return df
