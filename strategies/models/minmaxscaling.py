import pandas as pd
from utils.interfaces import ModelInterface

class MinMaxScaling(ModelInterface):
    MODEL_CONFIG = {
        "rolling_window": {
            "default": 20,
            "type": int,
            "description": "Window size for min/max calculations"
        }
    }
    
    def calculate_signal(self, df, rolling_window, factor_column):
        df = df.copy()
        df['rolling_min'] = df[factor_column].rolling(window=rolling_window).min()
        df['rolling_max'] = df[factor_column].rolling(window=rolling_window).max()
        df['scaled'] =2*((df[factor_column] - df['rolling_min']) / (df['rolling_max'] - df['rolling_min'])) - 1
        df['signal'] = df['scaled']
        return df
