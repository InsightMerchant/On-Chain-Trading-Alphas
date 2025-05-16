# strategies/models/zscore_model.py
import pandas as pd
import numpy as np
from utils.interfaces import ModelInterface

class ZScore(ModelInterface):
    MODEL_CONFIG = {
        "rolling_window": {
            "default": 20,
            "type": int,
            "description": "Window size for calculating rolling mean and std in z-score computation"
        }
    }
    
    def calculate_signal(self, df, rolling_window, factor_column):
        df = df.copy()
        df['sma'] = df[factor_column].rolling(window=rolling_window).mean()
        df['std'] = df[factor_column].rolling(window=rolling_window).std()
        df['zscore'] = (df[factor_column] - df['sma']) / df['std']
        df['signal'] = df['zscore']
        return df
