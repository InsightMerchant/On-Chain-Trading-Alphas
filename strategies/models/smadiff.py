# ===== smadiff.py =====
import pandas as pd
import numpy as np
from utils.interfaces import ModelInterface

class SimpleMovingAverageDiff(ModelInterface):
    MODEL_CONFIG = {
        "rolling_window": {
            "default": 20,
            "type": int,
            "description": "Lookback period for Simple Moving Average calculation"
        }
    }
    
    def calculate_signal(self, df, rolling_window, factor_column):
        """
        Simple Moving Average Difference (SMADIFF):
        Calculates the difference between the data column and its simple moving average
        """
        df = df.copy()
        # Compute SMA over the specified lookback
        df['temp_data'] = df[factor_column].rolling(window=rolling_window).mean()
        # Calculate difference between data and its moving average
        df['smadiff'] = df[factor_column] - df['temp_data']
        df['signal'] = df['smadiff']
        return df