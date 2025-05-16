import pandas as pd
from utils.interfaces import ModelInterface

class RateOfChangeV1(ModelInterface):
    MODEL_CONFIG = {
        "rolling_window": {
            "default": 20,
            "type": int,
            "description": "Lookback period for Rate of Change calculation"
        }
    }

    def calculate_signal(self, df, rolling_window, factor_column):
        df = df.copy()
        df['rocv1'] = (
            df[factor_column] / df[factor_column].shift(rolling_window) - 1
        ) * 100
        df['signal'] = df['rocv1']
        return df
