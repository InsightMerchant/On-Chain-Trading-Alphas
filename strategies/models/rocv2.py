# ===== roc_v2.py =====
import pandas as pd
from utils.interfaces import ModelInterface

class RateOfChangeV2(ModelInterface):
    MODEL_CONFIG = {
        "rolling_window": {
            "default": 20,
            "type": int,
            "description": "Lookback period for ROCv2 calculation"
        }
    }

    def calculate_signal(self, df, rolling_window, factor_column):
        df = df.copy()
        df['rocv2'] = (
            df[factor_column] - df[factor_column].shift(rolling_window)
        ) / df[factor_column]
        df['signal'] = df['rocv2']
        return df
