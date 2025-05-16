# ===== roc.py =====
import pandas as pd
from utils.interfaces import ModelInterface

class RateOfChange(ModelInterface):
    MODEL_CONFIG = {
        "rolling_window": {
            "default": 20,
            "type": int,
            "description": "Lookback period for Rate of Change calculation"
        }
    }

    def calculate_signal(self, df, rolling_window, factor_column):
        """
        Rate of Change (ROC):
        roc = ( (x_t / x_{t - rolling_window} - 1 ) * 100 ) / rolling_window
        """
        df = df.copy()
        # Compute ROC over the specified lookback
        df['roc'] = (
            (df[factor_column] / df[factor_column].shift(rolling_window) - 1)
            * 100
            / rolling_window
        )
        df['signal'] = df['roc']
        return df
