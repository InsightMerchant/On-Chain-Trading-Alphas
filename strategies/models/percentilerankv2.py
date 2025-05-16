# ===== percentilerankv2.py =====
import pandas as pd
from utils.interfaces import ModelInterface

class PercentileRankV2(ModelInterface):
    MODEL_CONFIG = {
        "rolling_window": {
            "default": 20,
            "type": int,
            "description": "Window size for rolling percentile rank calculation"
        }
    }

    def calculate_signal(self, df, rolling_window, factor_column):
        df = df.copy()
        df['percentile_rank'] = (
            df[factor_column]
            .rolling(window=rolling_window)
            .apply(lambda x: x.rank(pct=True).iloc[-1] - 0.5, raw=False)
        )
        df['signal'] = df['percentile_rank']
        return df
