# ===== percentilerank.py =====
import pandas as pd
from utils.interfaces import ModelInterface

class PercentileRank(ModelInterface):
    MODEL_CONFIG = {
        "rolling_window": {
            "default": 20,
            "type": int,
            "description": "Window size for rolling percentile rank calculation"
        }
    }

    def calculate_signal(self, df, rolling_window, factor_column):
        """
        Rolling Percentile Rank:
        percentile_rank = (rank_pct(x_t within past rolling_window) * 2) - 1
        """
        df = df.copy()
        # Compute rolling percentile rank scaled to [-1, 1]
        df['percentile_rank'] = (
            df[factor_column]
            .rolling(window=rolling_window)
            .apply(lambda x: x.rank(pct=True).iloc[-1] * 2 - 1, raw=False)
        )
        df['signal'] = df['percentile_rank']
        return df
