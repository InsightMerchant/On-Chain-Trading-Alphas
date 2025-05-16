import pandas as pd
from utils.interfaces import ModelInterface

class Direct(ModelInterface):
    MODEL_CONFIG: dict = {}

    def calculate_signal(self, df, rolling_window, factor_column):
        """
        Just copy the factor column straight into `signal`.
        No scaling, no thresholds.
        """
        if factor_column not in df.columns:
            raise ValueError(f"Factor column '{factor_column}' not found in DataFrame.")
        df['signal'] = df[factor_column]
        return df
