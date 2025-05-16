# engines/backtest_engine.py

import os, warnings
import pandas as pd
import numpy as np
from config.config_manager import ConfigManager
from utils.data_loader import DataLoader
from utils.metrics import calculate_metrics
from utils.utils import calculate_positions, calculate_pnl
from utils.transformation import TRANSFORMATIONS
from scipy.stats import spearmanr

class BacktestEngine:
    def __init__(self, reporter):
        self.reporter = reporter

    def _clean_signal(self, df: pd.DataFrame, warmup: int) -> pd.DataFrame:
        import numpy as np
        import warnings

        # Replace infinities with NaN
        df['signal'] = df['signal'].replace([np.inf, -np.inf], np.nan)

        total    = len(df)
        budget = max(total * 0.03 - warmup, 0.0)

        # Count NaNs after the warmup period
        if warmup > 0 and total > warmup:
            nan_count = int(df['signal'].iloc[warmup:].isna().sum())
        else:
            nan_count = int(df['signal'].isna().sum())

        if nan_count > budget:
            raise ValueError(
                f"rolling_window={warmup} produced {nan_count} NaNs, "
                f"but only {budget:.1f} allowed (3% of {total:.1f} minus warmup)"
            )

        # Drop NaNs but preserve the warmup slice
        if warmup > 0 and total > warmup:
            mask = df['signal'].isna().copy()
            mask.iloc[:warmup] = False
            df = df.loc[~mask]
        else:
            df = df.dropna(subset=['signal'])

        return df

    def run(self, strategy, df: pd.DataFrame, params: dict, logic_plugins: dict) -> dict:
        # Copy inputs
        df = df.copy()
        # Ensure datetime index
        if df.index.name != 'datetime' or not pd.api.types.is_datetime64_any_dtype(df.index):
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('datetime')
            else:
                raise ValueError("DataFrame must have a 'datetime' column or index.")
        # Remove duplicates
        if df.index.duplicated().any():
            dup_count = int(df.index.duplicated().sum())
            warnings.warn(f"{dup_count} duplicate timestamps detected; dropping extras.")
            df = df[~df.index.duplicated(keep='first')]
        # Prepare full index for reindexing
        interval = params.get('interval')
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq=interval)
        missing = full_idx.difference(df.index)

        # Unpack strategy params
        strat      = params.get('strategy_params', {})
        warmup     = int(strat.get('rolling_window', 0))
        long_th    = float(strat.get('long_threshold', 0))
        short_th   = float(strat.get('short_threshold', 0))
        fee        = strat.get('fee', 0.0006)
        factor_col = strat.get('factor_column')
        model      = params.get('model', 'default_model')
        symbol     = params.get('symbol')
        transform  = params.get('transformation', 'direct')
        if transform in TRANSFORMATIONS and transform != "direct" and factor_col in df.columns:
            df[factor_col] = TRANSFORMATIONS[transform](df[factor_col])
            
        # Load candles here (after factor-only df)
        cfg = ConfigManager("config/config.yaml").config
        loader = DataLoader(
            candle_path=cfg['data']['candle_path'],
            datasource_path=cfg['data']['datasource_path'],
            config=cfg,
            debug=False
        )
        # Use same interval for resampling and full range
        shift_override = params.get('shift_override', None)
        candle_df = loader.load_candle_data(
            symbol,
            base_interval=interval,
            resample_interval=interval,
            shift_override=shift_override
        )
        ds_meta = []
        for fname in params.get('datasource_files', []):
            _, meta = loader.load_datasource_data(
                filename          = fname,
                resample_interval = interval,
                factor_column     = factor_col
            )
            ds_meta.append(meta)
        # Prepare logic list
        logics = params.get('entryexit_logic', list(logic_plugins.keys()))
        if isinstance(logics, str):
            logics = [logics]
        results = {}
        for logic_name in logics:
            logic_fn = logic_plugins.get(logic_name)
            if logic_fn is None:
                continue
            # 1) Generate signal
            print(f"[BT] Generating signal for logic '{logic_name}'")
            df_sig = strategy(df.copy(), warmup, factor_col)

            # 2) Clean signal: inf → NaN, drop and budget check
            print(f"[BT] Cleaning signal and checking NaN budget")
            try:
                df_sig = self._clean_signal(df_sig, warmup)
            except ValueError as e:
                raw = str(e)
                print(f"[BT] Backtest skipped (window={warmup}):\n" f"   {raw}\n" f"   → Adjust your rolling_window or data clean-up.")
                continue

            # 3) Compute positions
            print(f"[BT] Computing positions (long_th={long_th}, short_th={short_th})")
            df_pos = calculate_positions(df_sig, logic_fn.apply, long_th, short_th)

            # 4) Reindex to full timeline and forward-fill
            print(f"[BT] Reindexing to full timeline and forward-filling all columns")
            full_idx = pd.date_range(df_pos.index.min(), df_pos.index.max(), freq=interval)
            df_pos = df_pos.reindex(full_idx).ffill()
            df_pos['position'] = df_pos['position'].ffill()
            df_pos['trade'] = df_pos['position'].diff().fillna(0).abs().astype(int)

            # 5) Merge candles after forward-fill
            print(f"[BT] Merging candle data post-ffill")
            candle_full = candle_df.reindex(full_idx).ffill()
            for col in ['close']:
                df_pos[col] = candle_full[col]
            df_pos.index.name = 'datetime'
            df_pos = df_pos.reset_index()

            # 6) Calculate PnL & metrics
            print(f"[BT] Calculating PnL and metrics")
            df_pnl = calculate_pnl(df_pos, fee)
            metrics = calculate_metrics(df_pnl, interval)
            
            # 7) Save outputs
            print(f"[BT] Saving outputs")
            report_dir = self.reporter.create_detailed_report_directory(
                'backtest', symbol, interval, factor_col,
                model, logic_name, transform,
                factor_columns=strat.get('factor_columns')
            )
            out_csv = os.path.join(report_dir, f"backtest_{logic_name}.csv")
            out_json = os.path.join(report_dir, "report.json")
            df_pnl.to_csv(out_csv, index=False)
            report = {}

            report.update(metrics)
            report.update({
                "symbol":         symbol,
                "model":          model,
                "interval":       interval,
                "candle_shift": shift_override,
                "formula":  factor_col
            })

            for i, m in enumerate(ds_meta, start=1):
                report[f"file_{i}"]              = m["file_name"]
                report[f"column_{i}"]            = m["column"]
                report[f"resample_method_{i}"]   = m["resample_method"]
                report[f"resample_interval_{i}"] = m["resample_interval"]

            report.update({
                "transformation":  transform,
                "logic":           logic_name,
                "rolling_window":  strat.get("rolling_window"),
                "long_threshold":  float(strat.get("long_threshold", 0)),
                "short_threshold": float(strat.get("short_threshold", 0))
            })

            self.reporter.save_json(report, out_json)
            results[logic_name] = {
                "output_csv":  out_csv,
                "output_json": out_json,
                "metrics":     metrics
            }
        return results
