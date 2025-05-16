import os
import matplotlib.pyplot as plt
import pandas as pd

from config.config_manager import ConfigManager
from utils.data_loader import DataLoader
from engines.backtest_engine import BacktestEngine

class WalkforwardEngine:
    def __init__(self, reporter):
        self.reporter  = reporter
        self.bt_engine = BacktestEngine(reporter)

    def train_test_split(self, df, train_ratio, date_column="datetime"):
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        split_index = int(len(df) * train_ratio)
        train_df = df.iloc[:split_index].reset_index(drop=True)
        test_df  = df.iloc[split_index:].reset_index(drop=True)
        return train_df, test_df

    def run(self, strategy, df, params, logic_plugins):
        train_ratio = params.get("train_ratio")
        if train_ratio is None:
            raise ValueError("`train_ratio` is required for train–test split.")
        train_df, test_df = self.train_test_split(
            df,
            train_ratio,
            date_column=params.get("date_column", "datetime")
        )

        symbol          = params["symbol"]
        interval        = params["interval"]
        strategy_params = params.get("strategy_params", {})
        factor_col      = strategy_params.get("factor_column")
        cfg = ConfigManager("config/config.yaml").config
        loader = DataLoader(
            candle_path=cfg['data']['candle_path'],
            datasource_path=cfg['data']['datasource_path'],
            config=cfg,
            debug=False
        )
        shift_override = params.get("shift_override")
        ds_meta = []
        for fname in params.get('datasource_files', []):
            _, meta = loader.load_datasource_data(
                filename          = fname,
                resample_interval = interval,
                factor_column     = factor_col  
            )
            ds_meta.append(meta)

        overall = {}
        logics  = params.get("entryexit_logic", list(logic_plugins.keys()))
        if isinstance(logics, str):
            logics = [logics]

        for logic_name in logics:
            logic_fn = logic_plugins.get(logic_name)
            if not logic_fn:
                continue

            wf_params = params.copy()
            wf_params["entryexit_logic"] = logic_name
            single_logic = { logic_name: logic_fn }

            # 2) in-sample backtest on train
            in_res = self.bt_engine.run(strategy, train_df, wf_params, single_logic)
            m_tr   = in_res[logic_name]["metrics"].copy()
            m_tr.pop("selected_config", None)
            in_csv = in_res[logic_name]["output_csv"]
            df_in  = pd.read_csv(in_csv, parse_dates=["datetime"])

            # 3) forward backtest on test only
            fw_res = self.bt_engine.run(strategy, test_df, wf_params, single_logic)
            m_fw   = fw_res[logic_name]["metrics"].copy()
            m_fw.pop("selected_config", None)
            out_csv = fw_res[logic_name]["output_csv"]
            df_out  = pd.read_csv(out_csv, parse_dates=["datetime"])

            # 4) compute sharpe difference
            sr_tr = m_tr.get("SR", m_tr.get("sharpe_ratio", 0))
            sr_fw = m_fw.get("SR", m_fw.get("sharpe_ratio", 0))
            sharpe_diff_pct = (
                abs(sr_fw - sr_tr) / ((sr_fw + sr_tr) / 2) * 100
                if (sr_fw + sr_tr) else 0
            )

            # 5) plot equity curves, marking the split boundary
            eq_tr = df_in["pnl"].cumsum()
            eq_fw = df_out["pnl"].cumsum() + eq_tr.iloc[-1]
            plt.figure(figsize=(10,6))
            plt.plot(df_in["datetime"], eq_tr, label="in_sample")
            plt.plot(df_out["datetime"], eq_fw, label="forward_test")
            plt.axvline(df_in["datetime"].max(), color="k", linestyle="--")
            plt.legend(); plt.xticks(rotation=45)
            plt.title(f"Train–Test equity ({logic_name})")

            report_dir   = self.reporter.create_detailed_report_directory(
                "walkforward", symbol, interval, factor_col,
                params.get("model"), logic_name,
                params.get("transformation","None"),
                factor_columns=strategy_params.get("factor_columns")
            )
            combined_png = os.path.join(report_dir, "combined_equity_curve.png")
            plt.savefig(combined_png, bbox_inches="tight")
            plt.close()
            in_dest  = os.path.join(report_dir, "in_sample.csv")
            out_dest = os.path.join(report_dir, "out_sample.csv")
            df_in.to_csv(in_dest, index=False)
            df_out.to_csv(out_dest, index=False)
            sc = {
                "symbol":         symbol,
                "model":          params.get("model"),
                "interval":       interval,
                "candle_shift":   shift_override,
                "formula":        factor_col
            }
            for i, m in enumerate(ds_meta, start=1):
                sc[f"file_{i}"]              = m["file_name"]
                sc[f"column_{i}"]            = m["column"]
                sc[f"resample_method_{i}"]   = m["resample_method"]
                sc[f"resample_interval_{i}"] = m["resample_interval"]

            sc.update({
                "transformation":  params.get("transformation","None"),
                "logic":           logic_name,
                "rolling_window":  strategy_params.get("rolling_window"),
                "long_threshold":  strategy_params.get("long_threshold"),
                "short_threshold": strategy_params.get("short_threshold")
            })

            merged = {
                "in-sample":              m_tr,
                "out-sample":             m_fw,
                "sharpe_ratio_diff_pct": sharpe_diff_pct,
                "selected_config":       sc
            }
            merged_json = os.path.join(report_dir, "walkforward_report.json")
            self.reporter.save_json(merged, merged_json)

            overall[logic_name] = {
                "report_json":    merged_json,
                "combined_plot":  combined_png,
                "in_sample_csv":  in_dest,
                "out_sample_csv": out_dest
            }

        return overall
