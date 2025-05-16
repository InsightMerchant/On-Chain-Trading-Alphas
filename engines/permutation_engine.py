import os
import json
import inspect
import tempfile
from itertools import product
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from tqdm import tqdm
from utils.model_loader import ModelLoader
from engines.backtest_engine import BacktestEngine
from utils.reporting import create_heatmap, Reporter

# ——————————————————————————————————————————————————————————————————————————————————
# global verbosity flag
PERMUTATION_VERBOSE = False

def set_permutation_verbose(flag: bool):
    """Turn PERMUTATION_VERBOSE on/off from the GUI."""
    global PERMUTATION_VERBOSE
    PERMUTATION_VERBOSE = flag
# ——————————————————————————————————————————————————————————————————————————————————

class SilentReporter(Reporter):
    """Reporter that does nothing for internal backtest runs."""
    def create_detailed_report_directory(self, *args, **kwargs):
        # Do not create any directory; return a dummy path
        return ""
    def save_json_with_config(self, *args, **kwargs):
        pass
    def save_json(self, *args, **kwargs):
        pass
    def save_plot(self, *args, **kwargs):
        pass

class PermutationEngine:
    def __init__(self, reporter):
        # reporter for permutation outputs (JSON + heatmap)
        self.reporter = reporter
        # silent reporter for internal backtests
        self._silent_reporter = SilentReporter()
        # backtest engine uses silent reporter
        self.bt_engine = BacktestEngine(self._silent_reporter)

    def run(self, strategy, df, params, logic_plugins):
        base_params        = params.get("strategy_params", {})
        permutation_params = params.get("permutation_params", {})
        model              = params.get("model", "default_model")
        symbol             = params.get("symbol", "unknown_symbol")
        interval           = params.get("interval", "unknown_interval")
        report_mode        = params.get("report_mode", "permutation")
        train_ratio        = params.get("train_ratio", 0.65)
        shift_override     = params.get("shift_override")

        prefix = "[BF]" if report_mode.lower() == "bruteforce" else "[PM]"

        print(f"{prefix} Running {symbol} {interval} {model} "
              f"{base_params.get('factor_column','')} "
              f"{params.get('transformation','None')} "
              f"logic={params.get('entryexit_logic')}")

        df_train = df.iloc[:int(len(df) * train_ratio)].copy()

        entryexit_logic = params.get("entryexit_logic", list(logic_plugins.keys()))
        if isinstance(entryexit_logic, str):
            entryexit_logic = [entryexit_logic]

        loader = ModelLoader()
        all_results      = {}
        filtered_results = {}

        for logic_name in entryexit_logic:
            plugin = logic_plugins.get(logic_name)
            if plugin is None:
                continue

            _, model_config = loader.load_model(model)
            transformation  = params.get("transformation", "None")
            report_dir      = self.reporter.create_detailed_report_directory(
                report_mode,
                symbol,
                interval,
                base_params.get("factor_column", "unknown_factor"),
                model,
                logic_name,
                transformation,
                factor_columns=base_params.get("factor_columns"),
            )
            os.makedirs(report_dir, exist_ok=True)
            json_path = os.path.join(report_dir, "permutation_report_all.json")

            keys   = list(permutation_params.keys())
            combos = list(product(*[permutation_params[k] for k in keys])) if keys else [()]

            records = []
            with ProcessPoolExecutor(max_workers=2) as executor:
                tasks = []
                for combo in combos:
                    tasks.append((
                        self._silent_reporter,
                        {keys[i]: combo[i] for i in range(len(combo))},
                        base_params,
                        strategy,
                        df_train.copy(),
                        list(inspect.signature(strategy).parameters.keys())[1:],
                        plugin,
                        model,
                        model_config,
                        logic_name,
                        interval,
                        symbol,
                        transformation,
                        params.get("logic_config", {}),
                        shift_override,
                        prefix,
                    ))

                for rec in tqdm(
                    executor.map(self._process_combo, tasks),
                    total=len(tasks),
                    desc=prefix,
                    disable=PERMUTATION_VERBOSE,
                    unit="combo",
                    dynamic_ncols=True,
                    leave=True
                ):
                    if rec:
                        records.append(rec)

            if not records:
                if PERMUTATION_VERBOSE:
                    print(f"{prefix} No successful permutations for "
                          f"{symbol} {interval} {model} {logic_name}; skipping output.")
                all_results[logic_name]      = []
                filtered_results[logic_name] = []
                continue

            with open(json_path, "w") as f:
                json.dump(records, f, indent=4)
            if PERMUTATION_VERBOSE:
                print(f"{prefix} Saved permutation JSON to {json_path}")

            create_heatmap(records, report_dir)
            heatmap_path = os.path.join(report_dir, "heatmap.png")
            if PERMUTATION_VERBOSE:
                print(f"{prefix} Generated heatmap at {heatmap_path}")

            all_results[logic_name]      = records
            filtered_results[logic_name] = [r for r in records if r.get("metrics", {}).get("SR") is not None]

        return all_results, filtered_results

    @staticmethod
    def _process_combo(args):
        (reporter,
         combo_dict,
         base_params,
         strategy,
         df,
         required_params,
         plugin,
         model,
         model_config,
         logic_name,
         interval,
         symbol,
         transformation_value,
         logic_config,
         shift_override,
         prefix) = args

        bt_params = {
            "strategy_params": { **base_params, **combo_dict },
            "entryexit_logic": logic_name,
            "model":           model,
            "symbol":          symbol,
            "interval":        interval,
            "transformation":  transformation_value,
            "shift_override":  shift_override
        }

        # Monkey-patch DataFrame.to_csv to no-op while running backtest
        orig_to_csv = pd.DataFrame.to_csv
        pd.DataFrame.to_csv = lambda *a, **k: None

        import sys, io, warnings
        _old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=r"Skipping backtest for rolling_window=.*"
                )
                engine = BacktestEngine(reporter)
                result = engine.run(strategy, df, bt_params, { logic_name: plugin })
        finally:
            sys.stdout = _old_stdout
            # restore original to_csv
            pd.DataFrame.to_csv = orig_to_csv

        logic_res = result.get(logic_name, {}) or {}
        metrics   = logic_res.get("metrics", {}).copy()
        metrics.pop("selected_config", None)

        sr  = metrics.get("SR", metrics.get("sharpe_ratio", None))
        mdd = metrics.get("MDD", metrics.get("max_drawdown", None))
        trades = metrics.get("num_of_trades", metrics.get("trade_numbers", None))

        status = "complete" if metrics else "skip_backtest"

        if PERMUTATION_VERBOSE:
            rw = bt_params["strategy_params"].get("rolling_window", "")
            lt = bt_params["strategy_params"].get("long_threshold", "")
            st = bt_params["strategy_params"].get("short_threshold", "")
            print(
                f"{prefix} {symbol} {interval} {model} "
                f"{base_params.get('factor_column','')}\t"
                f"{transformation_value}\t{logic_name}\t"
                f"rolling_window:{rw} "
                f"long_threshold:{lt} "
                f"short_threshold:{st} "
                f"SR:{sr} "
                f"MDD:{mdd} "
                f"trades:{trades} "
                f"({status})"
            )

        filtered = {k: combo_dict[k] for k in combo_dict if k in model_config or k in logic_config}

        return {
            "symbol":           symbol,
            "interval":         interval,
            "factor_column":    base_params.get("factor_column"),
            "transformation":   transformation_value,
            "model":            model,
            "entry_exit_logic": logic_name,
            "params":           filtered,
            "metrics":          metrics
        }
