import os
import pandas as pd
import shutil
from itertools import product, permutations, combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from .permutation_engine import PermutationEngine
from utils.datasource_utils import aggregate_factor_options
from utils.interval_helper import (
    format_interval, parse_interval, extract_interval_from_filename, get_common_intervals
)
from utils.timestamp_utils import get_canonical_timestamp_column
from helper.factor_loader import load_factor_dataframe
from utils.formula import load_formulas, parse_formula
from utils.formula import sanitize_for_folder
import hashlib
import warnings

def meets_bruteforce_criteria(metrics, threshold=1.2):
    sharpe = metrics.get("sharpe_ratio", metrics.get("SR", 0)) or 0
    return sharpe >= threshold

class BruteforceEngine:
    def __init__(self, config, data_loader, model_loader, logic_plugins, reporter, extra_params, max_workers=None):
        self.config = config
        self.data_loader = data_loader
        self.model_loader = model_loader
        self.logic_plugins = logic_plugins
        self.reporter = reporter
        self.extra_params = extra_params
        self.max_workers = max_workers or os.cpu_count() or 1

        # unpack selections
        self.symbols = extra_params.get("selected_symbols", [])
        self.intervals = extra_params.get("selected_intervals", [])
        self.models = extra_params.get("selected_models", [])
        self.logics = extra_params.get("selected_logics", [])
        self.formulas = extra_params.get("formula", ["<none>"])
        self.model_params = extra_params.get("model_params", {})
        self.logic_config = extra_params.get("logic_config", {})
        self.apply_criteria = extra_params.get("apply_bruteforce_criteria", True)
        self.transformation = extra_params.get("transformation", "None")

        # pre-load datasource mapping
        _, self.ds_map, self.ds_intervals = aggregate_factor_options(
            config["data"]["datasource_path"]
        )

    def _generate_factor_combos(self, model_name):
        _, model_config = self.model_loader.load_model(model_name)
        use_multi = model_config.get("use_multiple_factors", False)
        r = model_config.get("num_factors", 1)
        factors = self.extra_params.get("selected_factors", [])

        # formula mode short-circuit
        if self.formulas[0] != "<none>":
            return [()] 

        if use_multi:
            if model_config.get("order_sensitive", False):
                return permutations(factors, r)
            return combinations(factors, r)

        return combinations(factors, 1)

    def _prepare_combos(self):
        combos = []
        # include formulas in the outer product
        for (symbol, interval, model, logic, formula) in product(
            self.symbols,
            self.intervals,
            self.models,
            self.logics,
            self.formulas
        ):
            for factor_combo in self._generate_factor_combos(model):
                combos.append((symbol, interval, model, logic, factor_combo, formula))
        return combos

    def _process_one(self, combo):
        symbol, interval, model_choice, logic_choice, factor_combo, formula = combo
        mode_formula = (formula not in [None, "<none>"])

        try:
            df, factor_key, shift_val, _ = load_factor_dataframe(
                self.data_loader,
                self.config,
                self.ds_map,
                symbol,
                interval,
                formula,
                [] if mode_formula else list(factor_combo)
            )
            print(f"[BF] Shift {shift_val}")
            if df.empty:
                return None

            strat_params = self.model_params.get(model_choice, {}).copy()
            if mode_formula:
                strat_params["factor_column"] = factor_key
            else:
                if len(factor_combo) == 1:
                    strat_params["factor_column"] = factor_key
                else:
                    strat_params["factor_columns"] = [
                        self.ds_map[f][1] for f in factor_combo
                    ]
            # merge in any logic-specific config
            per_logic_cfg = self.logic_config.get(logic_choice, {})
            strat_params.update(per_logic_cfg)

            # 4) isolate the permuted parameters
            perm_params = {
                k: v for k, v in strat_params.items()
                if isinstance(v, list)
            }
            perm_full = {
                "model": model_choice,
                "entryexit_logic": logic_choice,
                "strategy_params": strat_params,
                "permutation_params": perm_params,
                "symbol": symbol,
                "interval": interval,
                "transformation": self.transformation,
                "report_mode": "bruteforce",
                "train_ratio": self.config.get("train_ratio", 0.65),
                "shift_override": shift_val,
                "logic_config": per_logic_cfg
            }

            pe = PermutationEngine(self.reporter)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                results, _ = pe.run(
                    self.model_loader.load_model(model_choice)[0],
                    df,
                    perm_full,
                    {logic_choice: self.logic_plugins[logic_choice]}
                )

            combos = results.get(logic_choice, [])
            if not combos:
                path = self.reporter.create_detailed_report_directory(
                    "bruteforce", symbol, interval,
                    strat_params.get("factor_column"),
                    model_choice, logic_choice,
                    self.transformation,
                    factor_columns=strat_params.get("factor_columns")
                )
                shutil.rmtree(path, ignore_errors=True)
                print(f"[BF] {combo} → no permutations; folder removed.")
                return None

            # 6) apply the criteria filter
            threshold = 1.2 if "short" in logic_choice.lower() else 1.8
            passed = [
                rec for rec in combos
                if meets_bruteforce_criteria(rec["metrics"], threshold)
            ]

            if self.apply_criteria:
                if passed:
                    print(f"[BF] {combo} → MEETS criteria (threshold={threshold})")
                else:
                    print(f"[BF] {combo} → fails criteria; folder removed.")
                    path = self.reporter.create_detailed_report_directory(
                        "bruteforce", symbol, interval,
                        strat_params.get("factor_column"),
                        model_choice, logic_choice,
                        self.transformation,
                        factor_columns=strat_params.get("factor_columns")
                    )
                    shutil.rmtree(path, ignore_errors=True)
                    return None
            else:
                print(f"[BF] {combo} → criteria disabled; proceeding.")

            return (combo, results)

        except Exception as e:
            print(f"Error in bruteforce combo {combo}: {e}")
            return None

    def run(self):
        overall = {}
        combos = self._prepare_combos()
        total = len(combos)
        print(f"[BF] Starting bruteforce: {total} total combinations")
        for idx, combo in enumerate(combos, start=1):
            print(f"[BF] Processing {idx}/{total} combinations")
            res = self._process_one(combo)
            if res:
                key, result = res
                overall[key] = result
            print(f"[BF] Completed {idx}/{total} combinations")
        print(f"[BF] Completed all {total} combinations")
        return overall

def run_bruteforce_mode(config, data_loader, model_loader, logic_plugins, reporter, extra_params):
    engine = BruteforceEngine(config, data_loader, model_loader, logic_plugins, reporter, extra_params)
    return engine.run()
