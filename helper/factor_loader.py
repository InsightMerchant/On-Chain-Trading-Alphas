import os
import pandas as pd
from utils.formula import parse_formula, evaluate_formula_on_df
from helper.shift_helper import (
    determine_shift_override_for_formula,
    determine_shift_override
)

def load_factor_dataframe(
    data_loader,
    config: dict,
    ds_map: dict,
    symbol: str,
    interval: str,
    formula: str,
    manual_factors: list
) -> tuple[pd.DataFrame, str, int, list[str]]:
    sym = symbol.lower()
    def _keep(fn: str) -> bool:
        fn = os.path.basename(fn).lower()
        return fn.startswith(f"{sym}_") or fn.startswith(f"gn_{sym}_") or fn.startswith(f"cq_{sym}_")

    ds_map = {k: v for k, v in ds_map.items() if _keep(v[0])}
    if not ds_map:
        raise ValueError(f"No datasource files for symbol '{symbol}'")

    def match_key(tok: str) -> str:
        return next((k for k in ds_map if k.split('(')[0].strip() == tok), None)

    # ── Formula mode ─────────────────────────────────────────────────────────
    if formula and formula != "<none>":
        tokens = parse_formula(formula)
        factor_map = {}
        missing = []
        for tok in tokens:
            key = match_key(tok)
            if key:
                factor_map[tok] = ds_map[key]
            else:
                missing.append(tok)
        if missing:
            raise ValueError(f"Missing factors for formula: {missing}")

        shift = determine_shift_override_for_formula(factor_map, config)

        # load & clean each raw factor individually
        iterator = iter(factor_map.items())
        first_tok, (first_file, first_col) = next(iterator)
        first_df, _ = data_loader.load_datasource_data(
            first_file,
            resample_interval=interval,
            factor_column=first_col
        )
        df = first_df.rename(columns={first_col: first_tok}).dropna()

        # now merge the rest of the tokens
        for tok, (fname, col) in iterator:
            ds_df, _ = data_loader.load_datasource_data(
                fname,
                resample_interval=interval,
                factor_column=col
            )
            ds = ds_df.rename(columns={col: tok}).dropna()
            df = pd.merge(df, ds, on="datetime", how="inner")

        if df.empty:
            raise ValueError("No overlapping data for formula factors after cleaning.")

        df, safe_col = evaluate_formula_on_df(df, formula)
        files = [path for path, _ in factor_map.values()]
        return df, safe_col, shift, files


    # ── Manual factors mode ─────────────────────────────────────────────────
    else:
        if not manual_factors or not manual_factors[0]:
            raise ValueError("No factor selected.")
        if len(set(manual_factors)) != len(manual_factors):
            raise ValueError("Duplicate factor selections.")

        first_file, first_col = ds_map[manual_factors[0]]
        shift = determine_shift_override(first_file, config)

        # Unpack the tuple here (2 indent levels under `else:`)
        first_df, _ = data_loader.load_datasource_data(
            first_file,
            resample_interval=interval,
            factor_column=first_col
        )
        df = first_df.dropna()

        for f in manual_factors[1:]:
            fname, col = ds_map[f]
            # Also unpack here
            ds_df, _ = data_loader.load_datasource_data(
                fname,
                resample_interval=interval,
                factor_column=col
            )
            ds = ds_df.dropna()
            df = pd.merge(df, ds, on="datetime", how="inner")

        if df.empty:
            raise ValueError("Merged factor data is empty after cleaning.")

        factor_col = ds_map[manual_factors[0]][1]
        files = [ds_map[f][0] for f in manual_factors]
        return df, factor_col, shift, files
