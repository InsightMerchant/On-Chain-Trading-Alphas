import pandas as pd
import os
import re
from PyQt5.QtWidgets import QMessageBox
import numpy as np
from utils.transformation import TRANSFORMATIONS
import csv

SAFE_FUNCS = {
    **TRANSFORMATIONS,
    "np": np,
    "pd": pd,
}

def sanitize_for_folder(name: str) -> str:
    s = name.replace("/", "d").replace("*", "m")
    forbidden_chars = '<>:"\\|?'
    for ch in forbidden_chars:
        s = s.replace(ch, "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_formulas(txt_path):
    formulas = []
    try:
        with open(txt_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                formulas.append(line)
        return formulas

    except Exception as e:
        QMessageBox.warning(None, "Error loading formulas", str(e))
        return []


def parse_formula(formula: str) -> list:
    """
    Extract column names from a formula like df['col'] or df["col"].
    """
    tokens = re.findall(r"df\[\s*['\"]([^'\"]+)['\"]\s*\]", formula)
    if tokens:
        return tokens
    allowed = set(['+', '-', 'd', 'm', '(', ')'])
    parts = re.split(r'([+\-*/()])', formula)
    return [p.strip() for p in parts if p.strip() and p.strip() not in allowed]


def validate_factors(factors, ds_map):
    mapping = {}
    for factor in factors:
        found = False
        for key, (filename, col_name) in ds_map.items():
            base = key.split("(")[0].strip()
            if base == factor:
                mapping[factor] = (filename, col_name)
                found = True
                break
        if not found:
            raise ValueError(f"Factor '{factor}' not found in datasource mapping.")
    return mapping


def merge_factor_data(factor_mapping, datasource_path, interval, data_loader):
    merged_df = None
    for factor, (filename, col_name) in factor_mapping.items():
        path = os.path.join(datasource_path, filename)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"Error reading file {path}: {e}")
        if col_name not in df.columns:
            raise ValueError(f"Column {col_name} not found in {filename}.")
        df = df.rename(columns={col_name: factor})
        if "datetime" not in df.columns:
            raise ValueError(f"'datetime' column not found in {filename}.")
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        df = df.set_index('datetime').resample(interval).last().dropna().reset_index()
        merged_df = df if merged_df is None else pd.merge(merged_df, df, on='datetime', how='inner')
    return merged_df


def evaluate_formula_on_df(df: pd.DataFrame, formula: str):
    env = {"__builtins__": {}}
    env.update(SAFE_FUNCS)
    env['df'] = df

    try:
        result = eval(formula, env)
    except Exception as e:
        raise ValueError(f"Error evaluating `{formula}`: {e}")

    if not isinstance(result, pd.Series):
        raise ValueError(f"Formula must return a pandas Series, got {type(result)}")

    col_name = sanitize_for_folder(formula)
    df[col_name] = result
    return df, col_name


def process_formula(formula, ds_map, datasource_path, interval, data_loader):
    try:
        factors = parse_formula(formula)
        factor_map = validate_factors(factors, ds_map)
        merged = merge_factor_data(factor_map, datasource_path, interval, data_loader)
        result_df, col = evaluate_formula_on_df(merged, formula)
        return result_df
    except Exception as e:
        QMessageBox.warning(None, "Formula Error", str(e))
        return pd.DataFrame()
