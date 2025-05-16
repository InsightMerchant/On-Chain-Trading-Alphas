import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def sanitize_factor_column(factor_column):
    if not factor_column:
        return ""
    return factor_column.rsplit(":", 1)[-1].strip()

class Reporter:
    def __init__(self, base_dir="reports"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
    
    def create_report_directory(self, *folders):
        path = os.path.join(self.base_dir, *folders)
        os.makedirs(path, exist_ok=True)
        return path
    
    def create_detailed_report_directory(self, mode, symbol, interval, factor_column, model, entryexit_logic, transformation="None", factor_columns=None):
        if factor_columns and isinstance(factor_columns, list):
            safe_factor = "_".join(sanitize_factor_column(f) for f in factor_columns)
        else:
            safe_factor = sanitize_factor_column(factor_column)

        safe_model = model.split('.')[-1] if '.' in model else model
        folder_name = f"{symbol}_{safe_factor}_{transformation}_{interval}_{safe_model}_{entryexit_logic}"

        if mode.lower() in ["backtest", "walkforward", "permutation", "bruteforce"]:
            path = os.path.join(self.base_dir, mode.lower(), folder_name)
        else:
            path = os.path.join(self.base_dir, "others", folder_name)

        os.makedirs(path, exist_ok=True)
        return path

    def save_json(self, data, filepath):
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    def save_json_with_config(self, report_data, config_data, filepath):
        report_data["selected_config"] = config_data
        self.save_json(report_data, filepath)

    def save_plot(self, plt_obj, filepath):
        plt_obj.savefig(filepath, bbox_inches="tight")
        plt_obj.close()


def create_heatmap(results: list[dict], report_dir: str):
    def format_val(val):
        return f"{val:.3f}" if isinstance(val, float) else str(val)

    combined = []
    for res in results:
        row = {f"param_{k}": v for k, v in res["params"].items()}
        row.update(res["metrics"])
        combined.append(row)
    df = pd.DataFrame(combined)
    if df.empty:
        return

    param_cols = [c for c in df.columns if c.startswith("param_")]
    has_rw = "param_rolling_window"  in param_cols
    has_lt = "param_long_threshold"  in param_cols
    has_st = "param_short_threshold" in param_cols

    metric_candidates = [c for c in df.columns if not c.startswith("param_")]
    numeric = [c for c in metric_candidates if pd.api.types.is_numeric_dtype(df[c])]
    metric = "SR" if "SR" in numeric else (numeric[-1] if numeric else None)
    if metric is None:
        print("No numeric metric for heatmap")
        return

    if has_rw:
        df["row_key"] = df["param_rolling_window"].apply(format_val)
        row_param = "param_rolling_window"
    else:
        df["row_key"] = df[param_cols[0]].apply(format_val)
        row_param = param_cols[0]

    if has_rw and has_lt and has_st:
        df["col_key"] = df.apply(
            lambda r: f"L:{format_val(r['param_long_threshold'])} | S:{format_val(r['param_short_threshold'])}",
            axis=1
        )
        pivot = df.pivot_table(
            index="row_key",
            columns="col_key",
            values=metric,
            aggfunc="mean"
        )
        pivot = pivot.reindex(
            sorted(pivot.index, key=lambda x: float(x)),
            axis=0
        )
        def sort_key(label):
            long_str, short_str = label.split(" | ")
            l = float(long_str.split(":",1)[1])
            s = float(short_str.split(":",1)[1])
            return (l, s)
        pivot = pivot[sorted(pivot.columns, key=sort_key)]
        col_label = "long|short"

    else:
        if has_rw and has_lt and not has_st:
            col_param = "param_long_threshold"
        elif has_rw and has_st and not has_lt:
            col_param = "param_short_threshold"
        elif len(param_cols) >= 2:
            sorted_params = sorted(param_cols)
            if row_param in sorted_params:
                sorted_params.remove(row_param)
            col_param = sorted_params[0]
        else:
            col_param = None

        if col_param:
            df["col_key"] = df[col_param].apply(format_val)
            pivot = df.pivot_table(
                index="row_key",
                columns="col_key",
                values=metric,
                aggfunc="mean"
            )
            pivot = pivot.reindex(
                sorted(pivot.index, key=lambda x: float(x)),
                axis=0
            )
            pivot = pivot[[c for c in sorted(pivot.columns, key=lambda x: float(x))]]
            col_label = col_param.replace("param_", "")
        else:
            pivot = df.groupby("row_key")[metric].mean().to_frame(metric)
            col_label = ""

    # Label axes
    pivot.index.name   = row_param.replace("param_", "")
    pivot.columns.name = col_label

    # Plot heatmap
    mask = pivot.isnull()
    plt.figure(figsize=(max(6, pivot.shape[1]*0.5), max(4, pivot.shape[0]*0.5)))
    sns.heatmap(
        pivot,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="viridis_r",
        annot_kws={"size": 8},
        cbar_kws={"label": metric}
    )
    plt.title(f"Heatmap of {metric}")
    plt.xlabel(pivot.columns.name)
    plt.ylabel(pivot.index.name)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "heatmap.png"), bbox_inches="tight")
    plt.close()
