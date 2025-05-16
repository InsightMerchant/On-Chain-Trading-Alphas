# helper/shift_helper.py
import os
def determine_shift_override(ds_filename: str, config: dict) -> int:
    fname = os.path.basename(ds_filename or "").lower()
    if fname.startswith("gn_"):
        return -70
    if "market-data" in fname or "market_data" in fname:
        key = "default_shift_override"
    else:
        key = "candle_shift_other"

    return config["data"].get(key, -60)

def determine_shift_override_for_formula(factor_map: dict, config: dict) -> int:
    file_names = [os.path.basename(path).lower() for path, _ in factor_map.values()]
    if all(f.startswith("gn_") for f in file_names):
        return -70
    if all("market-data" in f or "market_data" in f for f in file_names):
        return config["data"].get("default_shift_override", -10)
    return config["data"].get("candle_shift_other", -60)
