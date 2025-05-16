import os
import glob
import pandas as pd
from utils.timestamp_utils import detect_and_normalize_timestamp, TIMESTAMP_NAMES
from utils.interval_helper import extract_interval_from_filename, parse_interval

class DataLoader:
    def __init__(
        self,
        candle_path: str,
        datasource_path: str,
        config: dict,
        cache_path: str = None,
        debug: bool = False
    ):
        self.candle_path = candle_path
        self.datasource_path = datasource_path
        self.config = config
        self.debug = debug
        self.cache_path = cache_path or os.path.join(self.candle_path, 'cache')
        os.makedirs(self.cache_path, exist_ok=True)
        resample_cfg = os.path.join(os.getcwd(), 'config', 'resample.csv')
        try:
            df_res = pd.read_csv(resample_cfg)
            self.resample_map = dict(zip(df_res['File_name'], df_res['Resample_method_1']))
            self._debug(f"Loaded resample map for {len(self.resample_map)} files.")
        except Exception:
            self.resample_map = {}
            self._debug("No resample config found; defaulting all to 'last'.")

    def _debug(self, msg: str):
        if self.debug:
            print(f"[DataLoader] {msg}")

    def _find_candle_file(self, symbol: str, base_interval: str) -> str:
        patterns = [
            os.path.join(self.candle_path, f"{symbol}_*{base_interval}*.parquet"),
            os.path.join(self.candle_path, f"{symbol}_*{base_interval}*.csv")
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
        raise FileNotFoundError(
            f"No candle file for symbol={symbol}, interval={base_interval} in {self.candle_path}"
        )

    def _load_raw_candle_data(self, symbol: str, base_interval: str) -> pd.DataFrame:
        file_path = self._find_candle_file(symbol, base_interval)
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".parquet":
            df = pd.read_parquet(file_path, engine="pyarrow")
            self._debug(f"Loaded Parquet: {file_path}")
        else:
            df = pd.read_csv(file_path)
            self._debug(f"Loaded CSV: {file_path}")

        # Normalize timestamp
        df = detect_and_normalize_timestamp(df, canonical="datetime")
        if "datetime" not in df.columns:
            raise ValueError("Missing 'datetime' column after normalization.")

        # Ensure OHLCV numeric
        df = df.set_index("datetime")
        for col in ["close"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Drop any rows with NaNs in OHLCV
        df = df.dropna(subset=[c for c in ["close"] if c in df.columns])
        return df

    def _apply_shift(self, df: pd.DataFrame, shift_value: int) -> pd.DataFrame:
        # Shift only OHLCV, leave timestamp index intact
        cols = [c for c in ["close"] if c in df.columns]
        shifted = df[cols].shift(shift_value)
        df[cols] = shifted
        self._debug(f"Shifted close by {shift_value} periods")
        # Drop rows that become fully NaN after shift
        df = df.dropna(how="all", subset=cols)
        return df

    def _resample_candle_data(self, df: pd.DataFrame, resample_interval: str) -> pd.DataFrame:
        agg = {}
        if "close" in df.columns:
            agg["close"] = "last"

        resampled = df.resample(resample_interval).agg(agg)
        self._debug(f"Resampled to {resample_interval}, {len(resampled)} rows")
        return resampled

    def _find_candle_file(self, symbol: str, base_interval: str) -> str:
        sym = symbol.lower()
        iv  = base_interval.lower()

        # 1) try exact-interval match, case-insensitive
        for ext in ("parquet", "csv"):
            pattern = os.path.join(self.candle_path, f"{sym}_*{iv}*.{ext}")
            hits = glob.glob(pattern)
            if hits:
                return hits[0]

        # 2) no exact match — look for ANY symbol_*.csv|.parquet
        for fname in os.listdir(self.candle_path):
            lf = fname.lower()
            if lf.startswith(sym + "_") and lf.endswith((".csv", ".parquet")):
                return os.path.join(self.candle_path, fname)

        raise FileNotFoundError(f"No candle file for symbol={symbol}, interval={base_interval}")

    def load_candle_data(
        self,
        symbol: str,
        base_interval: str = "1m",
        resample_interval: str = "1m",
        shift_override: int = None,
        use_disk_cache: bool = True
    ) -> pd.DataFrame:

        # 1) determine shift
        shift_val = (
            int(shift_override)
            if shift_override is not None
            else int(self.config["data"].get("default_shift_override", -60))
        )
        print(f"[DataLoader] → shift_override = {shift_val}")

        # 2) cache path
        cache_file = os.path.join(
            self.cache_path,
            f"{symbol}_{resample_interval}_{shift_val}.parquet"
        )

        # 3) try loading the cache
        if use_disk_cache and os.path.exists(cache_file):
            df = pd.read_parquet(cache_file, engine="pyarrow")
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")
            print(f"[DataLoader] Loaded from cache: {cache_file}")
            return df

        # 4) load raw data, falling back to 1m if needed
        raw_iv = base_interval
        try:
            raw_df = self._load_raw_candle_data(symbol, base_interval)
            print(f"[DataLoader] Loaded raw candle at {base_interval}")
        except FileNotFoundError:
            raw_iv = "1m"
            raw_df = self._load_raw_candle_data(symbol, raw_iv)
            print(f"[DataLoader] No {base_interval} file; fell back to {raw_iv}")

        # 5) apply shift
        print(f"[DataLoader] Applying shift of {shift_val} periods")
        df = self._apply_shift(raw_df, shift_val)

        # 6) always resample to the target interval
        print(f"[DataLoader] Resampling from {raw_iv} → {resample_interval}")
        df = self._resample_candle_data(df, resample_interval)

        # 7) cache & return
        df = df.reset_index()  
        if use_disk_cache:
            df.to_parquet(cache_file, index=False, engine="pyarrow")
            print(f"[DataLoader] Saved cache: {cache_file}")

        df = df.set_index("datetime")
        return df
    
    def load_datasource_data(
        self,
        filename: str,
        resample_interval: str = "1m",
        factor_column: str = None
    ) -> tuple[pd.DataFrame, dict]:

        # announce which file we’re loading
        self._debug(f"Loaded DS CSV: {filename}")

        # 1) read raw file
        path = os.path.join(self.datasource_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Datasource not found: {path}")

        ext = os.path.splitext(filename)[1].lower()
        if ext == ".parquet":
            df = pd.read_parquet(path, engine="pyarrow")
            self._debug(f"Loaded DS Parquet: {filename}")
        else:
            df = pd.read_csv(path)

        # 2) normalize timestamp
        df = detect_and_normalize_timestamp(df, canonical="datetime")
        if "datetime" not in df.columns:
            raise ValueError("No 'datetime' after normalization in DS file.")
        df = df.set_index("datetime")

        # 3) isolate only the requested factor column (if present)
        if factor_column and factor_column in df.columns:
            df_to_res = df[[factor_column]]
        else:
            df_to_res = df

        # 4) figure out native interval & default method
        native_iv = extract_interval_from_filename(filename)
        method = self.resample_map.get(filename, 'last')
        if factor_column and 'volume' in factor_column.lower():
            method = 'sum'

        # ─── direct-pass if already same interval ─────────────────────────
        if native_iv and native_iv.lower() == resample_interval.lower():
            method = 'direct'
            actual_interval = 'original'

            self._debug(f"column: '{factor_column or 'auto-detected'}'")
            self._debug(f"resample_method: '{method}'")
            self._debug(f"resample_interval: '{actual_interval}'")

            df_ready = df_to_res.reset_index()

            # pick the one non-timestamp column
            data_cols = [c for c in df_ready.columns if c.lower() not in TIMESTAMP_NAMES]
            if not data_cols:
                raise ValueError(f"No data columns found in {filename}")
            col_name = data_cols[0]

            meta = {
                "file_name":         filename,
                "column":            col_name,
                "resample_method":   method,
                "resample_interval": actual_interval
            }
            return df_ready, meta

        # ─── otherwise resample to target interval ────────────────────────
        actual_interval = resample_interval
        try:
            out = df_to_res.resample(resample_interval).agg(method)
        except Exception as e:
            self._debug(f"Agg '{method}' failed: {e}; falling back to 'last'.")
            out = df_to_res.resample(resample_interval).last()

        df_ready = out.reset_index()

        data_cols = [c for c in df_ready.columns if c.lower() not in TIMESTAMP_NAMES]
        if not data_cols:
            raise ValueError(f"No data columns found in {filename}")
        col_name = data_cols[0]

        self._debug(f"column: '{col_name}'")
        self._debug(f"resample_method: '{method}'")
        self._debug(f"resample_interval: '{actual_interval}'")

        meta = {
            "file_name":         filename,
            "column":            col_name,
            "resample_method":   method,
            "resample_interval": actual_interval
        }
        return df_ready, meta

