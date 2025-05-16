import pandas as pd
import re

# Global set of timestamp-like column names.
TIMESTAMP_NAMES = {
    "datetime", "date", "time", "timestamp", "time_stamp", "open_time", "start_time", "end_time"
}

def parse_flexible_dates(date_series):

    sample = date_series.dropna().iloc[0]
    sample_str = str(sample).strip()
    # If sample starts with 4 digits, assume year-first (e.g. "2021/03/12 ...")
    if re.match(r"^\d{4}", sample_str):
        return pd.to_datetime(date_series, errors="coerce", dayfirst=False)
    else:
        return pd.to_datetime(date_series, errors="coerce", dayfirst=True)

def convert_to_day_first_format(dt_series, output_format="%d/%m/%Y %H:%M:%S"):

    return dt_series.dt.strftime(output_format)

def detect_and_normalize_timestamp(df, canonical="datetime", possible_names=None, convert_year_first=False):

    df = df.loc[:, ~df.columns.duplicated()]
    
    if possible_names is None:
        possible_names = TIMESTAMP_NAMES
    
    if canonical in df.columns:
        dt_series = parse_flexible_dates(df[canonical])
        if convert_year_first:
            # Check the first non-null value of the original column.
            sample_str = str(df[canonical].dropna().iloc[0]).strip()
            if re.match(r"^\d{4}", sample_str):
                dt_series = convert_to_day_first_format(dt_series)
                df[canonical] = dt_series
                return df
        df[canonical] = dt_series
        return df

    for col in df.columns:
        if col.lower() in {name.lower() for name in possible_names}:
            df = df.rename(columns={col: canonical})
            dt_series = parse_flexible_dates(df[canonical])
            if convert_year_first:
                sample_str = str(df[canonical].dropna().iloc[0]).strip()
                if re.match(r"^\d{4}", sample_str):
                    dt_series = convert_to_day_first_format(dt_series)
            df[canonical] = dt_series
            return df

    return df

def get_canonical_timestamp_column(config, default="datetime"):
    return config.get("timestamp_column", default)
