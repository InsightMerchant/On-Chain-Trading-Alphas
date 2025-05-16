import re

def extract_interval_from_filename(filename):
    patterns = [
        r'_(\d+)\s*(min|h|d)\b',               
        r'_(\d+)\s*(minutes?|hours?|days?)\b',  
        r'_(min|h|hour|d|day)\b'                
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename, re.IGNORECASE)
        if match:
            groups = match.groups()
            if len(groups) == 2:
                number, unit = groups
            elif len(groups) == 1:
                number = "1"
                unit = groups[0]
            else:
                continue
            unit = unit.lower()
            if unit.startswith("hour"):
                unit = "h"
            elif unit.startswith("day"):
                unit = "d"
            elif unit.startswith("min"):
                unit = "min"
            return f"{number}{unit}"
    return None

def parse_interval(interval):
    match = re.match(r"(\d+)\s*(min|h|d)", interval, re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid interval format: {interval}")
    num, unit = match.groups()
    num = int(num)
    unit = unit.lower()
    if unit == "min":
        return num
    elif unit == "h":
        return num * 60
    elif unit == "d":
        return num * 1440
    else:
        raise ValueError(f"Unsupported time unit: {unit}")

def format_interval(minutes):
    if minutes % 1440 == 0:
        return f"{minutes // 1440}d"
    elif minutes % 60 == 0:
        return f"{minutes // 60}h"
    else:
        return f"{minutes}min"

def get_valid_intervals_from_factor(factor_native, max_factor=24):
    base_minutes = parse_interval(factor_native)
    valid_intervals = []
    for mult in range(1, max_factor + 1):
        interval_minutes = base_minutes * mult
        valid_intervals.append(format_interval(interval_minutes))
    return valid_intervals

def get_common_intervals(candle_native, datasource_native):
    from utils.interval_helper import parse_interval, format_interval
    ds_minutes = parse_interval(datasource_native)
    standard_intervals = [1, 5, 10, 15, 30, 60, 120, 180, 240, 360, 1440]
    valid_intervals = [format_interval(x) for x in standard_intervals if x >= ds_minutes]
    return valid_intervals

