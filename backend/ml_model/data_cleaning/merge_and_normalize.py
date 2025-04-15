import os

import numpy as np
import pandas as pd


def normalize_column(series, invert=False, log=False):
    """
    Normalizes a pandas Series using min-max scaling.

    Args:
        series (pd.Series): The column to normalize.
        invert (bool): If True, higher original values become lower (1 - scaled).
        log (bool): If True, applies log1p transformation before normalization.

    Returns:
        pd.Series: Normalized values between 0 and 1.
    """
    data = np.log1p(series) if log else series
    min_val, max_val = data.min(), data.max()
    norm = (data - min_val) / (max_val - min_val)
    return 1 - norm if invert else norm


def merge_all_features(
    df_rent=None,
    df_crime=None,
    df_age=None,
    df_zone=None,
):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    base_path = os.path.join(BASE_DIR, "ml_model", "data", "clean")

    if df_rent is None:
        df_rent = pd.read_csv(os.path.join(base_path, "clean_rent.csv"))
    if df_crime is None:
        df_crime = pd.read_csv(os.path.join(base_path, "clean_crime.csv"))
    if df_age is None:
        df_age = pd.read_csv(os.path.join(base_path, "clean_age.csv"))
    if df_zone is None:
        df_zone = pd.read_csv(os.path.join(base_path, "borough_zones.csv"))

    df = pd.merge(df_rent, df_crime, on="borough", how="left")
    df = pd.merge(df, df_age, on="borough", how="left")
    df = pd.merge(df, df_zone, on="borough", how="left")

    df["norm_rent"] = normalize_column(df["avg_rent"], invert=True, log=True)
    df["norm_crime"] = normalize_column(df["total_crimes"], invert=True, log=True)
    df["norm_youth"] = normalize_column(df["young_ratio"])
    df["norm_centrality"] = normalize_column(df["zone"], invert=True)

    return df[["borough", "norm_rent", "norm_crime", "norm_youth", "norm_centrality"]]
