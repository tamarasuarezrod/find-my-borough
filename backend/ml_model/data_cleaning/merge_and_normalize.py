import pandas as pd

def merge_all_features(df_rent, df_crime, df_age, df_zone):
    df = pd.merge(df_rent, df_crime, on="borough", how="left")
    df = pd.merge(df, df_age, on="borough", how="left")
    df = pd.merge(df, df_zone, on="borough", how="left")

    # Normalize rent (1: cheap, 0: expensive)
    rent_min, rent_max = df["avg_rent"].min(), df["avg_rent"].max()
    df["norm_rent"] = 1 - (df["avg_rent"] - rent_min) / (rent_max - rent_min)

    # Normalize crime (1: safe, 0: dangerous)
    crime_min, crime_max = df["total_crimes"].min(), df["total_crimes"].max()
    df["norm_crime"] = 1 - (df["total_crimes"] - crime_min) / (crime_max - crime_min)

    # Normalize age (1: youthful, 0: older)
    youth_min, youth_max = df["young_ratio"].min(), df["young_ratio"].max()
    df["norm_youth"] = (df["young_ratio"] - youth_min) / (youth_max - youth_min)

    # Normalize zone (1: central, 0: far)
    zone_min, zone_max = df["zone"].min(), df["zone"].max()
    df["norm_centrality"] = 1 - (df["zone"] - zone_min) / (zone_max - zone_min)

    df = df[["borough", "norm_rent", "norm_crime", "norm_youth", "norm_centrality"]]
    
    return df
