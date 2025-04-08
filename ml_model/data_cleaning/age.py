import pandas as pd
from ml_model.utils import standardize_borough_names, filter_valid_boroughs, valid_boroughs

def clean_age_data():
    df_age = pd.read_excel("data/raw/londonboroughs.xlsx", sheet_name="Age London Boroughs")
    df_age.columns = df_age.columns.str.strip()
    df_age.rename(columns={
        "Lower tier local authorities": "borough",
        "Age (18 categories)": "age_group",
        "Observation": "population"
    }, inplace=True)

    df_age["borough"] = df_age["borough"].str.strip().str.lower()
    df_age["age_group"] = df_age["age_group"].str.strip()

    young_ages = [
        "Aged 20 to 24 years",
        "Aged 25 to 29 years",
        "Aged 30 to 34 years"
    ]

    # Total population per borough
    df_total_pop = df_age.groupby("borough")["population"].sum().reset_index()
    df_total_pop.rename(columns={"population": "total_population"}, inplace=True)

    # Young population per borough
    df_young = df_age[df_age["age_group"].isin(young_ages)]
    df_young_pop = df_young.groupby("borough")["population"].sum().reset_index()
    df_young_pop.rename(columns={"population": "young_population"}, inplace=True)

    # Merge and calculate youth ratio
    df_age_features = pd.merge(df_total_pop, df_young_pop, on="borough", how="inner")
    df_age_features["young_ratio"] = df_age_features["young_population"] / df_age_features["total_population"]

    df_age_features = df_age_features[["borough", "young_ratio"]]
    df_age_features['borough'] = standardize_borough_names(df_age_features['borough'])
    df_age_clean = filter_valid_boroughs(df_age_features, 'borough', valid_boroughs)

    df_age_clean.to_csv("data/clean/clean_age.csv", index=False)
    
    print("-- Age data cleaned and saved to data/clean/clean_age.csv")

    return df_age_clean
