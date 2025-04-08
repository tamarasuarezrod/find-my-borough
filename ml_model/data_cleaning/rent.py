import pandas as pd
from ml_model.utils import standardize_borough_names, filter_valid_boroughs, valid_boroughs

def clean_rent_data():
    df_rent = pd.read_excel("data/raw/voa-average-rent-borough.xls", sheet_name="Raw data", header=2)
    df_all = df_rent[df_rent['Category'] == 'All categories']
    df_rent_clean = df_all.groupby('Area')['Median'].median().reset_index()
    df_rent_clean.columns = ['borough', 'avg_rent']

    df_rent_clean['borough'] = standardize_borough_names(df_rent_clean['borough'])
    df_rent_clean = filter_valid_boroughs(df_rent_clean, 'borough', valid_boroughs)

    df_rent_clean.to_csv("data/clean/clean_rent.csv", index=False)
    
    print("-- Rent data cleaned and saved to data/clean/clean_rent.csv")

    return df_rent_clean
