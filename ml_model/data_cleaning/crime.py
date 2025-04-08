import pandas as pd
from ml_model.utils import standardize_borough_names, filter_valid_boroughs, valid_boroughs

def clean_crime_data():
    df_crime = pd.read_csv("data/raw/MPS Monthly Crime Dashboard_BoroughSNT_TNOCrimeDatafy22-23_03.csv")
    df_crime_filtered = df_crime[df_crime['Measure'] == 'Offences']
    df_crime_clean = df_crime_filtered.groupby('Area name')['Count'].sum().reset_index()
    df_crime_clean.columns = ['borough', 'total_crimes']

    df_crime_clean['borough'] = standardize_borough_names(df_crime_clean['borough'])
    df_crime_clean = filter_valid_boroughs(df_crime_clean, 'borough', valid_boroughs)

    df_crime_clean.to_csv("data/clean/clean_crime.csv", index=False)
    
    print("-- Crime data cleaned and saved to data/clean/clean_crime.csv")

    return df_crime_clean
