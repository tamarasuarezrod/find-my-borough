import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from data_cleaning import rent, crime, age, zone, merge_and_normalize
from data_generation.generate_synthetic_data import generate_synthetic_training_data
from model_training.train_model import train_model

def run_pipeline():
    print("🚀 Starting pipeline...")

    # Step 1: Clean each data source
    df_rent_clean = rent.clean_rent_data()
    df_crime_clean = crime.clean_crime_data()
    df_age_clean = age.clean_age_data()
    df_zone_clean = zone.load_zone_data()

    # Step 2: Merge and normalize all features
    df_final = merge_and_normalize.merge_all_features(
        df_rent_clean, df_crime_clean, df_age_clean, df_zone_clean
    )
    df_final.to_csv("data/clean/borough_features.csv", index=False)
    print("✅ Cleaned and merged borough features saved.")

    # Step 3: Generate synthetic training data
    generate_synthetic_training_data()
    
    train_model()
    print("✅ Model training completed.")

if __name__ == "__main__":
    run_pipeline()
