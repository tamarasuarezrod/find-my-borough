# Dataset Datasheet

## Motivation

- **Purpose**: This dataset was developed to support the creation of a personalized borough recommendation system for people relocating to London. The goal is to help users—particularly newcomers or international residents—navigate the city's diverse boroughs based on what matters most to them, such as affordability or lifestyle.
- **Created by**: Created by **Tamara Suárez** as part of an AI final project for the _Professional Certificate in Machine Learning and Artificial Intelligence_ from Imperial College London.
- **Funding**: This project is entirely self-funded and was developed as part of a personal portfolio to explore real-world applications of machine learning and recommender systems.

## Composition

- **Instances**: Each instance represents a synthetic user’s preferences combined with real features from a specific London borough, along with a simulated feedback score indicating how suitable that borough is for that user.

- **Size**: The dataset contains **600** training examples.  
  Each instance combines a **synthetic user preference vector** with **real borough-level features** and a **synthetic feedback score**.

  Structure of each instance: user_id, budget_weight, safety_weight, youth_weight, centrality_weight, norm_rent, norm_crime, norm_youth, norm_centrality, borough, raw_score, score.

- **Missing Data**:  
  Numerical missing values were imputed using column-wise means.  
  In particular, **City of London** was missing crime data, as this borough is not managed by the Metropolitan Police Service. This could impact the model’s predictions for that borough. Future iterations may require incorporating external crime data and ensuring normalization consistency across boroughs.

- **Confidentiality**:  
  The dataset does not include any real user data or personally identifiable information.  
  All user preferences and feedback scores were synthetically generated to simulate diverse scenarios for model training.  
  In future versions of the platform, real user interaction data will be collected to improve the model through retraining.

## Collection Process

- **How was the data acquired?**  
  All borough-level data was sourced from publicly available government repositories:

  - Rental prices from the [London Datastore](https://data.london.gov.uk/dataset/average-private-rents-borough).
  - Crime statistics from the [Metropolitan Police](https://data.london.gov.uk/dataset/mps-monthly-crime-dahboard-data).
  - Age demographics from the [Office for National Statistics (ONS)](https://www.ons.gov.uk/aboutus/transparencyandgovernance/freedomofinformationfoi/londonboroughpopulationbyageethnicityandhouseholdtype).
  - Borough zone information was manually curated based on Transport for London references.

- **Timeframe of collection:**
  - Crime data covers March 2021 to February 2025.
  - Rental and population data represent the latest available snapshots as of April 2025.
  - All datasets were accessed and cleaned in April 2025.

## Preprocessing / Cleaning / Labelling

Several preprocessing steps were required to consolidate and normalize borough-level data from multiple sources:

- **Standardization of borough names:** Raw files often used inconsistent naming formats. Borough names were lowercased, trimmed, and harmonized (e.g., replacing `&` with `and`, like in "kensington & chelsea") to ensure consistent merging across datasets.

- **Filtering valid boroughs:** A predefined list of 33 London boroughs was used to remove irrelevant or malformed entries from the source files.

- **Crime data processing:**

  - Aggregated total notifiable offences from March 2021 to February 2025.
  - Only rows marked as "Offences" were included.
  - A logarithmic transformation (`log1p`) was applied to total crime values to reduce the effect of outliers.

    - In particular, Westminster had an exceptionally high number of reported offences compared to other boroughs, which distorted min-max normalization and made relatively unsafe boroughs appear disproportionately safe.
    - Applying a log transformation reduced this skew and produced a more realistic and comparable safety metric across boroughs.

  - Values were normalized so that higher values represent safer areas.
  - The borough **City of London** had no crime data available and was retained with a `null` value placeholder.

- **Rent data processing:**

  - Filtered to only include rows labeled "All categories" (all property types).
  - Median rent values were aggregated per borough using the median across periods.
  - A logarithmic transformation (`log1p`) was also applied to rent values before normalization, to reduce skew caused by extremely high-rent boroughs.
  - Values were normalized such that `1` indicates the most affordable borough.

- **Age population processing:**

  - Youth population was computed as the proportion of residents aged 20–34.
  - Age groups were extracted from raw data and merged by borough.
  - Normalized so that `1` reflects the boroughs with the highest proportion of young adults.

- **Zone data (Centrality):**

  - A custom mapping was built to associate each borough with a travel zone (1 to 4).
  - Values were inverted and normalized, so that `1` represents the most central boroughs.

- **Synthetic label generation:**

  - Since no real user feedback was initially available, synthetic user scores were generated based on preference weights to simulate how users might rank boroughs.

- **Final dataset creation:**

  - All borough-level features were merged into a single table saved as [`borough_features.csv`](./backend/ml_model/data/clean/borough_features.csv). This represents the static characteristics of each London borough.
  - This file is later combined with synthetic user preferences and scores to create the final training dataset [`synthetic_training_data.csv`](./backend/data/clean/synthetic_training_data.csv).

- **Storage:**
  - Raw data is stored in `backend/data/raw`.
  - Cleaned and processed files are saved in `backend/data/clean`.

## Uses

- **Primary use**: To train and evaluate a borough recommendation model.
- **Secondary uses**:
  - Could support research on urban preferences and quality of life modeling.
  - May be extended with real user data to improve personalization.
- **Risks / Considerations**:
  - Since synthetic preferences were used, results should not be interpreted as reflecting real user behavior.
  - Comparisons between boroughs are only meaningful within the scope of normalized features.

## Distribution

- The dataset is currently stored in a **private GitHub repository**.

## Maintenance

- Maintained by the project creator. Updates may be made if official sources update their datasets or if user feedback is collected in production.
