# FindMyBorough

**FindMyBorough** is an AI-powered tool that helps you find the best London borough to live in, based on your personal preferences such as budget, safety, green spaces, and transport.

## What it does

FindMyBorough recommends boroughs in London using a personalized scoring model trained on synthetic user preferences and real borough-level data. Users can input how much they value factors like:

- Affordability
- Safety
- Youthful population
- Proximity to central London

The system returns a ranked list of boroughs that best match those preferences.

It also provides an elegant frontend to explore boroughs, see crowd-sourced opinions, and interact with borough profiles visually — inspired by platforms like NomadList.

## Tech Stack

- Django + Django REST Framework
- Scikit-learn (for the ML model)
- Python

## Project Structure

```
find-my-borough/
│
├── backend/                # Django project (APIs for boroughs, users, recommendations)
├── ml_model/               # Machine learning logic and pipelines
│   ├── data_cleaning/      # Data loading & cleaning scripts for each source
│   ├── training/           # Synthetic data generation and model training
│   ├── models/             # Trained PyTorch models
│   ├── recommend/          # Recommendation logic using the model
│   ├── utils.py            # Shared utilities
│   └── run_cleaning_pipeline.py
│
├── data/
│   ├── raw/                # Raw downloaded datasets
│   ├── clean/              # Cleaned, preprocessed datasets
│   └── README.md           # Data source documentation
│
├── requirements.txt
└── README.md
```

## Data Sources

Details about the datasets used in this project can be found in the [data/README.md](data/README.md).
