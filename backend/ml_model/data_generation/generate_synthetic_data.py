import pandas as pd
import numpy as np

def generate_synthetic_training_data():
    df_boroughs = pd.read_csv("data/clean/borough_features.csv")

    user_profiles = [
        {"user_id": 1, "budget_weight": 1.0, "safety_weight": 0.0, "youth_weight": 0.0, "centrality_weight": 0.0},
        {"user_id": 2, "budget_weight": 0.0, "safety_weight": 1.0, "youth_weight": 0.0, "centrality_weight": 0.0},
        {"user_id": 3, "budget_weight": 0.5, "safety_weight": 0.5, "youth_weight": 0.0, "centrality_weight": 0.0},
        {"user_id": 4, "budget_weight": 0.4, "safety_weight": 0.2, "youth_weight": 0.4, "centrality_weight": 0.0},
        {"user_id": 5, "budget_weight": 0.3, "safety_weight": 0.4, "youth_weight": 0.3, "centrality_weight": 0.0},
        {"user_id": 6, "budget_weight": 0.1, "safety_weight": 0.3, "youth_weight": 0.6, "centrality_weight": 0.0},
        {"user_id": 7, "budget_weight": 0.6, "safety_weight": 0.2, "youth_weight": 0.2, "centrality_weight": 0.0},
        {"user_id": 8, "budget_weight": 0.3, "safety_weight": 0.3, "youth_weight": 0.3, "centrality_weight": 0.1},
        {"user_id": 9, "budget_weight": 0.1, "safety_weight": 0.1, "youth_weight": 0.4, "centrality_weight": 0.4},
        {"user_id": 10, "budget_weight": 0.0, "safety_weight": 0.3, "youth_weight": 0.7, "centrality_weight": 0.0},
        {"user_id": 11, "budget_weight": 0.0, "safety_weight": 0.0, "youth_weight": 0.0, "centrality_weight": 1.0},
        {"user_id": 12, "budget_weight": 0.25, "safety_weight": 0.25, "youth_weight": 0.25, "centrality_weight": 0.25},
        # Outliers
        {"user_id": 13, "budget_weight": 0.0, "safety_weight": 0.0, "youth_weight": 0.0, "centrality_weight": 1.0},
        {"user_id": 14, "budget_weight": 0.0, "safety_weight": 0.0, "youth_weight": 1.0, "centrality_weight": 0.0},
        {"user_id": 15, "budget_weight": 0.0, "safety_weight": 1.0, "youth_weight": 0.0, "centrality_weight": 0.0},
        {"user_id": 16, "budget_weight": 1.0, "safety_weight": 0.0, "youth_weight": 0.0, "centrality_weight": 0.0},
        {"user_id": 17, "budget_weight": 0.0, "safety_weight": 0.5, "youth_weight": 0.5, "centrality_weight": 0.0},
        {"user_id": 18, "budget_weight": 0.0, "safety_weight": 0.0, "youth_weight": 0.5, "centrality_weight": 0.5},
        {"user_id": 19, "budget_weight": 0.5, "safety_weight": 0.0, "youth_weight": 0.0, "centrality_weight": 0.5},
        {"user_id": 20, "budget_weight": 0.25, "safety_weight": 0.25, "youth_weight": 0.0, "centrality_weight": 0.5},
    ]

    training_data = []

    for profile in user_profiles:
        uid = profile["user_id"]
        b_weight = profile["budget_weight"]
        s_weight = profile["safety_weight"]
        y_weight = profile["youth_weight"]
        c_weight = profile["centrality_weight"]

        borough_scores = []

        for _, row in df_boroughs.iterrows():
            raw_score = (
                b_weight * row["norm_rent"] +
                s_weight * row["norm_crime"] +
                y_weight * row["norm_youth"] +
                c_weight * row["norm_centrality"]
            )

            borough_scores.append((
                row["borough"],
                row["norm_rent"],
                row["norm_crime"],
                row["norm_youth"],
                row["norm_centrality"],
                raw_score
            ))

        borough_scores.sort(key=lambda x: x[5], reverse=True)
        top = borough_scores[:5]
        middle = borough_scores[5:20]
        bottom = borough_scores[-10:]

        for borough, rent, safety, youth, centrality, raw_score in top:
            training_data.append([uid, b_weight, s_weight, y_weight, c_weight, rent, safety, youth, centrality, borough, raw_score, 1.0])
        for borough, rent, safety, youth, centrality, raw_score in middle:
            training_data.append([uid, b_weight, s_weight, y_weight, c_weight, rent, safety, youth, centrality, borough, raw_score, 0.5])
        for borough, rent, safety, youth, centrality, raw_score in bottom:
            training_data.append([uid, b_weight, s_weight, y_weight, c_weight, rent, safety, youth, centrality, borough, raw_score, 0.0])

    df_synth = pd.DataFrame(training_data, columns=[
        "user_id", "budget_weight", "safety_weight", "youth_weight", "centrality_weight",
        "norm_rent", "norm_crime", "norm_youth", "norm_centrality",
        "borough", "raw_score", "score"
    ])

    df_synth.to_csv("data/clean/synthetic_training_data.csv", index=False)
    print("✅ synthetic_training_data.csv generated")

if __name__ == "__main__":
    generate_synthetic_training_data()
