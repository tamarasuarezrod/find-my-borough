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
            score = (
                b_weight * row["norm_rent"] +
                s_weight * row["norm_safety"] +
                y_weight * row["norm_youth"] +
                c_weight * row["norm_centrality"]
            )

            borough_scores.append((
                row["borough"],
                row["norm_rent"],
                row["norm_safety"],
                row["norm_youth"],
                row["norm_centrality"],
                score
            ))

        # Sort by score and label top, middle, and bottom scores
        borough_scores.sort(key=lambda x: x[5], reverse=True)
        top = borough_scores[:5]
        middle = borough_scores[5:20]
        bottom = borough_scores[-10:]

        for borough, rent, safety, youth, centrality, _ in top:
            training_data.append([uid, b_weight, s_weight, y_weight, c_weight, rent, safety, youth, centrality, borough, 1.0])
        for borough, rent, safety, youth, centrality, _ in middle:
            training_data.append([uid, b_weight, s_weight, y_weight, c_weight, rent, safety, youth, centrality, borough, 0.5])
        for borough, rent, safety, youth, centrality, _ in bottom:
            training_data.append([uid, b_weight, s_weight, y_weight, c_weight, rent, safety, youth, centrality, borough, 0.0])

    df_synth = pd.DataFrame(training_data, columns=[
        "user_id", "budget_weight", "safety_weight", "youth_weight", "centrality_weight",
        "norm_rent", "norm_safety", "norm_youth", "norm_centrality", "borough", "score"
    ])

    df_synth.to_csv("data/clean/synthetic_training_data.csv", index=False)
    print("✅ synthetic_training_data.csv generated")

if __name__ == "__main__":
    generate_synthetic_training_data()
