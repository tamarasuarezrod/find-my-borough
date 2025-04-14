import pandas as pd
import numpy as np
import os
import random

BASE_PATH = "ml_model/data/clean"
CURRENT_SITUATION_OPTIONS = ["student", "young_professional", "professional", "other"]
STAY_DURATION_OPTIONS = ["short_term", "medium_term", "long_term"]

def encode_one_hot(value, options):
    return [1.0 if value == opt else 0.0 for opt in options]

def normalize_weights(weights):
    total = sum(weights)
    return [w / total for w in weights]

def create_user_profile(user_id, situation, duration):
    if situation == "student":
        base_weights = [0.5, 0.2, 0.3, 0.0]
    elif situation == "young_professional":
        base_weights = [0.2, 0.3, 0.2, 0.3]
    elif situation == "professional":
        base_weights = [0.1, 0.5, 0.1, 0.3]
    else:  # other
        base_weights = np.random.dirichlet(np.ones(4), size=1).flatten().tolist()

    # Add slight noise
    weights = [max(0.0, min(1.0, w + random.uniform(-0.1, 0.1))) for w in base_weights]
    weights = normalize_weights(weights)

    return {
        "user_id": user_id,
        "budget_weight": weights[0],
        "safety_weight": weights[1],
        "youth_weight": weights[2],
        "centrality_weight": weights[3],
        "current_situation": situation,
        "stay_duration": duration,
    }

def generate_synthetic_training_data(n_users=100):
    df_boroughs = pd.read_csv(os.path.join(BASE_PATH, "borough_features.csv"))
    user_profiles = []

    user_id = 1
    for _ in range(n_users):
        situation = random.choice(CURRENT_SITUATION_OPTIONS)
        duration = random.choice(STAY_DURATION_OPTIONS)
        user_profiles.append(create_user_profile(user_id, situation, duration))
        user_id += 1

    training_data = []

    for profile in user_profiles:
        uid = profile["user_id"]
        b_weight = profile["budget_weight"]
        s_weight = profile["safety_weight"]
        y_weight = profile["youth_weight"]
        c_weight = profile["centrality_weight"]
        situation = profile["current_situation"]
        duration = profile["stay_duration"]

        situation_encoding = encode_one_hot(situation, CURRENT_SITUATION_OPTIONS)
        duration_encoding = encode_one_hot(duration, STAY_DURATION_OPTIONS)

        borough_scores = []
        for _, row in df_boroughs.iterrows():
            # base weighted score
            base_score = (
                b_weight * row["norm_rent"] +
                s_weight * row["norm_crime"] +
                y_weight * row["norm_youth"] +
                c_weight * row["norm_centrality"]
            )

            bonus = 0.0

            if situation == "student":
                bonus += (1 - row["norm_rent"]) * 0.2
            elif situation == "professional":
                bonus += row["norm_crime"] * 0.2
            elif situation == "young_professional":
                bonus += (row["norm_youth"] + row["norm_centrality"]) * 0.1

            if duration == "short_term":
                bonus += row["norm_centrality"] * 0.1
            elif duration == "long_term":
                bonus += row["norm_crime"] * 0.1

            raw_score = base_score + bonus

            borough_scores.append((row["borough"], row["norm_rent"], row["norm_crime"], row["norm_youth"], row["norm_centrality"], raw_score))

        borough_scores.sort(key=lambda x: x[5], reverse=True)
        n = 10

        top = borough_scores[:n]
        bottom = borough_scores[-n:]

        for borough, rent, crime, youth, centrality, raw_score in top:
            training_data.append([
                uid, b_weight, s_weight, y_weight, c_weight,
                *situation_encoding, *duration_encoding,
                rent, crime, youth, centrality,
                borough, raw_score, 1
            ])
        for borough, rent, crime, youth, centrality, raw_score in bottom:
            training_data.append([
                uid, b_weight, s_weight, y_weight, c_weight,
                *situation_encoding, *duration_encoding,
                rent, crime, youth, centrality,
                borough, raw_score, 0
            ])

    df_synth = pd.DataFrame(training_data, columns=[
        "user_id", "budget_weight", "safety_weight", "youth_weight", "centrality_weight",
        "situation_student", "situation_young_professional", "situation_professional", "situation_other",
        "stay_short_term", "stay_medium_term", "stay_long_term",
        "norm_rent", "norm_crime", "norm_youth", "norm_centrality",
        "borough", "raw_score", "score"
    ])

    output_path = os.path.join(BASE_PATH, "synthetic_training_data.csv")
    os.makedirs(BASE_PATH, exist_ok=True)
    df_synth.to_csv(output_path, index=False)
    print("✅ synthetic_training_data.csv generated with", len(df_synth), "rows")

if __name__ == "__main__":
    generate_synthetic_training_data()
