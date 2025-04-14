import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd

from ml_model.models.load_latest_model import load_latest_model

model = load_latest_model()

# One-hot encoding helpers
CURRENT_SITUATION_OPTIONS = ["student", "young_professional", "professional", "other"]
STAY_DURATION_OPTIONS = ["short_term", "medium_term", "long_term"]

def encode_one_hot(value, options):
    return [1.0 if value == opt else 0.0 for opt in options]

# Prediction function
def predict_score(user_vector, borough_vector):
    with torch.no_grad():
        x = torch.tensor([user_vector + borough_vector], dtype=torch.float32)
        return model(x).item()

# Load borough features
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "data", "clean", "borough_features.csv"))

def recommend_top_n_boroughs(profile, top_n=5):
    user_vector = [
        profile["budget_weight"],
        profile["safety_weight"],
        profile["youth_weight"],
        profile["centrality_weight"],
        *encode_one_hot(profile["current_situation"], CURRENT_SITUATION_OPTIONS),
        *encode_one_hot(profile["stay_duration"], STAY_DURATION_OPTIONS),
    ]

    recommendations = []
    for _, row in df.iterrows():
        borough_vector = [
            row["norm_rent"],
            row["norm_crime"],
            row["norm_youth"],
            row["norm_centrality"],
        ]

        score = predict_score(user_vector, borough_vector)

        recommendations.append({
            "borough": row["borough"],
            "score": score,
            "norm_rent": row["norm_rent"],
            "norm_crime": row["norm_crime"],
            "norm_youth": row["norm_youth"],
            "norm_centrality": row["norm_centrality"],
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:top_n]

# Example usage
if __name__ == "__main__":
    test_profiles = [
        {
            "label": "🧑‍🎓 Budget-focused student (short stay)",
            "budget_weight": 1.0, "safety_weight": 0.2, "youth_weight": 0.3, "centrality_weight": 0.2,
            "current_situation": "student", "stay_duration": "short_term"
        },
        {
            "label": "👮 Safety-focused professional (long stay)",
            "budget_weight": 0.2, "safety_weight": 1.0, "youth_weight": 0.1, "centrality_weight": 0.2,
            "current_situation": "professional", "stay_duration": "long_term"
        },
        {
            "label": "🌇 Young urban (medium stay)",
            "budget_weight": 0.0, "safety_weight": 0.2, "youth_weight": 0.6, "centrality_weight": 0.2,
            "current_situation": "young_professional", "stay_duration": "medium_term"
        },
        {
            "label": "🧘 Balanced user",
            "budget_weight": 0.3, "safety_weight": 0.3, "youth_weight": 0.2, "centrality_weight": 0.2,
            "current_situation": "other", "stay_duration": "medium_term"
        },
        {
            "label": "🧑‍💼 High-end young pro (long stay)",
            "budget_weight": 0.0, "safety_weight": 0.2, "youth_weight": 0.7, "centrality_weight": 0.3,
            "current_situation": "young_professional", "stay_duration": "long_term"
        },
    ]

    for profile in test_profiles:
        print(f"\nTop boroughs for {profile['label']}:")
        top = recommend_top_n_boroughs(profile)
        for rec in top:
            print(f"{rec['borough']:25s} | score={rec['score']:.4f} | rent={rec['norm_rent']:.2f} | safety={rec['norm_crime']:.2f} | youth={rec['norm_youth']:.2f} | centrality={rec['norm_centrality']:.2f}")
