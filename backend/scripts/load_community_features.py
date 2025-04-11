# Run with python manage.py shell < load_community_features.py

from borough.models import CommunityFeature

features = [
    ("diversity", "Diversity & inclusion"),
    ("cleanliness", "Cleanliness"),
    ("safety", "Safety"),
    ("green", "Green spaces"),
    ("vibe", "Social vibe"),
    ("transport", "Public transport"),
    ("internet", "Internet speed"),
    ("family", "Family friendly"),
    ("walkability", "Walkability"),
    ("openness", "Community openness"),
]

for fid, label in features:
    CommunityFeature.objects.get_or_create(id=fid, defaults={"label": label})

print("✅ Community features loaded")
