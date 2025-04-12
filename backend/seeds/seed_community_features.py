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

created, updated = 0, 0

for fid, label in features:
    obj, is_created = CommunityFeature.objects.get_or_create(id=fid, defaults={"label": label})
    if is_created:
        created += 1
    else:
        if obj.label != label:
            obj.label = label
            obj.save()
            updated += 1
