from pathlib import Path
import csv
from django.utils.text import slugify
from borough.models import Borough

csv_path = Path.cwd() / 'ml_model' / 'data' / 'clean' / 'borough_features.csv'

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        slug = slugify(row['borough'])
        try:
            borough = Borough.objects.get(slug=slug)

            borough.norm_rent = float(row['norm_rent']) if row['norm_rent'] else None
            borough.norm_crime = float(row['norm_crime']) if row['norm_crime'] else None
            borough.norm_youth = float(row['norm_youth']) if row['norm_youth'] else None
            borough.norm_centrality = float(row['norm_centrality']) if row['norm_centrality'] else None

            borough.save()
        except Borough.DoesNotExist:
            print(f"⚠️ Not found: {row['borough']}")
