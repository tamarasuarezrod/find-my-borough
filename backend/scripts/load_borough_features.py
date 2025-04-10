import os
import sys
import django
import csv
from django.utils.text import slugify

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

from borough.models import Borough

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
csv_path = os.path.join(base_dir, 'data', 'clean', 'borough_features.csv')

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        slug = slugify(row['borough'])
        try:
            borough = Borough.objects.get(slug=slug)

            borough.norm_rent = float(row['norm_rent'])
            borough.norm_crime = float(row['norm_safety'])
            borough.norm_youth = float(row['norm_youth'])
            borough.norm_centrality = float(row['norm_centrality'])

            borough.save()
            print(f"✅ Updated {borough.name}")
        except Borough.DoesNotExist:
            print(f"Not found: {row['borough']}")
