import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")

import django

django.setup()

from django.utils.text import slugify

from borough.models import Borough

valid_boroughs = [
    "barking and dagenham",
    "barnet",
    "bexley",
    "brent",
    "bromley",
    "camden",
    "city of london",
    "croydon",
    "ealing",
    "enfield",
    "greenwich",
    "hackney",
    "hammersmith and fulham",
    "haringey",
    "harrow",
    "havering",
    "hillingdon",
    "hounslow",
    "islington",
    "kensington and chelsea",
    "kingston upon thames",
    "lambeth",
    "lewisham",
    "merton",
    "newham",
    "redbridge",
    "richmond upon thames",
    "southwark",
    "sutton",
    "tower hamlets",
    "waltham forest",
    "wandsworth",
    "westminster",
]

for name in valid_boroughs:
    slug = slugify(name)
    borough, created = Borough.objects.get_or_create(
        name=name, slug=slug, defaults={"image_url": ""}
    )
    if created:
        print(f"✅ Created: {name}")
    else:
        print(f"Already exists: {name}")
