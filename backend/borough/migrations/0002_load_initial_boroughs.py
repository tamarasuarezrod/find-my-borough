from django.db import migrations
from django.utils.text import slugify


def load_initial_boroughs(apps, schema_editor):
    Borough = apps.get_model("borough", "Borough")

    borough_names = [
        "Barking and Dagenham",
        "Barnet",
        "Bexley",
        "Brent",
        "Bromley",
        "Camden",
        "City of London",
        "Croydon",
        "Ealing",
        "Enfield",
        "Greenwich",
        "Hackney",
        "Hammersmith and Fulham",
        "Haringey",
        "Harrow",
        "Havering",
        "Hillingdon",
        "Hounslow",
        "Islington",
        "Kensington and Chelsea",
        "Kingston upon Thames",
        "Lambeth",
        "Lewisham",
        "Merton",
        "Newham",
        "Redbridge",
        "Richmond upon Thames",
        "Southwark",
        "Sutton",
        "Tower Hamlets",
        "Waltham Forest",
        "Wandsworth",
        "Westminster",
    ]

    for name in borough_names:
        slug = slugify(name)
        Borough.objects.get_or_create(slug=slug, defaults={"name": name})


class Migration(migrations.Migration):

    dependencies = [
        ("borough", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(load_initial_boroughs),
    ]
