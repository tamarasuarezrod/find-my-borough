# Generated by Django 5.2 on 2025-04-14 19:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('borough', '0002_load_initial_boroughs'),
    ]

    operations = [
        migrations.AddField(
            model_name='borough',
            name='description',
            field=models.TextField(blank=True, null=True),
        ),
    ]
