from django.db import models

class Borough(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    image = models.ImageField(upload_to='boroughs/', blank=True, null=True)


    norm_rent = models.FloatField(null=True, blank=True)
    norm_crime = models.FloatField(null=True, blank=True)
    norm_youth = models.FloatField(null=True, blank=True)
    norm_centrality = models.FloatField(null=True, blank=True)

    def __str__(self):
        return self.name
