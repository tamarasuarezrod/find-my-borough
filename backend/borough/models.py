from django.db import models
from django.contrib.auth import get_user_model
from django.core.validators import MinValueValidator, MaxValueValidator
from cloudinary_storage.storage import MediaCloudinaryStorage

User = get_user_model()

class Borough(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    image = models.ImageField(
        upload_to='boroughs/', 
        storage=MediaCloudinaryStorage(),
        blank=True, 
        null=True,
        max_length=300
    )

    norm_rent = models.FloatField(null=True, blank=True)
    norm_crime = models.FloatField(null=True, blank=True)
    norm_youth = models.FloatField(null=True, blank=True)
    norm_centrality = models.FloatField(null=True, blank=True)

    def __str__(self):
        return self.name

class CommunityFeature(models.Model):
    id = models.CharField(primary_key=True, max_length=50)
    label = models.CharField(max_length=100)
    description = models.TextField(blank=True)

    def __str__(self):
        return self.label

class CommunityRating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    borough = models.ForeignKey(Borough, on_delete=models.CASCADE)
    feature = models.ForeignKey(CommunityFeature, on_delete=models.CASCADE)
    score = models.IntegerField(validators=[MinValueValidator(0), MaxValueValidator(5)])
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'borough', 'feature')
