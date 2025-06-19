from django.db import models

class KeplerMission(models.Model):
    mass = models.CharField(max_length=100)
    metallicity = models.CharField(max_length=100)
    radius = models.CharField(max_length=100)
    temperature = models.CharField(max_length=100)
