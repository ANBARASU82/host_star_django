from rest_framework import serializers
from .models import KeplerMission

class KeplerMissionSerializer(serializers.ModelSerializer):
    class Meta:
        model = KeplerMission
        fields = ['mass', 'metallicity', 'radius', 'temperature']
