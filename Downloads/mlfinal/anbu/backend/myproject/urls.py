from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('api/', include('myapp.urls')),  # Include app URLs
]
