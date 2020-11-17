from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings


class User(AbstractUser):
    GENDER_CHOICES = [(0, 'Male'), (1, 'Female')]
    gender = models.IntegerField(
        default=True, choices=GENDER_CHOICES, blank=True)
    age = models.IntegerField(null=True, blank=True)
    region = models.CharField(max_length=100, blank=True)
    mbti = models.CharField(max_length=10, blank=True)
    friends = models.ManyToManyField('self')


class FriendRequest(models.Model):
    sender = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='sending')
    receiver = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='receiving')
