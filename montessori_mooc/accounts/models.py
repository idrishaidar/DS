from django.contrib.auth.models import AbstractUser
from django.db import models


CHILD_AGE_CHOICES = [
    ('0-3', '0–3 years'),
    ('3-6', '3–6 years'),
    ('6-12', '6–12 years'),
    ('no_child', 'No children / Self-learning'),
]


class CustomUser(AbstractUser):
    username = None
    email = models.EmailField(unique=True)
    full_name = models.CharField(max_length=200, blank=True)
    child_age_range = models.CharField(max_length=20, choices=CHILD_AGE_CHOICES, blank=True)
    bio = models.TextField(blank=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = []

    def __str__(self):
        return self.email
