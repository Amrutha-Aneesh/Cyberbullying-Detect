from django.db import models

# Create your models here.
class user(models.Model):
    name=models.CharField(max_length=255)
    email=models.EmailField(max_length=255)
    password=models.CharField(max_length=255)
    rpwd=models.CharField(max_length=255)

class Image(models.Model):
    name = models.CharField(max_length=30)
    img = models.ImageField(upload_to = 'pix')