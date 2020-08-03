from django.db import models


# Create your models here.
class feedback(models.Model):
    name=models.CharField(max_length=30)
    number=models.CharField(max_length=13)
    feed=models.CharField(max_length=100)

class complain(models.Model):
    name=models.CharField(max_length=30)
    number=models.CharField(max_length=13)
    complain=models.CharField(max_length=100)
    
