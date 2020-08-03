from django.db import models


# Create your models here.
class hmail(models.Model):
    email = models.EmailField(max_length=20)
    account = models.BooleanField(default = False)
