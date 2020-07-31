from django.db import models

# Create your models here.
class homevisit(models.Model):
    date=models.DateField(auto_now=False, auto_now_add=False)
    location=models.CharField(max_length=50)
    visits=models.IntegerField()

    def __str__(self):
        return self.location