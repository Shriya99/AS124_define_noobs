from __future__ import unicode_literals
from django.db import models

# Create your models here.
import datetime
class Stats(models.Model):
    name = models.CharField(max_length=50)
    visit_count = models.IntegerField(default=0)
    day = models.DateField(blank=True,null=True)




    def mon(self):
        s= self.day.strftime('%B')
        return s

    def __str__(self):
        return self.name
