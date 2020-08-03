from django.contrib import admin

# Register your models here.
from .models import feedback,complain

admin.site.register(feedback)
admin.site.register(complain)