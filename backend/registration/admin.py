from django.contrib import admin

# Register your models here.
from .models import Register,Dosage

admin.site.register(Register)
admin.site.register(Dosage)
# admin.site.register(Stat)