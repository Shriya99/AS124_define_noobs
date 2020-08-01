from django.contrib import admin

# Register your models here.
from .models import Register,Dosage,History

admin.site.register(Register)
admin.site.register(Dosage)
admin.site.register(History)