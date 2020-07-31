from .models import *
from django import forms

class EventForm(forms.ModelForm):

    class Meta:
        model = Event
        fields = ('day','start_time','end_time','notes')
