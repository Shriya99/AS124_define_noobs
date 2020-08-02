from django import forms
from .models import *


class RegisterForm(forms.ModelForm):
    class Meta:
        model = Register
        fields = ['first_name','middle_name','last_name','dob','gender','address','aadhar1','aadhar2','aadhar3','fullaadhar','phone','imagepath','lang_pref']
