from __future__ import unicode_literals
from django.http import HttpResponse
import traceback
#from registration import models

# Include the `fusioncharts.py` file that contains functions to embed the charts.
from django.shortcuts import render
from django.views.generic import View
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import filters
from stats_graphs.models import Stats
from django.utils.timezone import get_current_timezone
from django.utils.dateparse import parse_date
import datetime
from django.contrib.auth import login,logout,authenticate
from django.shortcuts import redirect
from django.contrib import messages
from .forms import NewUserForm
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required


def yoga(request):
    return render(request,'yoga.html')
def faq(request):
    return render(request,'faq_final.html')
def chart(request):
    return render(request,'diet.html')
@login_required
def home(request):
    try:
        if request.method=='POST':
            labels = []
            data = []
            check = True
            query = request.POST['name']
            #queryset = City.objects.order_by('-population')[:5]
            labels = [
                'January',
                'February',
                'March',
                'April',
                'May',
                'June',
                'July',
                'August',
                'September',
                'October',
                'November',
                'December'
                ]

            p = Stats.objects.filter(name=query.lower())

            #d=1
            for l in labels:
                #labels.append(city.name)

                #p = Stats.objects.filter(date.month=l)

                sum = 0
                c=0
                #d=d+1
                for per in p:
                    s = per.day.strftime('%B')
                    if  l == s:

                #        d=d+1
                        sum = sum+per.visit_count
                        c=c+1
                if c != 0:
                    data.append(sum/c)
                else:
                    data.append(0)


            return render(request, 'start.html', {
                'labels': labels,
                'data': data,
                'check':check

            })
        else:

            labels = []
            data = []
            #query = request.GET['name']
            #queryset = City.objects.order_by('-population')[:5]
            check = False
            labels = [
                'January',
                'February',
                'March',
                'April',
                'May',
                'June',
                'July',
                'August',
                'September',
                'October',
                'November',
                'December'
                ]





            p = Stats.objects.all()

            #d=1
            for l in labels:
                #labels.append(city.name)

                #p = Stats.objects.filter(date.month=l)

                sum = 0
                c=0
                #d=d+1
                for per in p:
                    s = per.day.strftime('%B')
                    if  l == s:

                        #d=d+1
                        sum = sum+per.visit_count
                        c=c+1
                if c != 0:
                    data.append(sum/c)
                else:
                    data.append(0)
                 # inserts average visit_count for that month
                #data.append(1)
            
            return render(request, 'start.html', {
                'labels': labels,
                'data': data,
                'check':check
            })
    except Exception as e:
#        trace_back = traceback.format_exc()
#        message = str(e) + " " + str(trace_back)
        return render(request,"failure.html" ,{'message':str(e) ,'data':"Try Again", 'link':'/#st-at'})





def bmi(request):
	return render(request, 'bmi.html')


def calorie(request):
	return render(request, 'calorie.html')

def bmiuser(request):
	return render(request, 'bmiuser.html')


def calorieuser(request):
	return render(request, 'calorieuser.html')


def logout_request(request):
    logout(request)
    messages.info(request, "Logged out successfully!")
    return redirect("/")

def login_request(request):
    if request.user.is_authenticated:
        return redirect('main:home')
    if request.method == 'POST':
        form = AuthenticationForm(request=request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f"You are now logged in as {username}")
                return redirect('main:home')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    form = AuthenticationForm()
    return render(request = request,
                    template_name = "login.html",
                    context={"form":form})


def homeuser(request):
    try:
        if request.method=='POST':
            labels = []
            data = []
            check = True
            query = request.POST['name']
            #queryset = City.objects.order_by('-population')[:5]
            labels = [
                'January',
                'February',
                'March',
                'April',
                'May',
                'June',
                'July',
                'August',
                'September',
                'October',
                'November',
                'December'
                ]

            p = Stats.objects.filter(name=query.lower())

            #d=1
            for l in labels:
                #labels.append(city.name)

                #p = Stats.objects.filter(date.month=l)

                sum = 0
                c=0
                #d=d+1
                for per in p:
                    s = per.day.strftime('%B')
                    if  l == s:

                #        d=d+1
                        sum = sum+per.visit_count
                        c=c+1
                if c != 0:
                    data.append(sum/c)
                else:
                    data.append(0)


            return render(request, 'startuser.html', {
                'labels': labels,
                'data': data,
                'check':check

            })
        else:

            labels = []
            data = []
            #query = request.GET['name']
            #queryset = City.objects.order_by('-population')[:5]
            check = False
            labels = [
                'January',
                'February',
                'March',
                'April',
                'May',
                'June',
                'July',
                'August',
                'September',
                'October',
                'November',
                'December'
                ]





            p = Stats.objects.all()

            #d=1
            for l in labels:
                #labels.append(city.name)

                #p = Stats.objects.filter(date.month=l)

                sum = 0
                c=0
                #d=d+1
                for per in p:
                    s = per.day.strftime('%B')
                    if  l == s:

                        #d=d+1
                        sum = sum+per.visit_count
                        c=c+1
                if c != 0:
                    data.append(sum/c)
                else:
                    data.append(0)
                 # inserts average visit_count for that month
                #data.append(1)

            return render(request, 'startuser.html', {
                'labels': labels,
                'data': data,
                'check':check
            })
    except Exception as e:
#        trace_back = traceback.format_exc()
#        message = str(e) + " " + str(trace_back)
        return render(request,"failure.html" ,{'message':str(e) ,'data':"Try Again", 'link':'/'})