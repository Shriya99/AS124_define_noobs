

# Create your views here.
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.contrib.auth.decorators import login_required

from .forms import *
from django.shortcuts import render, redirect
from .models import Event
import traceback
from django.http import HttpResponse
#from datetime import date
from django.utils.timezone import get_current_timezone
from django.utils.dateparse import parse_date
import datetime


# Create your views here.
def recent_events(request):
    try:
        dt = datetime.date.today()

        dates = Event.objects.order_by('day')
        return render(request,'index.html',{'dates':dates , 'dt':dt.strftime('%m')})
    except Exception as e:
#        trace_back = traceback.format_exc()
#        message = str(e) + " " + str(trace_back)
        return render(request,"failure.html" ,{'message':str(e) ,'data':"Retry", 'link':'/event/recent/'})



#def add_new_event(request):
#    try:
#        if request.method == 'POST':
#            form = EventForm(request.POST)
            #print(dir(form))
#            if form.is_valid():
#
#                form.save()

#                return redirect('recent_events')
#        else:
#            form = EventForm()
#        return render(request, 'add_new.html', {'form': form})
#    except Exception as e:
#        trace_back = traceback.format_exc()
#        message = str(e) + " " + str(trace_back)
#        return HttpResponse(message)
