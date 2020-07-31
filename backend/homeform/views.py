from django.http import HttpResponse
from django.shortcuts import render
from .models import homevisit
from django.contrib.auth.decorators import login_required
import traceback
# Create your views here.

@login_required
def homereg(request):
    try:
        if request.method=='POST':
            date=request.POST['date']
            location=request.POST['location']
            number=request.POST['number']
            entry=homevisit(date=date,location=location,visits=number)
            entry.save()
            return render(request,"success.html" ,{'message':"Successfully Recorded",'data':"New Register", 'link':'/homeform/homereg'})
        else:
            return render(request,'homeform/register.html')
    except Exception as e:
 #       trace_back = traceback.format_exc()
 #       message = str(e) + " " + str(trace_back)
 #       return HttpResponse(message)
        return render(request,"faliure.html" ,{'message':"Failed to Record",'data':"New Register", 'link':'/homeform/homereg'}) 
