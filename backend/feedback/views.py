from django.shortcuts import render

# Create your views here.
from .models import feedback,complain

def feed(request):
    if request.method=='POST':
        type2 = request.POST.get('feed')
        type1 = request.POST.get('com')
        name = request.POST.get('name','')
        number = request.POST.get('pno','')
        data = request.POST.get('note')
        if type2=='on':
            feed1=feedback(name=name,number=number,feed=data)
            feed1.save()
            return render(request,"success.html" ,{'message':"Response Recorded",'data':"New Responce", 'link':'/feedback/new'})
        else:
            com1=complain(name=name,number=number,complain=data)
            com1.save()
            return render(request,"success.html" ,{'message':"Response Recorded",'data':"New Responce", 'link':'/feedback/new'})
    else:
        return render(request,"feedback.html")
	#except Exception as e:
		#trace_back = traceback.format_exc()
		#message = str(e) + " " + str(trace_back)
	#	return render(request,"failure.html" ,{'message':str(e) ,'data':"Try Again", 'link':'/feedback/new'})
    