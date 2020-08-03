from django.shortcuts import render

# Create your views here.
from .models import feedback,complain

def feed(request):
   try:
		if request.method == 'POST':
			type1 = request.POST.get('feed')
            type2 = request.POST.get('com')
			name = request.POST.get('name', '')
			number = request.POST.get('pnum', '')
			data = request.POST.get('note')
			if(type2=='on'):
				comp1=complain(name=name,number=number,complain=data)
                comp1.save()
			else:
				feed1=feedback(name=name,number=number,feed=data)
                feed1.save()
		    return render(request,"success.html" ,{'message':"Responce Recorded",'data':"New Responce", 'link':'/feedback/new'})
        else:
            return render(request,"feeback.html")
	except Exception as e:
		#trace_back = traceback.format_exc()
		#message = str(e) + " " + str(trace_back)
		return render(request,"failure.html" ,{'message':str(e) ,'data':"Try Again", 'link':'/feedback/new'})
    