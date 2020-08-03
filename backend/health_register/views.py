from django.shortcuts import render

# Create your views here.
from .forms import NewUserForm
from django.contrib.auth import login,logout,authenticate
from .models import hmail
from django.shortcuts import redirect

def newstaff(request):
    if request.method == "POST":
        form =NewUserForm(request.POST)
        em=request.POST.get('email')
        print(em)
        mail=hmail.objects.filter(email=em)
        if mail:
            mail1=hmail.objects.get(email=em)
            if mail1.account:
                return render(request,"failure.html" ,{'message':'Account Already Exist' ,'data':"Try Again", 'link':'/newreg/new_staff/'})
            else:
                if form.is_valid():
                    mail1.account=True
                    mail1.save()
                    user = form.save()
                    username = form.cleaned_data.get('username')
                    login(request, user)
                    return redirect("main:home")

                else:
                    for msg in form.error_messages:
                        print(form.error_messages[msg])

                    return render(request = request,
                                template_name = "userreg.html",
                                context={"form":form})
        else:
            print('NO')
            return render(request,"failure.html" ,{'message':'Not-authorised Health User' ,'data':"Try Again", 'link':'/newreg/new_staff/'})
    else:
        form = NewUserForm
        return render(request = request,
                    template_name = "userreg.html",
                    context={"form":form})
