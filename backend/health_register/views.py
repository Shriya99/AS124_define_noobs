from django.shortcuts import render

# Create your views here.
from .models import hmail

def newstaff(request):
    if request.method == "POST":
        username=request.POST.get('uname')
        em=request.POST.get('id_email')
        pass1=request.POST.get('password1')
        pass2=request.POST.get('password2')
        if pass1==pass2:
            mail=hmail.objects.filter(email=em)
            if mail:
                print('yes')
                newuser = User() 
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
