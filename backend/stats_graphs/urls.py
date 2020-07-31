from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

app_name = 'main'
urlpatterns=[
    path('home/', views.home,name='home'),
    path("register/", views.register, name="register"),
    path("logout/", views.logout_request, name="logout"),
    path('', views.login_request, name="login"),
    #path('pie/', views.pie,name='pie'),
    path('bmi',views.bmi,name='bmi'),
    path('calorie',views.calorie,name='calorie')
    # path('test-api', views.get_data),
    
    ]
