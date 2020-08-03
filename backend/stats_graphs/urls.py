from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

app_name = 'main'
urlpatterns=[
    path('', views.homeuser,name='homeuser'),
    path('yoga/', views.yoga,name='yoga'),
    path('calorieuser/',views.calorieuser,name="calorieuser"),
    path('bmiuser/',views.bmiuser,name="bmiuser"),
    path('home/', views.home,name='home'),
    path('faq/', views.faq,name='faq'),
    path('chart/', views.chart,name='chart'),
    #path("register/", views.register, name="register"),
    path("logout/", views.logout_request, name="logout"),
    path('login/', views.login_request, name="login"),
    #path('pie/', views.pie,name='pie'),
    path('bmi',views.bmi,name='bmi'),
    path('calorie',views.calorie,name='calorie')
    # path('test-api', views.get_data),
    
    ]
