from django.urls import path
from . import views
urlpatterns=[
    path('new_staff/',views.newstaff,name='newstaff'),
    ]
