from django.urls import path

from . import views

urlpatterns = [
    path('homereg/',views.homereg,name='homereg')
]