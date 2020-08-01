from django.urls import path
from . import views
urlpatterns=[
    path('user-data/', views.get_data_with_biometric, name="user-data"),
]