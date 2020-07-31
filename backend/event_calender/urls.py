from django.urls import path
from . import views
urlpatterns=[
    path('recent',views.recent_events,name='recent_events'),
#    path('add_new_event',views.add_new_event,name='add_new_event'),
    ]
