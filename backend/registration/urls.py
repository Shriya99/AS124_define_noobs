from django.urls import path
from . import views
urlpatterns=[
    path('updated/',views.dosage_updated,name ="dosage-updated"),
    path('new-patient/', views.patient_create, name="patient-create"),
    path('attendance/', views.give_attendance, name="give_attendance"),
#    path('reg-success/', views.reg_success, name = 'reg_success'),
  #  path('broadcast_sms/',views.broadcast_sms,name='broadcast_sms'),
    path('patient-data/',views.get_patient_data,name='get_patient_data'),
    path('getdata/',views.searchdata,name='searchdata'),
    path('portal',views.whatsApp_portal,name='whatsApp_portal')
]
