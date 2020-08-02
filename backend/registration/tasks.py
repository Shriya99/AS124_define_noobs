from backend.celery import app
from datetime import datetime
import pytz
from .models import Register,Dosage
from twilio.rest import Client
from django.conf import settings
from datetime import date
from translate import Translator

@app.task(bind=True)
def task1(self):

    message_to_broadcast = ("You missed the dose your dose today")
    translation = ("You missed the dose your dose today")
    #translator= Translator(to_lang=)
    #translation = translator.translate(quote)
    client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
    pat = Dosage.objects.filter(visit_status=False,dosage_date=date.today())
    for p in pat:
        # p.visit_status = False

        reg = Register.objects.get(phone=p.phone_no)
        if reg.lang_pref != 'English':
            if reg.lang_pref == 'Hindi':
                ln = "HI"
            elif reg.lang_pref == 'Bengali':
                ln = "BN"    
            translator= Translator(to_lang=ln)
            translation = translator.translate(message_to_broadcast)
        client.messages.create(to=p.phone_no,
                               from_=settings.TWILIO_NUMBER,
                               body=translation)
    pat1 = Dosage.objects.filter(visit_status=True)
    # locations = Register.objects.values_list('camp_loc', flat=True).distinct()
    # for locate in locations:
    #     count = Dosage.objects.filter(loc=locate,visit_status=True).count()
    #     object,created = Stat.objects.update_or_create(location=locate,defaults={'present':count},)
    #     # m = Stat.objects.filter(location=locate)
    #     # m.present=count
    #     totalcount = Register.objects.filter(camp_loc=locate).count()
    #     # x = Stat.objects.filter(location=locate)
    #     # x.registered=totalcount
    #     object.registered = totalcount
    #     object.save()
    for p in pat1:
        p.visit_status=False
        p.save()

    #        serializer = PatientSerializer(data, context={'request': request}, many=True)
    # return HttpResponse("messages sent", 200)
