from django.conf import settings
from celery import Celery
from celery.schedules import crontab
import os

os.environ['DJANGO_SETTINGS_MODULE'] = 'backend.settings'
redis_host = settings.REDIS_HOST
app = Celery('backend',
broker='redis://' + settings.REDIS_HOST + ':6379',
backend='redis://' + settings.REDIS_HOST + ':6379',
include=['registration.tasks']
)
app.conf.update(
CELERY_TASK_SERIALIZER='json',
CELERY_RESULT_SERIALIZER='json',
CELERY_TASK_RESULT_EXPIRES=3600,
CELERY_TIMEZONE='UTC',
CELERYBEAT_SCHEDULE = {
'task1': {
'name':'task1',
'task': 'registration.tasks.task1',
# 'schedule': crontab(minute=0, hour=0)
'schedule': crontab(minute='*/2')
},
# 'task2': {
# 'name':'task2',
# 'task': 'registration.tasks.task2',
# # 'schedule': crontab(minute=0, hour='*/1')
# 'schedule': crontab(minute='*/3')
# },
},
)

if __name__ == '__main__':
    app.start()

