# define_noobs-AS124-PoshanSaathi
## Problem Statement : Promoting holistic nutrition among women and children through/with the help of IT for Poshan Abhiyaan.(AS124)

### SIH 2020 Grand Finale Repository for P.S. No. AS124 solution , Team - #define_noobs


#### Steps to Run :
- Clone this project. Remember to create a django project to get your own secret key in settings.py file. Copy it to this settings.py as SECRET_KEY ='your-generated-secret-key'
- Install required packages
- Set up own database and add the credentials in settings.py
- Run command python manage.py makemigrations
- Run command python manage.py migrate
- Run command python manage.py createsuperuser
- Run command python manage.py runserver
###### Run command pip install translate to enable multilingual query portal

#### Features

- Health Worker Registration and Login
- New Patient Registration
- Biometric Attendance
- Dosage Update
- Automatic SMS Reminder( in Bengali/Hindi/English as preferred by patient)
- Printable Patient Data
- Locationwise Statistics
- Events Calendar
- BMI and Calorie Calculator
- Home Visit Register
- Whatsapp Query Portal( interaction supported in Hindi/Bengali/English as preferred by user)
- Offline Support
- Yoga & Exercise Card
- Email filter so that only health workes can access it
- Feedback and complaints form 

#### Tech Stack Used

- Django
- MySQL
- Celery
- Twilio
- Redis and Ngrok Server
- HTML
- CSS
- Bootstrap
