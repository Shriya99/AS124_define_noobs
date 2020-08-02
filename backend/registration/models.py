from django.db import models

# Create your models here.

class Register(models.Model):
    first_name = models.CharField(max_length=50)
    middle_name = models.CharField(max_length=50,default='')
    last_name = models.CharField(max_length=50)
    dob = models.CharField(max_length=50)
    height_cm = models.CharField(max_length=50)
    weight = models.CharField(max_length=50)
    gender = models.CharField(max_length=50)
    address = models.CharField(max_length=100)
    camp_loc=models.CharField(max_length=100,default='')
    aadhar1=models.CharField(max_length=4)
    aadhar2 = models.CharField(max_length=4)
    aadhar3 = models.CharField(max_length=4)
    fullaadhar=models.CharField(max_length=12,default='',unique=True)
    phone = models.CharField(max_length=13,default='')
    imagepath = models.CharField(max_length=100,default='')
    verstat = models.CharField(max_length=20,default='')
    lang_pref = models.CharField(max_length=20,default='English')

    def __str__(self):
        return self.aadhar1+self.aadhar2+self.aadhar3

class Dosage(models.Model):
    matchedaadhar = models.CharField(max_length=50,unique=True)
    Diagnosis = models.BooleanField(default = False)
    dosage_details = models.CharField(max_length=300)
    visit_status = models.BooleanField(default = False)
    dosage_date = models.DateField(blank=True,null=True,default = '1990-09-09')
    initial_bmi=models.DecimalField(max_digits=5, decimal_places=2)
    #dosage_date = models.CharField(max_length=50, default='June', null=True)
    phone_no = models.CharField(max_length=13, default='')
    loc=models.CharField(max_length=100,default='')

    def __str__(self):
        return self.matchedaadhar

# class Stat(models.Model):
#     location = models.CharField(max_length=50,unique=True)
#     month = models.CharField(max_length=2,default='')
#     year = models.CharField(max_length=4,default='')
#     present = models.BigIntegerField(default=0)
#     registered = models.BigIntegerField(default=0)
#
#     def __str__(self):
#         return self.location

class History(models.Model):
    histaadhar = models.CharField(max_length=50,unique=True)
    Diagnosis1 = models.CharField(max_length=50)
    history1 = models.CharField(max_length=300)
    history_date1 = models.DateField(blank=True, null=True)
    bmi1 = models.DecimalField(max_digits=5, decimal_places=2,null=True,blank=True)
    Diagnosis2 = models.CharField(max_length=50)
    history2 = models.CharField(max_length=300)
    history_date2 = models.DateField(blank=True, null=True)
    bmi2 = models.DecimalField(max_digits=5, decimal_places=2,null=True,blank=True)
    Diagnosis3 = models.CharField(max_length=50)
    history3 = models.CharField(max_length=300)
    history_date3 = models.DateField(blank=True, null=True)
    bmi3 = models.DecimalField(max_digits=5, decimal_places=2,null=True,blank=True)
    history_count = models.IntegerField(default=0)
    # dosage_date = models.DateField(blank=True,null=True,default = '1990-09-09')
    # #dosage_date = models.CharField(max_length=50, default='June', null=True)
    # phone_no = models.CharField(max_length=13, default='')
    # loc=models.CharField(max_length=100,default='')
    def __str__(self):
        return self.histaadhar
