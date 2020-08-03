import zlib

from django.shortcuts import render
import cv2
import traceback
import os
import numpy as np
import pytesseract
from django.http import JsonResponse
from django.http import HttpResponse
from django.shortcuts import render, redirect
import datetime
import base64
from random import seed
from random import randint
from django.conf import settings
from django.http import HttpResponse
from twilio.rest import Client
from rest_framework import parsers
from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework.exceptions import ParseError
from rest_framework.parsers import FileUploadParser
from .models import Register,Dosage,History
from event_calender.models import *
from twilio.twiml.messaging_response import MessagingResponse
from django.views.decorators.csrf import csrf_exempt
from stats_graphs.models import Stats
from django.contrib.auth.decorators import login_required
from django.utils.timezone import get_current_timezone
from django.utils.dateparse import parse_date
#import datetime
#from translate import Translator
from translate import Translator #for multilingual portal

@login_required
def patient_create(request):
	try:
		if request.method == "POST":
			first_name = request.POST.get('first_name', '')
			middle_name = request.POST.get('middle_name', '')
			last_name = request.POST.get('last_name', '')
			dob = request.POST.get('dob', '')
			gender = request.POST.get('gender', '')
			#age=request.POST.get('age','')
			height = request.POST.get('hight')
			weight = request.POST.get('weight')
			address = request.POST.get('address', '')
			camp_loc=request.POST.get('camp_loc','')
			aadhar1 = request.POST.get('aadhar1', '')
			aadhar2 = request.POST.get('aadhar2', '')
			aadhar3 = request.POST.get('aadhar3', '')
			phone = request.POST.get('phone', '')

			imagebase64 = request.POST.get('imurl2','')
			imagebase64 += "=" * ((4 - len(imagebase64) % 4) % 4)
			#print("Image base\n"+imagebase64)
			aadharbase64 = request.POST.get('imurl1','')
			aadharbase64 += "=" * ((4 - len(aadharbase64) % 4) % 4)
			fullaadhar = aadhar1 + aadhar2 + aadhar3
			with open('./faces/faceimage'+fullaadhar+'.jpg', "wb") as fh:
				fh.write(base64.b64decode(imagebase64))
			with open('aadharimage'+fullaadhar+'.jpg','wb') as fh1:
				fh1.write(base64.b64decode(aadharbase64))
			#print("Aadhar base\n"+aadharbase64)
			# files = request.FILES  # multivalued dict
			imagepath = 'faces/faceimage'+fullaadhar+'.jpg'
			lang_pref = request.POST.get('language', '')
			# image = files.get("image")
			# aadharimage = files.get("aadharimage")
			# print(aadharimage)
			# with open('aadharimageuploaded'+fullaadhar+'.jpg', 'wb+') as destination:
			# 	for chunk in aadharimage.chunks():
			# 		destination.write(chunk)
			weight = int(weight,base=10)
			height = int(height, base=10)
			print(weight)
			print(height)
			initial_bmi = (weight)/((height/100)**2)
			print(initial_bmi)
			resultstring = ocr('aadharimage'+fullaadhar+'.jpg',fullaadhar,request)
			if resultstring=='Verified':
				ver = "Verified"
				register = Register(first_name=first_name,middle_name=middle_name,last_name=last_name,dob=dob,height_cm=height,weight=weight,gender=gender,address=address,camp_loc=camp_loc,aadhar1=aadhar1,aadhar2=aadhar2,aadhar3=aadhar3,fullaadhar=fullaadhar,phone=phone,imagepath=imagepath,verstat=ver,lang_pref=lang_pref)
				register.save()
			else:
				ver ="Verification Pending"
				register = Register(first_name=first_name,middle_name=middle_name,last_name=last_name,dob=dob,gender=gender,height_cm=height,weight=weight,address=address,camp_loc=camp_loc,aadhar1=aadhar1,aadhar2=aadhar2,aadhar3=aadhar3,fullaadhar=fullaadhar,phone=phone,imagepath=imagepath,verstat=ver,lang_pref=lang_pref)
				register.save()
			dosage_details=""
			visit_status=False
			check=False
			#initial_bmi = round(initial_bmi,2)
			dosage = Dosage(matchedaadhar=fullaadhar,Diagnosis=check,dosage_details=dosage_details,visit_status=visit_status,dosage_date=None,initial_bmi=initial_bmi,phone_no=phone,loc=camp_loc)
			dosage.save()

			history = History(histaadhar=fullaadhar,Diagnosis1=False,history1=dosage_details,history_date1=None,bmi1=0,Diagnosis2=False,history2=dosage_details,history_date2=None,bmi2=0,Diagnosis3=False,history3=dosage_details,history_date3=None,bmi3=0,history_count=0)
			history.save()
			# os.remove('faceimage' + fullaadhar + '.jpg')
			os.remove('aadharimage'+fullaadhar+'.jpg')
			# else:
			# 	# os.remove('faceimage' + fullaadhar + '.jpg')
			# 	os.remove('aadharimage' + fullaadhar + '.jpg')
			# 	return HttpResponse('Failure'+resultstring)
			return render(request,"success.html" ,{'message':"Registration Successful",'data':"New Registration", 'link':'/registration/new-patient/'})
		return render(request, 'registration/registration.html')
	except Exception as e:
		#trace_back = traceback.format_exc()
		#message = str(e) + " " + str(trace_back)
		return render(request,"failure.html" ,{'message':str(e) ,'data':"Try Again", 'link':'/registration/new-patient/'})


def ocr(aadhar_image_name,fullaadhar,request):
	pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' #TODO:Change link
	new_name = cv2.imread(aadhar_image_name)
	img = cv2.resize(new_name, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
	img = cv2.GaussianBlur(img, (5, 5), 0)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	kernel = np.ones((1, 1), np.uint8)
	img = cv2.dilate(img, kernel, iterations=1)
	img = cv2.erode(img, kernel, iterations=1)
	save_path = os.path.dirname(os.path.abspath(aadhar_image_name)) + '\\' + 'processed'+fullaadhar+'.jpg'
	cv2.imwrite(save_path,img)
	# result = pytesseract.image_to_string(img, lang="eng")
	result = pytesseract.image_to_string(img, lang='eng', config='outputbase digits')
	custom_config = r'-c --psm 6 --oem 3'
	result1 = pytesseract.image_to_string(img, lang='eng+ben')
	print("Result only numbers:\n" + result)
	print("All results\n" + result1)
	request_arr=[request.POST.get('aadhar1',''),request.POST.get('aadhar2',''),request.POST.get('aadhar3','')]
	no_match =''
	for x in request_arr:
		if x.lower() in result1.lower():
			continue
		else:
			no_match = x
			break
	if no_match=='':
		return("Verified")
	else:
		return(no_match+'is not matching')

@login_required
def give_attendance(request):
	if(request.method=='POST'):
		attendbase64 = request.POST.get('imurlat', '')
		seed(1)
		value = randint(0, 999999999999)
		imagename = 'facetesting'+str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))+str(value)+'.jpg'
		with open('./facematch/'+imagename, "wb") as fh:
			fh.write(base64.b64decode(attendbase64))

		# This module takes images  stored in diskand performs face recognition
		test_img = cv2.imread('facematch/'+imagename)  # test_img path
		faces_detected, gray_img = faceDetection(test_img)
		print("faces_detected:", faces_detected)

		# Comment belows lines when running this program second time.Since it saves training.yml file in directory
		faces, faceID = labels_for_training_data('faces')
		face_recognizer = train_classifier(faces, faceID)
		face_recognizer.write('trainingData.yml')

		# Uncomment below line for subsequent runs
		# face_recognizer=cv2.face.LBPHFaceRecognizer_create()
		# face_recognizer.read('trainingData.yml')#use this to load training data for subsequent runs

		#name = {0: "Shriya", 1: "Sagnik"}  # creating dictionary containing names for each label
		try:
			for face in faces_detected:
				(x, y, w, h) = face
				roi_gray = gray_img[y:y + h, x:x + h]
				label, confidence = face_recognizer.predict(roi_gray)  # predicting the label of given image
				print("confidence:", confidence)
				print("label:", label)
				draw_rect(test_img, face)
				predicted_name = label
				# if (confidence > 37):  # If confidence more than 37 then don't print predicted face text on screen
				# 	continue
				# put_text(test_img, predicted_name, x, y)
				matchedaadhar = Register.objects.get(fullaadhar__contains=str(label)).fullaadhar
				os.remove('facematch/'+imagename)
				return render(request, 'registration/dosage.html',{'matchedaadhar':matchedaadhar})
		except Exception as e:
			#trace_back = traceback.format_exc()
			#message = str(e) + " " + str(trace_back)
			return render(request,"failure.html" ,{'message':str(e) ,' data':"Try Again", 'link':'/registration/attendance/'})

		# resized_img = cv2.resize(test_img, (1000, 1000))
		# cv2.imshow("face dtecetion tutorial", resized_img)
		# cv2.waitKey(0)  # Waits indefinitely until a key is pressed
		# cv2.destroyAllWindows()

	return render(request, 'registration/attend.html')


def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  # convert color image to grayscale
    face_haar_cascade = cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')  # Load haar classifier
    faces = face_haar_cascade.detectMultiScale(gray_img, scaleFactor=1.32,
                                               minNeighbors=5)  # detectMultiScale returns rectangles

    return faces, gray_img


# Given a directory below function returns part of gray_img which is face alongwith its label/ID
def labels_for_training_data(directory):
    faces = []
    faceID = []

    for path, subdirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("Skipping system file")  # Skipping files that startwith .
                continue

            #id = os.path.basename(path)  # fetching subdirectory names
            id = filename[-13:-4]
            img_path = os.path.join(path, filename)  # fetching image path
            print("img_path:", img_path)
            print("id:", id)
            test_img = cv2.imread(img_path)  # loading each image one by one
            if test_img is None:
                print("Image not loaded properly")
                continue
            faces_rect, gray_img= faceDetection(test_img)  # Calling faceDetection function to return faces detected in particular image
            if len(faces_rect) != 1:
                continue  # Since we are assuming only single person images are being fed to classifier
            (x, y, w, h) = faces_rect[0]
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from grayscale image
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces, faceID


# Below function trains haar classifier and takes faces,faceID returned by previous function as its arguments
def train_classifier(faces, faceID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceID))
    return face_recognizer


# Below function draws bounding boxes around detected face in image
def draw_rect(test_img, face):
    (x, y, w, h) = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=5)


# Below function writes name of person for detected label
def put_text(test_img, text, x, y):
    cv2.putText(test_img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 4)

@login_required
def dosage_updated(request):
	try:
		if request.method == 'POST':
			dosage_details = request.POST.get('dosage_details', '')
			dosage_date = request.POST.get('dosage_date', '')
			matchedaadhar = request.POST.get('matchedaadhar', '')
			check = request.POST.get('check')
			if(check=='on'):
				check="Clinical Checkup"
			else:
				check="Regular Checkup"
			height = request.POST['height_cm']
			weight = request.POST['weight']
			height = "0"+height
			weight = "0"+weight
			height = int(height,base=10)
			weight = int(weight,base=10)
			print(height)
			print(weight)
			if height==0 or weight==0:
				bmi=0
			else:
				bmi = weight/((height/100)**2)
				bmi = round(bmi,2)
			visit_status = True
			dosage_object = Dosage.objects.get(matchedaadhar=matchedaadhar)
			patient = Register.objects.get(fullaadhar=matchedaadhar)
			bmi_reg = dosage_object.initial_bmi
			p = dosage_object.dosage_date
			if p != None:
				try:
					update_stat = Stats.objects.get(name=patient.camp_loc.lower(),day=p)
					update_stat.visit_count = update_stat.visit_count+1
				except Stats.DoesNotExist:
					add_stat = Stats(name=patient.camp_loc.lower(),day=p,visit_count=1)
					add_stat.save()
			hist_obj, created = History.objects.get_or_create(histaadhar=matchedaadhar)
			count = hist_obj.history_count
			count=count+1
			hist_obj.history_count = count
			if count==1:
				hist_obj.Diagnosis1=check
				hist_obj.history1=dosage_details
				hist_obj.history_date1=dosage_date
				hist_obj.bmi1 = bmi_reg if bmi==0 else bmi
			elif count==2:
				hist_obj.Diagnosis2=check
				hist_obj.history2 = dosage_details
				hist_obj.history_date2 = dosage_date
				hist_obj.bmi2 = hist_obj.bmi1 if bmi==0 else bmi
			elif count==3:
				hist_obj.Diagnosis3=check
				hist_obj.history3 = dosage_details
				hist_obj.history_date3 = dosage_date
				hist_obj.bmi3 = hist_obj.bmi2 if bmi==0 else bmi
			else:
				hist_obj.Diagnosis1=hist_obj.Diagnosis2
				hist_obj.history1 = hist_obj.history2
				hist_obj.history_date1 = hist_obj.history_date2
				hist_obj.bmi1 = hist_obj.bmi2
				hist_obj.Diagnosis2=hist_obj.Diagnosis3
				hist_obj.history2 = hist_obj.history3
				hist_obj.history_date2 = hist_obj.history_date3
				hist_obj.bmi2 = hist_obj.bmi3
				hist_obj.Diagnosis3=check
				hist_obj.history3 = dosage_details
				hist_obj.history_date3 = dosage_date
				hist_obj.bmi3 = hist_obj.bmi2 if bmi==0 else bmi
			hist_obj.save()
			dosage_object.dosage_details = dosage_details
			dosage_object.dosage_date = dosage_date
			dosage_object.visit_status = visit_status
			dosage_object.save()
		return render(request,"success.html" ,{'message':"Dosage Updated",'data':"New Attendance", 'link':'/registration/attendance'})
	except Exception as e:
		#trace_back = traceback.format_exc()
		#message = str(e) + " " + str(trace_back)
		return render(request,"failure.html" ,{'message':str(e) ,'data':"Try Again", 'link':'/registration/attendance'})

@login_required
def get_patient_data(request):
	try:
		if request.method=='POST':
			inputaadhar = request.POST.get('inputaadhar','')
			patient_data = Register.objects.get(fullaadhar=inputaadhar)
			#patient_dose = Dosage.objects.get(matchedaadhar=inputaadhar)
			dose_history = History.objects.get(histaadhar=inputaadhar)
			path = patient_data.imagepath
			labels = []
			labels=['visit 1','Visit 2','Visit 3']
			bmchart=[]
			bmchart.append(int(dose_history.bmi1))
			bmchart.append(int(dose_history.bmi2))
			bmchart.append(int(dose_history.bmi3))
			with open(path, "rb") as image_file:
				encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
		return render(request, 'registration/data.html',{'data': patient_data,'dose': dose_history,'img':encoded_string,'label':labels, 'cdata':bmchart})
	except Exception as e:
		#trace_back = traceback.format_exc()
		#message = str(e) + " " + str(trace_back)
		return render(request,"failure.html" ,{'message':str(e) ,'data':"Try Again", 'link':'/registration/getdata/'})

@login_required
def searchdata(request):
	return render(request,'registration/getdata.html')


@csrf_exempt
def whatsApp_portal(request):

	lang = "EN" # change to BN . Dont use translate if lang = EN
	try:
		if request.method == 'POST':
			incoming_msg = request.POST['Body'].lower()
			aadhar = incoming_msg[1:]
			resp = MessagingResponse()
			msg = resp.message()
			#msg1 = resp.message()
			#msg2 = resp.message()

			responded = False
			try:
				#quote = "booo"
				#msg.body(quote)
				dos = Dosage.objects.get(matchedaadhar=aadhar)
				res = Register.objects.get(fullaadhar=aadhar)
				quote = "Your next dosage date and details are"
				if res.lang_pref == 'Hindi':
					lang = "HI"
				if res.lang_pref == 'Bengali':
					lang = "BN"
				if incoming_msg[0] == '1' and lang != "EN":
					responded = True
					translator= Translator(to_lang=lang)
					translation = translator.translate(quote)
					quote1 = str(dos.dosage_date)
					quotet2 = dos.dosage_details
					quote2 = translator.translate(quotet2)
					quote3 = ": "
					quote4 = " "
					quote5 = translator.translate("Dosage date")
					quote6 = translator.translate("Dosage details")
					msg.body(translation+quote3+quote4+quote5+quote3+quote1+quote4+quote6+quote3+quote2)
				elif incoming_msg[0] == '1' and lang == "EN":
					responded = True
					#translator= Translator(to_lang=lang)
					#translation = translator.translate(quote)
					quote1 = str(dos.dosage_date)
					quote2 = dos.dosage_details
					#quote2 = translator.translate(quotet2)
					quote3 = ": "
					quote4 = " "
					quote5 = "Dosage date"
					quote6 = "Dosage details"
					msg.body(quote+quote3+quote4+quote5+quote3+quote1+quote4+quote6+quote3+quote2)

				if incoming_msg[0] == '2' and lang != "EN":
					upcoming = Event.objects.order_by('day')
					dt = datetime.date.today()
					responded = True
		        #dates = Event.objects.order_by('day')
					for up in upcoming:
						if dt.strftime('%B') == up.day.strftime('%B'):
							translator= Translator(to_lang=lang)
							quote = str(up.day)
							quotet = up.notes
							quote1 = translator.translate(quotet)
							quotet2 = str(up.start_time)
							quote2 = translator.translate(quotet2)
							quote3 = str(up.end_time)
							quote4 = " "
							#quote8 = '/n'
							msg.body(quote1 +quote4+ quote+quote4+ quote2+quote4 + quote3 )
				elif incoming_msg[0] == '2' and lang == "EN":
					#if incoming_msg[0] == '2' and lang != "EN":
					upcoming = Event.objects.order_by('day')
					dt = datetime.date.today()
					responded = True
			        #dates = Event.objects.order_by('day')
					for up in upcoming:
						if dt.strftime('%B') == up.day.strftime('%B'):
								#translator= Translator(to_lang=lang)
							quote = str(up.day)								#quote1 = up.notes
								#quote1 = translator.translate(quotet)
							quote1 = up.notes
							quote2 = str(up.start_time)
								#quote2 = translator.translate(quotet2)
							quote3 = str(up.end_time)
							quote4 = " "
								#quote8 = '/n'
							msg.body(quote1 +quote4+ quote+quote4+ quote2+quote4 + quote3 )

				if incoming_msg[0] == '3' and lang != "EN":
					translator= Translator(to_lang=lang)
					quotet1 = "Available dates for checkup "
					quote1 = translator.translate(quotet1)
					responded = True
					#msg.body(quotet1)
					try:
						absentees_checkup = Event.objects.filter(notes="checkup for absentees")
						msg.body(quote1)

						for a in absentees_checkup:
							quote2 = str(a.day)
							msg.body(quote2)
					except Event.DoesNotExist:
						quotet3 = 'No dates available for checkup'
						quote3 = translator.translate(quotet3)
						msg.body(quote3)

				elif incoming_msg[0] == '3' and lang == "EN":
					#translator= Translator(to_lang=lang)
					quote1 = "Available dates for checkup "
					#quote1 = translator.translate(quotet1)
					responded = True
					#msg.body(quotet1)
					try:
						absentees_checkup = Event.objects.filter(notes="checkup for absentees")
						msg.body(quote1)

						for a in absentees_checkup:
							quote2 = str(a.day)
							msg.body(quote2)
					except Event.DoesNotExist:
						quote3 = 'No dates available for checkup'
						#quote3 = translator.translate(quotet3)
						msg.body(quote3)

				if incoming_msg[0] != '2' and incoming_msg[0] != '1' and incoming_msg[0] != '3' and lang != "EN":
					quote = 'Wrong input. Type 1 with your aadhar to get dosage details and 2 with your aadhar for events.'
					translator= Translator(to_lang=lang)
					translation = translator.translate(quote)
					responded = True
					msg.body(translation)
				elif incoming_msg[0] != '2' and incoming_msg[0] != '1' and incoming_msg[0] != '3' and lang == "EN":
					quote = 'Wrong input. Type 1 with your aadhar to get dosage details and 2 with your aadhar for events.'
					translator= Translator(to_lang=lang)
					responded = True
					translation = translator.translate(quote)
					msg.body(translation)








			except Dosage.DoesNotExist:
				try:
					inc = incoming_msg[10:]
					dos = Dosage.objects.get(matchedaadhar=inc)
					res = Register.objects.get(fullaadhar=inc)
					search = Event.objects.filter(notes="checkup for absentees")
					responded = True
					#lang = "EN"
					if res.lang_pref == 'Hindi':
						lang = "HI"
					if res.lang_pref == 'Bengali':
						lang = "BN"
					if res.lang_pref == 'English':
						done = 0
						#translator= Translator(to_lang=lang)
						#fixed = ""
						for d in search:
							if str(d.day) == incoming_msg[0:10]:
								#fixed = Event.objects.get(day=d.day)
								done = 1
								#break
						#translator= Translator(to_lang=lang)
							if done == 1:
								dos.dosage_date = d.day
								dos.save()
								quote = 'Your checkup date has been set.'
								quote2 = str(d.day)
								quote3 = str(d.start_time)
								quote4 = str(d.end_time)
								quote5 = " "
						#translator= Translator(to_lang=lang)
								#translation = translator.translate(quote)
								#msg.body(translation+quote2)
								msg.body(quote+quote5+quote2+quote5+quote3+quote5+quote4)
								break

						if done == 0:
							quote = 'This date is not available for checkup.'
							#translation = translator.translate(quote)
							msg.body(quote)
					else:
						done = 0
						translator= Translator(to_lang=lang)
						#fixed = ""
						for d in search:
							if str(d.day) == incoming_msg[0:10]:
								#fixed = Event.objects.get(day=d.day)
								done = 1
								#break
						#translator= Translator(to_lang=lang)
							if done == 1:
								dos.dosage_date = d.day
								dos.save()
								quote = 'Your checkup date has been set.'
								quote2 = str(d.day)
								quote3 = str(d.start_time)
								quote4 = str(d.end_time)
								quote5 = " "

						#translator= Translator(to_lang=lang)
								translation = translator.translate(quote)
								#msg.body(translation+quote2)
								msg.body(translation+quote5+quote2+quote5+quote3+quote5+quote4)
								break

						if done == 0:
							quote = 'This date is not available for checkup.'
							translation = translator.translate(quote)
							msg.body(translation)



				except Dosage.DoesNotExist:
					responded = True
					quote = " Type 1 with aadhar number to get dosage details, 2 with aadhar number to get event details, 3 with aadhar to get checkup dates and <date><aadhar number> to reschedule dosage date"
					msg.body(quote)

				#quote = 'Enter aadhar number correctly.Type 1 with your aadhar(1<aadhar>) to get dosage details and 2 with your aadhar(2<aadhar>) for events. '
				#translator= Translator(to_lang=lang)
				#translation = translator.translate(quote)
				#responded = True
				#msg.body(quote)










			if not responded:
				quote = 'Wrong input. Type 1 for list of upcoming events and type your aadhar number to get dosage details'
				translator= Translator(to_lang=lang)
				translation = translator.translate(quote)
				msg.body(translation)
				#quote = 'Wrong input. Type 1 for list of upcoming events and type your aadhar number to get dosage details'
				#translator= Translator(to_lang="BN")
				#translation = translator.translate(quote)
				#msg1.body(translation)
				#quote = 'Wrong input. Type 1 for list of upcoming events and type your aadhar number to get dosage details'
				#translator= Translator(to_lang=lang)
				#translation = translator.translate(quote)
				#msg2.body(quote)
			#retu##rn str(resp)
			return HttpResponse(str(resp))
		else:
			return HttpResponse("No")
	except Exception as e:
		trace_back = traceback.format_exc()
		message = str(e) + " " + str(trace_back)
		return HttpResponse(message)
