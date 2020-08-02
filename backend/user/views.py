from django.shortcuts import render
import cv2
import traceback
import os
import numpy as np
import pytesseract
from django.http import JsonResponse
from django.http import HttpResponse
from django.shortcuts import render, redirect
from datetime import datetime
from django.utils.dateparse import parse_date
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
from registration.models import Register,Dosage,History
from twilio.twiml.messaging_response import MessagingResponse
from django.views.decorators.csrf import csrf_exempt
from stats_graphs.models import Stats
from django.contrib.auth.decorators import login_required

# Create your views here.

def get_data_with_biometric(request):
    if (request.method == 'POST'):
        attendbase64 = request.POST.get('imurlat', '')
        seed(1)
        value = randint(0, 999999999999)
        imagename = 'facetesting' + str(datetime.now().strftime('%Y_%m_%d_%H_%M_%S')) + str(value) + '.jpg'
        with open('./facematch/' + imagename, "wb") as fh:
            fh.write(base64.b64decode(attendbase64))

        # This module takes images  stored in diskand performs face recognition
        test_img = cv2.imread('facematch/' + imagename)  # test_img path
        faces_detected, gray_img = faceDetection(test_img)
        print("faces_detected:", faces_detected)

        # Comment belows lines when running this program second time.Since it saves training.yml file in directory
        faces, faceID = labels_for_training_data('faces')
        face_recognizer = train_classifier(faces, faceID)
        face_recognizer.write('trainingData.yml')

        # Uncomment below line for subsequent runs
        # face_recognizer=cv2.face.LBPHFaceRecognizer_create()
        # face_recognizer.read('trainingData.yml')#use this to load training data for subsequent runs

        # name = {0: "Shriya", 1: "Sagnik"}  # creating dictionary containing names for each label
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
                patient_data = Register.objects.get(fullaadhar=matchedaadhar)
                patient_dose = History.objects.get(histaadhar=matchedaadhar)
                path = patient_data.imagepath


                labels = []
                #queryset = City.objects.order_by('-population')[:5]
                
                #d=1
                #labels.append(patient_dose.history_date1)
                #labels.append(patient_dose.history_date2)
                #labels.append(patient_dose.history_date3)
                labels=['visit 1','Visit 2','Visit 3']
                #print(labels)
                bmchart=[]
                bmchart.append(int(patient_dose.bmi1))
                bmchart.append(int(patient_dose.bmi2))
                bmchart.append(int(patient_dose.bmi3))
                #print(bmchart)
                
                

                with open(path, "rb") as image_file:
                        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                #os.remove('facematch/' + imagename)
                return render(request, 'datauser.html',
                                  {'data': patient_data, 'dose': patient_dose, 'img': encoded_string, 'label':labels, 'cdata':bmchart })
                #os.remove('facematch/' + imagename)
                #return render(request, 'registration/dosage.html', {'matchedaadhar': matchedaadhar})
        except Exception as e:
            # trace_back = traceback.format_exc()
            # message = str(e) + " " + str(trace_back)
            return render(request, "failure.html",
                          {'message': str(e), 'data': "Try Again", 'link': '/user/user-data/'})

    # resized_img = cv2.resize(test_img, (1000, 1000))
    # cv2.imshow("face dtecetion tutorial", resized_img)
    # cv2.waitKey(0)  # Waits indefinitely until a key is pressed
    # cv2.destroyAllWindows()

    return render(request, 'userdata.html')

def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)  # convert color image to grayscale
    face_haar_cascade = cv2.CascadeClassifier(
        'HaarCascade/haarcascade_frontalface_default.xml')  # Load haar classifier
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

            # id = os.path.basename(path)  # fetching subdirectory names
            id = filename[-13:-4]
            img_path = os.path.join(path, filename)  # fetching image path
            print("img_path:", img_path)
            print("id:", id)
            test_img = cv2.imread(img_path)  # loading each image one by one
            if test_img is None:
                print("Image not loaded properly")
                continue
            faces_rect, gray_img = faceDetection(
                test_img)  # Calling faceDetection function to return faces detected in particular image
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
