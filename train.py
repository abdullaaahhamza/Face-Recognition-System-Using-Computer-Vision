import cv2
import numpy as np
import os
import json

#1 Load existing name if available or create new

name_file='names.json'
if os.path.exists(name_file):
    with open(name_file,'r')as f:
        names=json.load(f)
else:
    names={}

#2 Function to add name

person_name=input('Enter the name of the person to add :')
person_id=len(names)+1
names[str(person_id)]=person_name

#3 Setup Camera

cam=cv2.VideoCapture(0)
detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

face_samples=[]
ids=[]
count=0
print(f'Starting to capture the faces {person_name} with ID:{person_id}, Look at the camera...')

while count<30:
    ret,frame=cam.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray,1.3,5)

    for (x,y,w,h)in faces:
        face_samples.append(gray[y:y+h,x:x+w])
        ids.append(person_id)
        count+=1

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('Training',frame)

    if cv2.waitKey(100) & 0xFF==27:
        break

#4 Train the model

recognizer=cv2.face.LBPHFaceRecognizer_create()

if os.path.exists('my_model.yml'):
    recognizer.read('my_model.yml')
    print('Updating existing model')
    recognizer.update(face_samples,np.array(ids))
else:
    print('Creatinng a new model')
    recognizer.train(face_samples,np.array(ids))
recognizer.write('my_model.yml')

#5 Save the names

with open(name_file,'w')as f:
    json.dump(names,f)

cam.release()
cv2.destroyAllWindows()