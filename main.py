import cv2
import json
import os

#1 Load existing names

if not os.path.exists('names.json'):
    print('Name file not found')
    exit()
with open('names.json','r')as f:
    names=json.load(f)

#2 Load trained model

recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('my_model.yml')
detector=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

#3 Setup camera

cam=cv2.VideoCapture(0)
font=cv2.FONT_HERSHEY_SIMPLEX
print('Starting the face recognition. Press "ESC" to exit.')

while True:
    ret,frame=cam.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector.detectMultiScale(gray,1.3,5)
    for(x,y,w,h)in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        id,confidence=recognizer.predict(gray[y:y+h,x:x+w])
        if confidence<100:
            name_key=str(id)
            if name_key in names:
                name=names[name_key]
            else:
                name='Unknown'
            confidence_text=f'{round(100-confidence)}%'
        else:
            name='Unknown'
            confidence_text='0%'
        cv2.putText(frame,name+confidence_text,(x,y-10),font,0.8,(255,255,255),2)
    cv2.imshow('Face Recognition',frame)
    if cv2.waitKey(10)& 0xFF==27:
        break
cam.release()
cv2.destroyAllWindows()