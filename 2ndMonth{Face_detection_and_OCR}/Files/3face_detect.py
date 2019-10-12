import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('Data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('Data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('Data/haarcascade_smile.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Data/face-trainner.yml")

labels = {"person_name": 1}
with open("Data/face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
    	roi_gray = gray[y:y+h, x:x+w]
    	roi_color = frame[y:y+h, x:x+w]
    	id_, conf = recognizer.predict(roi_gray)
    	if conf>=4 and conf <= 85:
    		cv2.putText(frame, labels[id_], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    	cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    	subitems = smile_cascade.detectMultiScale(roi_gray)
    	for (ex,ey,ew,eh) in subitems:
    		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()