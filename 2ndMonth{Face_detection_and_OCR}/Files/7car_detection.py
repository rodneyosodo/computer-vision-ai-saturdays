import cv2
import numpy as np

body_classifier = cv2.CascadeClassifier('Data/haarcascade_fullbody.xml')

cap = cv2.VideoCapture('../Images/walking.avi')

while cap.isOpened():

    ret, frame = cap.read()
    frame = cv2.resize(frame, (int(frame.shape[1] / 2), int(frame.shape[1] / 2)), interpolation = cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)
    
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Pedestrians', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()