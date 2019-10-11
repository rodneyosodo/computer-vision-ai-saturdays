from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

confidence_score = 0.5
prototxt = "Data/deploy.prototxt"
model = "Data/weights.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt, model)
cap = cv2.VideoCapture(0)
time.sleep(2)

while True:
    ret, frame = cap.read()
    #frame = cv2.resize(frame, (400, 400))
    #print(frame)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]
        
        if confidence < confidence_score:
            continue
 
		# compute the (x, y)-coordinates of the bounding box for the object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()