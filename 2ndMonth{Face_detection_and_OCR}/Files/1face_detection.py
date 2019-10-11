import cv2

eye_cascade = cv2.CascadeClassifier('Data/haarcascade_eye.xml')
frontalface_cascade = cv2.CascadeClassifier('Data/haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('Data/haarcascade_smile.xml')

img = cv2.imread('../Images/group.jpg')
#video = cv2.VideoCapture(0)

#ret, img = video.read()
frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

eyes = eye_cascade.detectMultiScale(frame, 1.1, 4)
faces = frontalface_cascade.detectMultiScale(frame, 1.1, 4)
smiles = smile_cascade.detectMultiScale(frame, 1.1, 4)

for (x,y,w,h) in faces:
    frame = cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 255), 4)
    faceROI = frame[y:y+h,x:x+w]
    eyes = eye_cascade.detectMultiScale(faceROI, 1.1, 4)
    #smiles = smile_cascade.detectMultiScale(faceROI, 1.1, 4)
    for (x2,y2,w2,h2) in eyes:
        eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        radius = int(round((w2 + h2)*0.25))
        frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

cv2.imshow("Image", img)
#if cv2.waitKey(1) & 0xFF == ord('q'):
#    break
#video.release()
cv2.waitKey(0)
cv2.destroyAllWindows()