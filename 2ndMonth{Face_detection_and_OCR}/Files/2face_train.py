import cv2
import os
import numpy as np
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "FaceTrain")

face_cascade = cv2.CascadeClassifier('Data/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("JPG"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (int(img.shape[1] / 3), int(img.shape[1] / 3)), interpolation = cv2.INTER_AREA)
            faces = face_cascade.detectMultiScale(img, 1.1, 5)

            for (x,y,w,h) in faces:
                roi = img[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("Data/face-labels.pickle", 'wb') as f:
	pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("Data/face-trainner.yml")