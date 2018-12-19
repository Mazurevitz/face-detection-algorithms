import cv2
import os.path
from pathlib import Path
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')
recognizer = cv2.face.EigenFaceRecognizer_create()
recognizer.read("trainer.yml")

base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "images")

x_train = []
y_labels = []
label_ids = {}
current_id = 0

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            print(path)

            IMG = cv2.imread(path)

            gray = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            print ("Found {0} faces!".format(len(faces)))

            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                cv2.rectangle(IMG, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('Face found', IMG)
            cv2.waitKey(0)