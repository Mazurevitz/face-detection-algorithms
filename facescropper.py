import cv2
import os.path
from pathlib import Path
from matplotlib import pyplot as plt
import math
from operator import itemgetter

face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')

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
            for (x,y,w,h) in faces:
                cv2.rectangle(IMG,(x,y),(x+w,y+h),(255,0,0),2)

                roi_gray = gray[y:y+h, x:x+w]
                roi_color = IMG[y:y+h, x:x+w]

                roi_gray = cv2.resize(roi_gray, (250,250))
                roi_color = cv2.resize(roi_color, (250,250))

                roi_gray = cv2.blur(roi_gray,(5,5))

                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=7)
                eyes = sorted(eyes, key=itemgetter(2))
                eyes = eyes[-2:]
                # noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=7)
                print(eyes[0], eyes[1])
                print('\nsecond ',eyes[0][0], eyes[1][0])

                eye_ctr = 0


                rotation = math.atan2(eyes[0][1]-eyes[1][1], eyes[0][0]-eyes[1][0]) * 180 / math.pi

                print('rotation', rotation)

                for (ex,ey,ew,eh) in eyes:
                    eye_ctr += 1
                    print('eye/w = ',ew/w)
                    if(ew/w < 0.5):
                        cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                        cv2.circle(roi_gray, (math.floor(ex+ew/2),math.floor(ey+eh/2)),(5),(0,0,255),2)
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                        cv2.circle(roi_color, (math.floor(ex+ew/2),math.floor(ey+eh/2)),(5),(0,0,255),2)
                    print('left ', eye_ctr)
                    # if (eye_ctr > 1):
                    #     break

                # for (nx,ny,nw,nh) in noses:
                #     cv2.rectangle(roi_gray,(nx,ny),(nx+nw,ny+nh),(0,150,150),2)
                #     cv2.circle(roi_gray, (math.floor(nx+nw/2),math.floor(ny+nh/2)),(5),(0,150,0),2)
                #     cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,150,150),2)
                #     cv2.circle(roi_color, (math.floor(nx+nw/2),math.floor(ny+nh/2)),(5),(0,150,0),2)


                if(len(eyes) > 1):
                    M = cv2.getRotationMatrix2D((w/2, h/2), rotation, 1.0)
                    if (rotation < 0) :
                        rotation = 180 + rotation
                        print('new rotation: ', rotation)
                        roi_color = cv2.warpAffine(roi_color, M, (w, h))
                    if (rotation < 10):
                        roi_color = cv2.warpAffine(roi_color, M, (w, h))
                        print('new rotation: ', rotation)
                        


            cv2.imshow('Face found', roi_color)
            cv2.imshow('face-grey-blur', roi_gray)
            cv2.imshow('face-grey-blur', roi_gray)

            cv2.waitKey(0)