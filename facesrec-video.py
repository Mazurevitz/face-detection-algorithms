import numpy as np
import cv2
import pickle
from math import log1p

def main():
    cap = cv2.VideoCapture("emily-test.mp4")

    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')

    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.read("trainer.yml")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,360))

    labels = {}
    with open("labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}

    while(True):
        #capture frame by frame
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # print(x,y,w,h)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (250,250))
            # print(len(roi_gray))
            roi_color = frame[y:y+h, x:x+w]

            # eyes = eye_cascade.detectMultiScale(roi_gray)
            # for (ex,ey,ew,eh) in eyes:
            #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

            id_, conf = recognizer.predict(roi_gray)
            # print("id: {0}, conf: {1}".format(id_, conf))
            # if conf>=12000:
                # print(id_)
                # print(labels[id_])
            font = cv2.FONT_HERSHEY_DUPLEX
            color = (255, 255, 255)
            stroke = 1
            name = labels[id_]
            cv2.putText(frame, name, (x,y), font, log1p(w/64), color, stroke, cv2.LINE_AA)
            # print(name)

            # img_item = "my-image-clr.png"
            # cv2.imwrite(img_item, roi_color)

            color = (255, 0, 0)
            stroke = 2
            end_coord_x = x + w
            end_coord_y = y + h
            cv2.rectangle(frame, (x,y), (end_coord_x, end_coord_y), color, stroke)


        #display resulting frame
        cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

main()
