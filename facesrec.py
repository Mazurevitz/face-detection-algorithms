import numpy as np
import cv2
import pickle

def main():
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.read("trainer.yml")

    labels = {}
    with open("labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v:k for k,v in og_labels.items()}

    while(True):
        #capture frame by frame
        ret, frame = cap.read()
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grey, scaleFactor=1.5, minNeighbors=5)
        for (x, y, w, h) in faces:
            # print(x,y,w,h)
            roi_grey = grey[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            id_, conf = recognizer.predict(roi_grey)
            if conf>=65 and conf <= 90:
                print(id_)
                print(labels[id_])
                font = cv2.FONT_HERSHEY_SIMPLEX
                color = (255, 255, 255)
                stroke = 2
                name = labels[id_]
                cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

            img_item = "my-image-clr.png"
            cv2.imwrite(img_item, roi_color)

            color = (255, 0, 0)
            stroke = 2
            end_coord_x = x + w
            end_coord_y = y + h
            cv2.rectangle(frame, (x,y), (end_coord_x, end_coord_y), color, stroke)

        #display resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()