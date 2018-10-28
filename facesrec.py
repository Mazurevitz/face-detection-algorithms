import numpy as np
import cv2

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')

while(True):
    #capture frame by frame
    ret, frame = cap.read()
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print(x,y,w,h)
        roi_grey = grey[y:y+h, x:x+w]
        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_grey)

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