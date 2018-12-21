import os.path
from pathlib import Path
import math
from operator import itemgetter
from matplotlib import pyplot as plt
import cv2

eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')
eyes_number = 2

def rotation_matrix_from_eyes(face_image, w, h):
    eyes = select_and_crop_eyes(face_image, eyes_number)
    if len(eyes) > 1:
        rotation = get_angle_between_eyes(eyes)
    else:
        rotation = 0

    print('rotation', rotation)
    eye_ctr = 0
    for (ex,ey,ew,eh) in eyes:
        eye_ctr += 1
        print('eye/w = ',ew/w)
        if(ew/w < 0.5):
            cv2.rectangle(face_image,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.circle(face_image, (math.floor(ex+ew/2),math.floor(ey+eh/2)),(5),(0,0,255),2)
        print('eye_ctr: ', eye_ctr)

    if len(eyes) > 1:
        if (rotation < 0):
            rotation = 180 + rotation
            print('<0 new rotation: ', rotation)

        if (rotation > 10):
            rotation = 180 - rotation
            print('>20 new rotation: ', rotation)

        if (rotation > 15):
            rotation = 0
            print('>15 new rotation: ', rotation)

        M = cv2.getRotationMatrix2D((w/2, h/2), rotation, 1.0)
    return M

def get_angle_between_eyes(eyes):
    # noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=7)
    print(eyes[0], eyes[1])
    print('\nsecond ',eyes[0][0], eyes[1][0])
    return math.atan2(eyes[0][1]-eyes[1][1], eyes[0][0]-eyes[1][0]) * 180 / math.pi

def select_and_crop_eyes(face_image, crop_by_number):
    eyes = eye_cascade.detectMultiScale(face_image, scaleFactor=1.1, minNeighbors=7)
    eyes = sorted(eyes, key=itemgetter(2))
    return eyes[-crop_by_number:]

def normalize_image(gray_image, ):
    eq_histogram_image = cv2.equalizeHist(gray_image)


def main():
    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "images")

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                print(path)

                IMG = cv2.imread(path)
                IMG_copy = cv2.imread(path)
                width, height = IMG.shape[:2]
                print("width: {0}, height{1}".format(width, height))

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

                    rotation_matrix = rotation_matrix_from_eyes(roi_color, w, h)

                    roi_gray_warped = cv2.warpAffine(roi_gray, rotation_matrix, (250, 250))
                    img_warped = cv2.warpAffine(IMG_copy, rotation_matrix, (width, height))
                    print("M: {0}, w: {1}, h: {2}".format(rotation_matrix, w, h))

                    new_warped = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
                    faces_warped = face_cascade.detectMultiScale(new_warped, scaleFactor=1.2, minNeighbors=5)

                    for (x,y,w,h) in faces_warped:
                        cv2.rectangle(img_warped,(x,y),(x+w,y+h),(255,0,0),2)
                        new_roi_gray_warped = new_warped[y:y+h, x:x+w]
                        new_roi_gray_warped = cv2.resize(new_roi_gray_warped, (250,250))
                        new_roi_gray_warped = cv2.blur(new_roi_gray_warped,(5,5))

                        eq_histogram_image = cv2.equalizeHist(new_roi_gray_warped)

                        plt.hist(IMG.ravel(),256,[0,256], label="color")
                        plt.hist(roi_gray.ravel(),256,[0,256], label="grey")
                        plt.hist(eq_histogram_image.ravel(),256,[0,256], label="normalized histogram")
                        plt.legend()

                cv2.imshow('1.original', IMG_copy)
                cv2.imshow('2.face that was found', roi_color)
                cv2.imshow('3.face in grey', roi_gray)
                cv2.imshow('4.after rotation', roi_gray_warped)
                cv2.imshow('5.warped original', img_warped)
                cv2.imshow('6.after rotation', new_roi_gray_warped)
                cv2.imshow('7.after normalization', eq_histogram_image)
                plt.show()

                cv2.waitKey(0)

main()
