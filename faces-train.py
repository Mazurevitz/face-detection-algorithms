import os
import pickle
from PIL import Image
import numpy as np
import cv2

def main():
    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')
    recognizer = cv2.face.EigenFaceRecognizer_create()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(BASE_DIR, "images")

    x_train = []
    y_labels = []
    label_ids = {}
    current_id = 0

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                # print(label, path)
                if not label in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                id_ = label_ids[label]
                # print(label_ids)

                pil_image = Image.open(path).convert("L") #greyscale
                size = (300, 300)
                final_image = pil_image.resize(size, Image.ANTIALIAS)
                
                image_array = np.array(final_image, "uint8")
                # print(image_array)
                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi = image_array[y:y+h, x:x+w]
                    x_train.append(roi)
                    y_labels.append(id_)

    # print(y_labels)
    # print(x_train)

    with open("labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("trainer.yml")

main()