import cv2 as cv
import os.path
from pathlib import Path
from matplotlib import pyplot as plt

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "augmented_images")

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                print(path)
                
                IMG = cv.imread(path)
                IMG = cv.resize(IMG, (300, 300))
                gray_image = cv.cvtColor(IMG, cv.COLOR_BGR2GRAY)

                eq_histogram_image = cv.equalizeHist(gray_image)

                plt.hist(IMG.ravel(),256,[0,256], label="color")
                plt.hist(gray_image.ravel(),256,[0,256], label="grey")
                plt.hist(eq_histogram_image.ravel(),256,[0,256], label="normalized histogram")
                plt.legend()
                plt.show()

                cv.imshow('color', IMG)
                cv.imshow('grey', gray_image)
                cv.imshow('Equalized Image', eq_histogram_image)

                cv.waitKey(0)
                cv.destroyAllWindows()
