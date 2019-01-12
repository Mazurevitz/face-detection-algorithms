import os.path
from pathlib import Path
import math
from operator import itemgetter
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import cv2

class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    
    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image
        
    def dominantColors(self):
    
        #read image
        # img = cv2.imread(self.IMAGE)
        img = self.IMAGE
        
        #convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        #reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        #save image after operations
        self.IMAGE = img
        
        #using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(img)
        
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        
        #returning after converting to integer from float
        return self.COLORS.astype(int)

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    
    def plotClusters(self):
        #plotting 
        fig = plt.figure()
        ax = Axes3D(fig)        
        for label, pix in zip(self.LABELS, self.IMAGE):
            ax.scatter(pix[0], pix[1], pix[2], color = self.rgb_to_hex(self.COLORS[label]))
        plt.show()

    def plotHistogram(self):
       
        #labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS+1)
       
        #create frequency count tables    
        (hist, _) = np.histogram(self.LABELS, bins = numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()
        
        #appending frequencies to cluster centers
        colors = self.COLORS
        
        #descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()] 
        
        #creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0
        
        #creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 500
            
            #getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]
            
            #using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r,g,b), -1)
            start = end	
        
        #display chart
        plt.figure()
        plt.axis("off")
        # plt.imshow(chart)
        # plt.show()


eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')
eyes_number = 2

def rotation_matrix_from_eyes(face_image, w, h):
    eyes = select_and_crop_eyes(face_image, eyes_number)
    if len(eyes) > 1:
        rotation = get_angle_between_eyes(eyes)
    else:
        rotation = 0

    print('first rotation', rotation)
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

        if (rotation > 20):
            rotation = rotation - 180
            print('>20 - 180 new rotation: ', rotation)

        if (rotation > 20):
            rotation = 0
            print('>20 = 0 new rotation: ', rotation)
        
        if (math.fabs(rotation) > 20):
            rotation = 0
            print("abs new rotation ", rotation)

    M = cv2.getRotationMatrix2D((w/2, h/2), rotation, 1.0)
    return M

def get_angle_between_eyes(eyes):
    # noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=7)
    # print(eyes[0], eyes[1])
    # print('\nsecond ',eyes[0][0], eyes[1][0])
    return math.atan2(eyes[0][1]-eyes[1][1], eyes[0][0]-eyes[1][0]) * 180 / math.pi

def select_and_crop_eyes(face_image, crop_by_number):
    eyes = eye_cascade.detectMultiScale(face_image, scaleFactor=1.1, minNeighbors=7)
    print('Found {0} eyes!'.format(len(eyes)))
    eyes = sorted(eyes, key=itemgetter(2))
    return eyes[-crop_by_number:]

def detect_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # lower = np.array([0, 48, 80])
    # upper = np.array([20, 255, 255])

    lower = np.array([0, 48, 80])
    upper = np.array([20, 255, 255])

    converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

	# apply a series of erosions and dilations to the mask
	# using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

	# blur the mask to help remove noise, then apply the
	# mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(image, image, mask = skinMask)

	# show the skin in the image along with the mask

    # cv2.imshow("skin", skin)

    # cv2.imshow('frame',image)
    # color = ('b','g','r')
    # for i,col in enumerate(color):
    #     histr = cv2.calcHist([image],[i],None,[256],[0,256])
    #     plt.plot(histr,color = col)
    #     plt.xlim([0,256])
    # plt.show()

    return skin

    # cv2.imshow('mask',mask)
    # cv2.imshow('res',res)

def cluster_color_show(img):
    # img = cv2.imread(img_name)
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 12
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    # cv2.imshow('res2',res2)

def main():
    face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_alt2.xml')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    image_dir = os.path.join(base_dir, "images")

    progress = 0
    clusters = 6

    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
                # print(path)

                IMG = cv2.imread(path)
                # cv2.imshow('before cutout', IMG)
                
                green = np.uint8([[[0,255,0 ]]])
                hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
                print (hsv_green)

                # print("cutout show!")

                IMG_copy = cv2.imread(path)
                width, height = IMG.shape[:2]
                # print("width: {0}, height{1}".format(width, height))

                gray = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                print ("Found {0} faces!".format(len(faces)))

                # Draw a rectangle around the faces
                for (x,y,w,h) in faces:
                    # cv2.rectangle(IMG,(x,y),(x+w,y+h),(255,0,0),2)

                    roi_gray = gray[y:y+h, x:x+w]
                    roi_color = IMG[y:y+h, x:x+w]

                    # cluster_color_show(roi_color)

                    # dc = DominantColors(roi_color, clusters)
                    # colors = dc.dominantColors()
                    # print('rgb', colors)
                    # print(colors[0])

                    # color_array = np.zeros((1,clusters,3), dtype=int)
                    # color_array[0] = colors
                    # print('color array: ', color_array)

                    # hsv_colors = cv2.cvtColor(np.uint8(color_array), cv2.COLOR_RGB2HSV)
                    # print('hsv', hsv_colors)

                    # # TEST TEST TEST
                    # color_test = np.zeros((1,1,3), dtype=int)
                    # color_test[0][0] = colors[0]
                    # print('color test: ', color_test)

                    # hsv_test = cv2.cvtColor(np.uint8(color_test), cv2.COLOR_RGB2HSV)
                    # print('hsv test', hsv_test)
                    # END OF TEST

                    # face_cutout = detect_color(roi_color)
                    # cv2.imshow('after cutout', face_cutout)


                    roi_gray = cv2.resize(roi_gray, (250,250))
                    roi_color = cv2.resize(roi_color, (250,250))

                    roi_gray = cv2.blur(roi_gray,(5,5))

                    rotation_matrix = rotation_matrix_from_eyes(roi_color, w, h)

                    roi_gray_warped = cv2.warpAffine(roi_gray, rotation_matrix, (250, 250))
                    img_warped = cv2.warpAffine(IMG_copy, rotation_matrix, (height, width))
                    # print("M: {0}, w: {1}, h: {2}".format(rotation_matrix, w, h))

                    new_warped = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
                    faces_warped = face_cascade.detectMultiScale(new_warped, scaleFactor=1.1, minNeighbors=5)

                    for (x,y,w,h) in faces_warped:
                        # cv2.rectangle(img_warped,(x,y),(x+w,y+h),(255,0,0),2)
                        new_roi_gray_warped = new_warped[y:y+h, x:x+w]
                        new_roi_gray_warped = cv2.resize(new_roi_gray_warped, (250,250))
                        cv2.fastNlMeansDenoising(new_roi_gray_warped, new_roi_gray_warped)
                        new_roi_gray_warped = cv2.blur(new_roi_gray_warped,(5,5))

                        eq_histogram_image = cv2.equalizeHist(new_roi_gray_warped)

                        plt.hist(IMG.ravel(),256,[0,256], label="color")
                        plt.hist(roi_gray.ravel(),256,[0,256], label="grey")
                        plt.hist(eq_histogram_image.ravel(),256,[0,256], label="normalized histogram")
                        plt.legend()

                print("file: {0}".format(file))
                dirname = root.split(os.path.sep)[-1]
                # print("directory : {0}, root: {1}".format(dirname, base_dir))
                joined = ".".join([file, "jpg"])
                folder_path = os.path.join(base_dir, "augmented_images", dirname)
                save_path = os.path.join(folder_path, joined)
                # new_path = os.path.join()
                # print("savepath: {0}".format(save_path))
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                cv2.imwrite(save_path, eq_histogram_image)
                # print("saved: {0}".format(save_path))
                print("progress: {0}/{1}".format(progress, 50))
                progress += 1

                # cv2.imshow('1.original', IMG_copy)
                # cv2.imshow('2.face that was found', roi_color)
                # cv2.imshow('3.face in grey', roi_gray)
                # cv2.imshow('4.after rotation', roi_gray_warped)
                # cv2.imshow('5.warped original', img_warped)
                # cv2.imshow('6.after rotation', new_roi_gray_warped)
                # cv2.imshow('7.after normalization', eq_histogram_image)
                # plt.show()

                cv2.waitKey(0)

# img = 'paul.png'
# clusters = 6
# dc = DominantColors(img, clusters)
# colors = dc.dominantColors()
# print(colors)
main()
# cluster_color_show(img)
