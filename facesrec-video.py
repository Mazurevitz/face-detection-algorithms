import numpy as np
import cv2
import pickle
from math import log1p
import os.path
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import time
from sklearn.preprocessing import scale

default_color = (255, 255, 255)
correctly_recognized_color = (0, 255, 0)
not_recognized_color = (0, 0, 255)

def make_upper_lower(value, frame_size, error):
    return value*1-(frame_size*error), value*1+(frame_size*error)

# def face_in_boundary(current_frame, starting_frame, ending_frame,
#                x, y, face_x, face_y, vid_w, vid_h):
#     face_x_down, face_x_up = make_upper_lower(face_x, vid_w, 0.1)
#     face_y_down, face_y_up = make_upper_lower(face_y, vid_h, 0.1)

#     if (current_frame > starting_frame and current_frame < ending_frame
#         and x > face_x_up and y > face_y_up
#         and x < face_x_down and y < face_y_down):
#         return True


def get_recognition_color(detected_name, actual_name, current_frame, starting_frame, ending_frame,
               x, y, face_x, face_y, vid_w, vid_h):
    face_x_down, face_x_up = make_upper_lower(face_x, vid_w, 0.1)
    face_y_down, face_y_up = make_upper_lower(face_y, vid_h, 0.1)

    color = not_recognized_color

    if (current_frame > starting_frame and current_frame < ending_frame
            and x < face_x_up and y < face_y_up
            and x > face_x_down and y > face_y_down):
        if detected_name == actual_name:
            # print("correctly recognized {0}".format(detected_name))
            color = correctly_recognized_color
        else:
            # print("not recognized correctly")
            color = not_recognized_color
    else:
        color = default_color

    return color

# def assign_face_check():
#     if(frame > 0 and frame < 238):
#         color = get_recognition_color(name, "bryan-cranston", current_frame, 0, 238, x, y, 290, 70, width, height)

#     # if(frame > 239 and frame < 342):
#     #     color = get_recognition_color(name, "eddie-redmaine", current_frame, 239, 342, x, y, 290, 80, width, height)
#     #     color = get_recognition_color(name, "benedict-cumberbatch", current_frame, 239, 342, x, y, 80, 40, width, height)
#     #     color = get_recognition_color(name, "bryan-cranston", current_frame, 239, 342, x, y, 515, 60, width, height)

#     if(frame > 238 and frame < 416):
#         color = get_recognition_color(name, "eddie-redmaine", current_frame, 343, 520, x, y, 240, 70, width, height)

#     if(frame > 417 and frame < 651):
#         color = get_recognition_color(name, "benedict-cumberbatch", current_frame, 521, 755, x, y, 260, 75, width, height)

def regressor_to_classifier(predictions, threshold = 0.5):
    output = []
    for prediction in predictions:
        if prediction is not None and prediction < threshold: 
            output.append(1)
        else: 
            output.append(0)
    return output

def confusion_matrix(true, predictions):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for t, p in zip(true, predictions):
        if t == 1 and p == 1: 
            TP += 1
        elif t == 0 and p == 1:
            FP += 1
        elif t == 1 and p == 0:  
            FN += 1
        else: 
            TN += 1
    print("TP = {}\nFP = {}\nTN = {}\nFN = {}".format(TP, FP, TN, FN))
    print("Precision = {}".format(str(TP / (TP + FP))))
    print("Recall = {}".format(str(TP / (FN + TP))))
    return TP, FP, TN, FN


def main():
    video_name = "redmaine-eddie-cumber-short-cut"
    cap = cv2.VideoCapture("videos/{0}.mp4".format(video_name))

    current_frame = 0
    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(3)  # float
    height = cap.get(4) # float

    video_length_frames = int(cv2.VideoCapture.get(cap, property_id))
    print("video length: {0}".format(video_length_frames))

    face_cascade = cv2.CascadeClassifier(
        'haar/haarcascade_frontalface_alt2.xml')
    # eye_cascade = cv2.CascadeClassifier('haar/haarcascade_eye.xml')

    recognizer = cv2.face.EigenFaceRecognizer_create()
    # recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    video_name = "_".join([video_name, 'output2.avi'])
    output_destination = os.path.join("output-videos", video_name)
    out = cv2.VideoWriter(output_destination, fourcc, 20.0, (int(width), int(height)))

    labels = {}
    font = cv2.FONT_HERSHEY_DUPLEX
    stroke = 1

    recognition_frame = [0 for x in range(video_length_frames+1)]
    recognition_values = [0 for x in range(video_length_frames+1)]
    true = [0 for x in range(video_length_frames+1)]

    # plt.figure(figsize=(15,5), dpi=120)
    # plt.xlim(0, video_length_frames)
    # plt.ylim(10000, 15000)

    badly_recognized = 0
    not_a_face = 0

    start = time.time()

    with open("labels.pickle", 'rb') as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}

    while(True):
        # capture frame by frame
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % 15 == 0:
            print("frame: {0}/{1}".format(current_frame, video_length_frames))
        current_frame += 1
        current_frame_text = "current frame: {0}/{1}".format(current_frame, video_length_frames)
        cv2.putText(frame, current_frame_text, (20, 340), font,
                    0.5, default_color, stroke, cv2.LINE_AA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5)
        
        conf = None
        recognition_values[current_frame] = conf

        for (x, y, w, h) in faces:
            # print(x, y, w, h)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (250, 250))
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
            name = labels[id_]

            # if(current_frame > 0 and current_frame < 49):
            #     color = get_recognition_color(name, "bryan-cranston", current_frame, 0, 49, x, y, 290, 70, width, height)

            # elif(current_frame > 50 and current_frame < 99):
            #     color = get_recognition_color(name, "eddie-redmaine", current_frame, 50, 99, x, y, 240, 70, width, height)

            # elif(current_frame > 100 and current_frame < 150):
            #     color = get_recognition_color(name, "benedict-cumberbatch", current_frame, 100, 149, x, y, 260, 75, width, height)

            if(current_frame > 0 and current_frame < 238):
                color = get_recognition_color(name, "bryan-cranston", current_frame, 0, 238, x, y, 290, 70, width, height)

            if(current_frame > 238 and current_frame < 416):
                color = get_recognition_color(name, "eddie-redmaine", current_frame, 238, 416, x, y, 240, 70, width, height)

            if(current_frame > 417 and current_frame < 651):
                color = get_recognition_color(name, "benedict-cumberbatch", current_frame, 417, 651, x, y, 260, 75, width, height)
 



            name_confidence = "{0} conf:{1}".format(name, int(conf))
            coordinates = "x:{0} y:{1} w:{2} h:{3}".format(x, y, w, h)
            cv2.putText(frame, name_confidence, (x-int(w), y),
                        font, log1p(w/128), color, stroke, cv2.LINE_AA)
            cv2.putText(frame, coordinates, (x-int(w/2), y+int(h*1.2)),
                        font, log1p(w/128), color, stroke, cv2.LINE_AA)

            # print(name)

            # img_item = "my-image-clr.png"
            # cv2.imwrite(img_item, roi_color)

            # color = (255, 0, 0)
            # stroke = 1
            end_coord_x = x + w
            end_coord_y = y + h
            cv2.rectangle(frame, (x, y), (end_coord_x,
                                        end_coord_y), color, stroke)

        recognition_frame[current_frame] = current_frame
        recognition_values[current_frame] = conf
        if color == not_recognized_color:
            badly_recognized += 1
            # plt.axvspan(current_frame, current_frame+1, facecolor='red', alpha=0.3, label="badly recognized")

        elif color == default_color:
            not_a_face += 1
            # plt.axvspan(current_frame, current_frame+1, facecolor='darkorange', alpha=0.3, label="not a face")
        if color == correctly_recognized_color:
            true[current_frame] = 1
        if conf is None:
            # plt.axvspan(current_frame, current_frame+1, facecolor='gray', label="no face detected")
            a = 1



        # display resulting frame
        cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            break

    recognition_values.pop(0)
    recognition_frame.pop(0)

    x_roc = []
    y_roc = []

    min_rec_val = int(min(x for x in recognition_values if x is not None)) + 1
    max_rec_val = int(max(x for x in recognition_values if x is not None)) - 1
    for threshold in range(min_rec_val, max_rec_val):
        print("threshold: {0} ".format(threshold))

        bool_predictions = regressor_to_classifier(recognition_values, threshold)
        TP, FP, TN, FN = confusion_matrix(true, bool_predictions)
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        x_roc.append(FPR)
        y_roc.append(TPR)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot(x_roc, y_roc)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig('ROC_Curve_{0}'.format("w_denoise"), bbox_inches='tight')
    plt.show()
    
    end = time.time()
    print(end - start)

    only_detected_frames = video_length_frames - not_a_face

    print("wrongly recognized: {0}%, not a face: {1}%, correct: {2}%\nwo no face - bad: {3}%, correct: {4}%"
          .format(round(badly_recognized/video_length_frames, 3),
                  round(not_a_face/video_length_frames, 3),
                  round((video_length_frames-not_a_face-badly_recognized)/video_length_frames, 3),
                  round(badly_recognized/only_detected_frames, 3),
                  round((only_detected_frames-badly_recognized)/only_detected_frames, 3)))

    # normalized_recongition_values = recognition_values / np.linalg.norm(recognition_values) * 100
    # plt.plot(recognition_frame, recognition_values, label="confidence")
    red_patch = mpatches.Patch(color='red', alpha=0.3, label='Wrong recognition')
    orange_patch = mpatches.Patch(color='darkorange', alpha=0.3, label='False Positive')
    black_patch = mpatches.Patch(color='gray', label='No face detected')
    white_patch = mpatches.Patch(color='white', label='Correct detection')
    # plt.legend(handles=[red_patch, orange_patch, black_patch, white_patch])
    # plt.savefig('performance_{0}.{1}'.format(video_name, 'jpg'), bbox_inches='tight')
    # plt.show()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

main()
