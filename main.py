import cv2
import keras
import time
import numpy as np
from collections import Counter, defaultdict
import math

from keras.models import Sequential


#506
#712
#801

def ellerDistance(x1,y1,x2,y2):
    return math.sqrt(pow((x1-x2),2)+pow((y1-y2),2))

classes = ['T-shirt/top', "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
model = keras.models.load_model('convolutional_fashion_model.h5')


#Constants
MIN_AREA = 500
MAX_AREA = 10000
TIME_TO_DETECT = 100
TIME_TO_WARN = 200
TIME_TO_FORGET = 50
BIGGEST_SIZE = 15000
DISTANCE_TO_UNDETECT = 200



cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_AUTOFOCUS, False)

_,test_frame = cap.read()

dimentions = test_frame.shape

time.sleep(4)
_, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (21, 21), 0)


track_temp = []
track_master = []
track_temp2 = []

temp_detected = []
detected_bags = []

top_contour_dict = defaultdict(int)
obj_detected_dict = defaultdict(int)
bags_centroids = defaultdict(list)

frameno = 0

consecutiveFrame = 20

undetect_centroids = []

while True:

    _, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    difference = cv2.absdiff(first_gray, gray_frame)
    _, difference = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)
    kernel2 = np.ones((3, 3), np.uint8)
    difference = cv2.morphologyEx(difference, cv2.MORPH_CLOSE, kernel2, iterations=2)

    if len(undetect_centroids) != 0:
        for centroid in undetect_centroids:
            cv2.line(difference, (centroid[0], centroid[1]), (centroid[2], centroid[3]), (255, 255, 255), 5)

    frameno = frameno + 1

    contours, _ = cv2.findContours(difference, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tempContours = []
    human_move_centroids = []
    undetect_centroids.clear()

    for contour in contours:

        moments = cv2.moments(contour)
        if moments['m00'] == 0:
            pass
        else:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            (x, y, w, h) = cv2.boundingRect(contour)

            if x == 0 or y == 0 or x + w == dimentions[1] or y + h == dimentions[0]:
                continue

            if cv2.contourArea(contour) > BIGGEST_SIZE:
                human_move_centroids.append([cx, cy])

            if cv2.contourArea(contour) < MIN_AREA or cv2.contourArea(contour) > MAX_AREA:
                pass
            else:
                tempContours.append(contour)

                sumcxcy = cx + cy

                track_temp.append([cx + cy, frameno])
                track_master.append([cx + cy, frameno])

                countuniqueFrame = set(j for i, j in track_master)

                if len(countuniqueFrame) > consecutiveFrame or False:
                    mainframeno = min(j for i, j in track_master)
                    for i, j in track_master:
                        if j != mainframeno:
                            track_temp2.append([i, j])

                    track_master = list(track_temp2)
                    track_temp2 = []

                countcxcy = Counter(i for i, j in track_master)
                for i, j in countcxcy.items():
                    if j >= consecutiveFrame:
                        top_contour_dict[i] += 1

                if sumcxcy in top_contour_dict:
                    if top_contour_dict[sumcxcy] > TIME_TO_DETECT:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                        bags_centroids[sumcxcy] = [cx, cy]
                        obj_detected_dict[sumcxcy] = frameno

                    if top_contour_dict[sumcxcy] > TIME_TO_WARN:
                        crop_img = frame[y:y + h, x:x + w]
                        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                        crop_img = cv2.resize(crop_img, (28, 28))
                        crop_img = np.invert(crop_img)
                        crop_img = (crop_img.astype(np.float32)) / 255.0
                        crop_img = crop_img.reshape(1, 28, 28, 1)
                        prediction = model.predict(crop_img)
                        resClass = np.argmax(prediction[0])
                        if resClass == 8:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                            obj_detected_dict[sumcxcy] = frameno
                            bags_centroids[sumcxcy] = [cx, cy]
                            resPredict = int(max(prediction[0]) * 100)
                            tempClass = str(resPredict) + '%'
                            cv2.putText(frame, tempClass, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA, False)
                        prediction = None

    for i, j in list(obj_detected_dict.items()):
        for ch in human_move_centroids:
            dist = ellerDistance(ch[0], ch[1], bags_centroids[i][0], bags_centroids[i][1])
            if dist < DISTANCE_TO_UNDETECT:
                cv2.line(frame, (ch[0], ch[1]), (bags_centroids[i][0], bags_centroids[i][1]), (255, 0, 0), 5)
                undetect_centroids.append([ch[0], ch[1], bags_centroids[i][0], bags_centroids[i][1]])
                if frameno - obj_detected_dict[i] > TIME_TO_FORGET:
                    obj_detected_dict.pop(i)
                    top_contour_dict[i] = 0

    cv2.imshow('Abandoned Object Detection', frame)

    key = cv2.waitKey(65)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()