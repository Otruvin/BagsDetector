import cv2
import keras
import numpy as np

cap = cv2.VideoCapture(0)

while True:

    _, frame = cap.read()

    cv2.imshow("frame", frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()