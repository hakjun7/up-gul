import dlib
import numpy as np
import cv2

detector = dlib.get_frontal_face_detector()

cap = cv2.VideoCapture(0)

while True:
    ret, img_frame = cap.read()
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    dets = detector(img_gray, 1)
    for face in dets:
        cv2.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()),(0, 0, 255), 3)

    cv2.imshow('result', img_frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()