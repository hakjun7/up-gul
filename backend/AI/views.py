from django.shortcuts import render
from django.http.response import JsonResponse
# DRF
from rest_framework.response import Response
from rest_framework.decorators import api_view
# AI
import cv2
import dlib       # dlib face detection
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model

from PIL import Image

# face_detection = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml') # opencv face detection
detector = dlib.get_frontal_face_detector()  # dlib face detection
emotion_classifier = load_model('emotion/model_filter(best).h5', compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful",
            "Happy", "Sad", "Surpring", "Neutral"]


def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)


@api_view(['GET'])
def emotions(request):
    data = []
    camera = cv2.VideoCapture(0)  # api 요청받을때 사진 받으면 필요없음
    camera.set(3, 640)  # api 요청받을때 사진 받으면 필요없음
    camera.set(4, 480)  # api 요청받을때 사진 받으면 필요없음

    while True:
        ret, frame = camera.read()  # api 요청받을때 사진 받으면 필요없음
        # frame = cv2.imread('사진.jpg', cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray, 1)  # dlib face detection
        # faces = face_detection.detectMultiScale(gray,           # opencv face detection
        #                                         scaleFactor=1.1,
        #                                         minNeighbors=5,
        #                                         minSize=(30,30))

        emotion_all = {}

        if len(faces) > 0:
            # face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]      #opencv face detection
            # (fX, fY, fW, fH) = face

            (fX, fY, fW, fH) = rect_to_bb(faces[0])  # dlib face detection

            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            cv2.putText(frame, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

            negative = [0, 1, 2, 4]
            positive = [3, 5]

            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                if i in negative:
                    emotion_all.update({emotion: prob*100})
                elif i in positive:
                    emotion_all.update({emotion: prob*100})
                else:
                    emotion_all.update({emotion: prob*100})
                # angry, sad, disgusting, fearful - 부정적 / happy,surprising - 긍정적 / neutral - 무표정

        if emotion_all:
            pos_prob = 0
            neg_prob = 0
            for i in range(len(EMOTIONS)):
                if i in negative:
                    neg_prob += emotion_all[EMOTIONS[i]]
                elif i in positive:
                    pos_prob += emotion_all[EMOTIONS[i]]

            if neg_prob > 40:
                emotion = 'negative'
            elif pos_prob > 40:
                emotion = 'positive'
            else:
                emotion = 'neutral'

        if emotion_all:
            data.append(emotion_all)
            pos_prob = 0
            neg_prob = 0
            for i in range(len(EMOTIONS)):
                if i in negative:
                    neg_prob += emotion_all[EMOTIONS[i]]
                elif i in positive:
                    pos_prob += emotion_all[EMOTIONS[i]]

            if neg_prob > 40:
                emotion = 'negative'
            elif pos_prob > 40:
                emotion = 'positive'
            else:
                emotion = 'neutral'
            data.append(emotion)
            print('face on')
        else:
            emotion_all.update({'face': None})
            data.append(emotion_all)
            print('face off')

        break

    return Response(data)


@api_view(['POST'])
def emotions_api(request, user_name):
    data = []
    image_file = request.data['image']  # get the image
    image_object = Image.open(image_file).convert('RGB')
    frame = np.array(image_object)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 1)  # dlib face detection

    emotion_all = {}

    if len(faces) > 0:
        # face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]      #opencv face detection
        # (fX, fY, fW, fH) = face

        (fX, fY, fW, fH) = rect_to_bb(faces[0])  # dlib face detection

        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]

        cv2.putText(frame, label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        negative = [0, 1, 2, 4]
        positive = [3, 5]

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            if i in negative:
                emotion_all.update({emotion: prob*100})
            elif i in positive:
                emotion_all.update({emotion: prob*100})
            else:
                emotion_all.update({emotion: prob*100})
            # angry, sad, disgusting, fearful - 부정적 / happy,surprising - 긍정적 / neutral - 무표정

    if emotion_all:
        pos_prob = 0
        neg_prob = 0
        for i in range(len(EMOTIONS)):
            if i in negative:
                neg_prob += emotion_all[EMOTIONS[i]]
            elif i in positive:
                pos_prob += emotion_all[EMOTIONS[i]]

        if neg_prob > 40:
            emotion = 'negative'
        elif pos_prob > 40:
            emotion = 'positive'
        else:
            emotion = 'neutral'

    if emotion_all:
        data.append(emotion_all)
        pos_prob = 0
        neg_prob = 0
        for i in range(len(EMOTIONS)):
            if i in negative:
                neg_prob += emotion_all[EMOTIONS[i]]
            elif i in positive:
                pos_prob += emotion_all[EMOTIONS[i]]

        if neg_prob > 40:
            emotion = 'negative'
        elif pos_prob > 40:
            emotion = 'positive'
        else:
            emotion = 'neutral'
        data.append(emotion)
        print('face on')
    else:
        emotion_all.update({'face': None})
        data.append(emotion_all)
        print('face off')

    return Response(data)
