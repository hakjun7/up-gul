import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import threading
from django.http import StreamingHttpResponse



face_detection = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('emotion/emotion_model.hdf5', compile=False)

EMOTIONS = ["Angry" ,"Disgusting","Fearful", "Happy", "Sad", "Surpring", "Neutral"]


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(3, 640) # 비디오 크기조절
        self.video.set(4, 480)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            #minSize = (int(minW), int(minH)),
            )
        if len(faces) > 0: # 얼굴 감지됬을때
            face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            #얼굴 하나 검출
            (fX, fY, fW, fH) = face
            # Resize the image to 48x48 for neural network
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_classifier.predict(roi)[0]
            #pred - 7가지 감정 수치화
            emotion_probability = np.max(preds)
            #가장 큰감정
            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()


cam = VideoCamera()


def gen(camera):
    while True:
        frame = cam.get_frame()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


#@gzip.gzip_page
def streaming(request):
    try:
        return StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")
    except:  # This is bad! replace it with proper handling
        print('errer')