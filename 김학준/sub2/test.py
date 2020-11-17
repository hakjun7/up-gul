import cv2
import numpy as np   
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# 얼굴 인식모델, 감정분석 트레이닝 모델 불러오기
face_detection = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
emotion_classifier = load_model('emotion/emotion_model.hdf5', compile=False)

EMOTIONS = ["Angry" ,"Disgusting","Fearful", "Happy", "Sad", "Surpring", "Neutral"]

# 캠
camera = cv2.VideoCapture(0)
camera.set(3,640)
camera.set(4,480)

while True:
    # Capture image from camera
    ret, frame = camera.read()
    # 그레이스케일
    # Grayscale은 1차원인 0~255만 따지게 되므로 연산량이 대폭 감소한다.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detection.detectMultiScale(gray,
                                            scaleFactor=1.1, # 이미지에서 얼굴 크기가 서로 다른 것을 보상해주는값
                                            #각 이미지 척도에서 이미지 크기를 줄이는 방법을 지정하는 매개 변수.(모르겠당)
                                            minNeighbors=5, # 얼굴 사이의 최소 간격(픽셀)
                                            minSize=(30,30)) # 얼굴의 최소 크기
    
    # 빈 창 생성 (그래프띄울 창)
    canvas = np.zeros((250, 300, 3), dtype="uint8")

    # Perform emotion recognition only when face is detected
    if len(faces) > 0: # 얼굴 감지됬을때
        # For the largest image(얼굴 2개잡혔을때 더 큰얼굴)
        face = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        #print(face)
        (fX, fY, fW, fH) = face
        # Resize the image to 48x48 for neural network
        roi = gray[fY:fY + fH, fX:fX + fW] #얼굴자르기
        roi = cv2.resize(roi, (48, 48)) # 얼굴자이즈 48*48
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # Emotion predict
        preds = emotion_classifier.predict(roi)[0]
        #pred - 7가지 감정 수치화
        emotion_probability = np.max(preds)
        # 가장 큰 감정
        label = EMOTIONS[preds.argmax()]
        
        # Assign labeling
        cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
 
        # Label printing
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            #emotion - 감정 prob - 수치(백분율)
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            #print(text)
            
            w = int(prob * 300)  # 그래프길이
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    # Open two windows
    
    # 얼굴인식 화면
    cv2.imshow('Emotion Recognition', frame)
    # 감정상태 화면
    cv2.imshow("Probabilities", canvas)
    
    # q누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 종료
camera.release()
cv2.destroyAllWindows()