import dlib
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import time

detector = dlib.get_frontal_face_detector()
emotion_classifier = load_model('model_filter(best).h5', compile=False)
EMOTIONS = ["Angry" ,"Disgusting","Fearful", "Happy", "Sad", "Surpring", "Neutral"]

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

cap = cv2.VideoCapture(0)

    
    
while True:
    start = time.time()
    ret, frame = cap.read()
    # print(ret, frame)
# for i in range(1,31):
#     frame = cv2.imread(f'./dataset/User.1.{i}.jpg', cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)

    canvas = np.zeros((250, 300, 3), dtype="uint8")
    emotion_all = {}

    if len(faces) > 0:

        (fX, fY, fW, fH) = rect_to_bb(faces[0])

        roi = gray[(fY):fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48)) #얼굴 화면밖으로 나가면 error
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        

        preds = emotion_classifier.predict(roi)[0]

        emotion_probability = np.max(preds)

        label = EMOTIONS[preds.argmax()]

        cv2.putText(frame, label, (fX, fY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame, (fX, (fY-20)), (fX + fW, fY + fH + 10), (0, 0, 255), 2)
        
        negative = [0,1,2,4]
        positive = [3,5]

        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):

            text = "{}: {:.2f}%".format(emotion, prob * 100)

            if i in negative:
                emotion_all.update({emotion:prob*150})
            elif i in positive:
                emotion_all.update({emotion:prob*100})
            else:
                emotion_all.update({emotion:prob*100})

            

            w = int(prob * 300)  # 그래프길이
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

        # if cv2.waitKey(1) & 0xFF == ord('w'):
        #     print(emotion_all)
    # Open two windows
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
            emotion = 'nuetral'

        print(emotion_all)
        print(emotion)

    else:
        emotion_all.update({'face':None})
        print(emotion_all)
    # #얼굴인식 화면
    # print(frame)
    cv2.imshow('Emotion Recognition', frame)
    # # 감정상태 화면
    cv2.imshow("Probabilities", canvas)
    # break
    # q누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # end = time.time()
    # print(end-start)

# 종료
cap.release()
cv2.destroyAllWindows()