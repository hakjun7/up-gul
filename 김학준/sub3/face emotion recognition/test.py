import dlib
import numpy as np
import cv2

## face detector와 landmark predictor 정의
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)


# range는 끝값이 포함안됨   
ALL = list(range(0, 68)) 
RIGHT_EYEBROW = list(range(17, 22))  
LEFT_EYEBROW = list(range(22, 27))  
RIGHT_EYE = list(range(36, 42))  
LEFT_EYE = list(range(42, 48))  
NOSE = list(range(27, 36))  
MOUTH_OUTLINE = list(range(48, 61))  
MOUTH_INNER = list(range(61, 68)) 
JAWLINE = list(range(0, 17)) 

index = ALL

while True:
    ret, img_frame = cap.read()
    img_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
    dets = detector(img_gray, 1)
    for face in dets:
        shape = predictor(img_frame, face) #얼굴에서 68개 점 찾기
        list_points = []

        for p in shape.parts():
            list_points.append([p.x, p.y])

        list_points = np.array(list_points)

        for i,pt in enumerate(list_points[index]):
            pt_pos = (pt[0], pt[1])
            cv2.circle(img_frame, pt_pos, 2, (0, 255, 0), -1)
        
        cv2.rectangle(img_frame, (face.left(), face.top()), (face.right(), face.bottom()),
            (0, 0, 255), 3)

    cv2.imshow('result', img_frame)
    key = cv2.waitKey(1)

    if key == 27:
        break
    
    elif key == ord('1'):
        index = ALL
    elif key == ord('2'):
        index = LEFT_EYEBROW + RIGHT_EYEBROW
    elif key == ord('3'):
        index = LEFT_EYE + RIGHT_EYE
    elif key == ord('4'):
        index = NOSE
    elif key == ord('5'):
        index = MOUTH_OUTLINE+MOUTH_INNER
    elif key == ord('6'):
        index = JAWLINE

cap.release()

# ## 비디오 읽어오기
# video = cv2.VideoCapture(0)

# ## 각 frame마다 얼굴 찾고, landmark 찍기
# # while True:
# #     ret,frame = cap.read()
# #     gray = cv2.cvtColor(frame,cv2.COLOR_BAYER_BG2GRAY)

# #     faces = detector.    
# cap = video.read()

# for frame in cap:    
#     img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     r = 200. / img.shape[1]
#     dim = (200, int(img.shape[0] * r))    
#     resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#     rects = detector(resized, 1)
#     for i, rect in enumerate(rects):
#         l = rect.left()
#         t = rect.top()
#         b = rect.bottom()
#         r = rect.right()
#         shape = predictor(resized, rect)
#         for j in range(68):
#             x, y = shape.part(j).x, shape.part(j).y
#             cv2.circle(resized, (x, y), 1, (0, 0, 255), -1)
#         cv2.rectangle(resized, (l, t), (r, b), (0, 255, 0), 2)
#         cv2.imshow('frame', resized)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# cv2.destroyAllWindows()