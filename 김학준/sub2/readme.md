### emotion training

아직



### 얼굴인식 구현

트레이닝 된 모델 받아서 구현

`test.py`

문제점

- 표정의 변화가 커야 인식이 잘됨

  - 68개점이 학습된 모델(점으로 얼굴구조 찾는법) 하면 인식률이 좋아질까?

    https://github.com/sunsmiling/facial-emotion-detector

  - 트레이닝 모델

    https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat

- 사람마다 표정변화가 다르기때문에 인식률 떨어짐
  
  - 맨처음 얼굴(무표정)을 기준으로 잡을 필요가 있을것같음 (How??)

다른 emotion 트레이닝 모델(아직 안해봄)

https://github.com/omar178/Emotion-recognition#p4





### django live video streaming

test.py 코드를 저번 프로젝트때 django streaming구현했던코드 `django_streaming(iot pjt).py` 참고하면서 코딩,수정중



표정수치 데이터만 받을수 있으면 되는지?

비디오 스트리밍 필요가 있나?