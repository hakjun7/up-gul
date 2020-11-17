# 없굴

*외모를 보지 않고 대화하고 싶은 우리를 위해서 만들었습니다.*

팀명 : 자급자족

주제 : AI 감정 분석을 이용한 블라인드 만남



## 소개

- ### 주요 기술

  - Facial Expression Detection (CNN, Convolutional Neural Network)
    - dlib, keras
    - Ref. https://www.kaggle.com/shawon10/facial-expression-detection-cnn
  - WebRTC (Web Real-Time Communications)
    - firebase, cloud firestore
    - Ref. https://webrtc.org/getting-started/firebase-rtc-codelab

- ### 배포

  - firebase, cloud firestore, AWS EC2 (Ubuntu 18.04), Nginx, Gunicorn
  - https://j3b307.p.ssafy.io/
  - https://up-gul.web.app/



## 파일 구조

#### backend

- 감정 분석용 서버
- Django rest framework
- AWS ec2 배포 (branch release)
- 개발 환경 실행 방법
  - venv 생성
  - pip install -r requirements.txt 
  - python manage.py runserver



#### frontend

- Vue, Vuex
- firebase 연동
  - backend, DB, deploy
- firebase 배포 (branch develop)
- 개발 환경 실행 방법
  - npm i
  - npm run serve
  - 감정 분석 이용시, api 주소 -> localhost:8000 변경



#### [멤버 이름 ]디렉토리

- 개개인별 프로젝트 진행에 필요한 학습 정리
- AI, 웹 기술, 인프라 등 관련 내용



#### 기획 디렉토리

- 프로젝트 회의록, 아이디어, 기능 명세, ERD, 와이어프레임

