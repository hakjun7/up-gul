

## 명세

- 기본
  - 머신 러닝 방법 중 하나인 **신경망 네트워크**(Neural Network)
  - 이 중, 심층 신경망 네트워크 (Deep Neural Network) 사용하여 이미지 처리, 자연어 처리
  - 이미지 + 텍스트 => 이미지 캡셔닝
    - 이미지 입력 들어오면 이미지 모델로 특성 뽑고 텍스트 모델에 전달, 이미지 묘사 텍스트 생성
    - sub3에서 이를 사용
  - 키워드
    - Detection, Object Tracking, Video Captioning, Machine Translation, Text Summarization 등



- sub-2

  - CNN (Convolutional Neural Networks) 컨볼루션 신경망
    - 이미지 데이터에 적합
    - 물제 형태 인지, 색깔 구별 등 특성 뽑기
  - RNN (Recurrent Neural Networks) 순환 신경망
    - 순서가 있는 데이터에 적합
    - 이미지에서 뽑아낸 특성으로 문장을 생성
  - CNN + RNN => 이미지 캡셔닝

  

## Deep learning

#### 코드

```
https://github.com/hunkim/DeepLearningZeroToAll/tree/master/tf2
```



#### TF로 Linear regression 구현 단계

![KakaoTalk_20200909_011640309](sub-2%20%ED%95%99%EC%8A%B5.assets/KakaoTalk_20200909_011640309.jpg)

![KakaoTalk_20200909_011640309_01](sub-2%20%ED%95%99%EC%8A%B5.assets/KakaoTalk_20200909_011640309_01.jpg)



#### Linear Regression

![LinR](sub-2%20%ED%95%99%EC%8A%B5.assets/LinR.jpg)



#### Logistic Regression

![logR](sub-2%20%ED%95%99%EC%8A%B5.assets/logR.jpg)



#### Softmax Regression

![SoftR](sub-2%20%ED%95%99%EC%8A%B5.assets/SoftR.jpg)