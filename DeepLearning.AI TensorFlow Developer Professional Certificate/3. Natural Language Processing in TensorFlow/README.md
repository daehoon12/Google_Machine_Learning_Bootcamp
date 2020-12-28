# 개념  
## ImageGenerator  
   1. Training을 위해 이미지를 쉽게 로드할 수 있는 기능  
   2. Train Image 크기를 선택할 수 있는 기능  
   3. Directory 이름에 따라 이미지에 자동으로 레이블을 지정할 수 있는 기능  
   4. Image Augmentation 기능  
   
   
- Train Data와 Validation Data의 Acc가 차이가 많이 나면 **Overfitting**을 의심해야한다.  
- Q. Overffiting이 작은 Dataset에 많이 나타나는 이유는?  
- A. Training 중 모든 feature에 대한 likelihood가 적기 때문이다.  

## Image Augmentation 
   1. 이미지 crop, 좌우 반전, size 조절 등 다양한 기법을 통해 원래 data의 양을 늘림.  
   2. Image 처리에 Cycle이 필요해 Training 속도가 약간 느려짐.  
   3. Augmentation data는 **Memory**에 저장됨.  
 
## Dropout  
- Model을 regularize하는 방법 중 하나. Hidden Layer에서 드롭아웃(dropout)을 확률 p로 적용하는 경우, Hidden Unit들을 p 확률로 제거하는 것이 됨.  

### Dropout을 하는 이유  
- 랜덤으로 Unit을 비활성화 시켜서 모든 Unit이 골구로 Learning 될 수 있게 해줌.  
- 어떤 Unit들 너무 학습이 되어 Weight나 Bias가 커지면 다른 Unit들의 learning에 문제가 생기는 현상을 막아줌.  
- 어떤 뉴런의 이웃 뉴런들은 비슷한 weight를 가질 수 있고, 그래서 마지막 training을 왜곡할 수 있기 때문이다.  
## Quiz  
- Train Data와 Validation Data의 Acc가 차이가 많이 나면 **Overfitting**을 의심해야한다.  
- Overffiting이 작은 Dataset에 많이 나타나는 이유는, Training 중 **모든 feature에 대한 likelihood가 적기 때문**이다.  
- Transfer learning Model에서도 Image Augmentation 사용 가능  
- DropOut이 너무 높으면, **network의 특성을 잃어, 비효율적인 learning**이 일어나기도 한다.  
