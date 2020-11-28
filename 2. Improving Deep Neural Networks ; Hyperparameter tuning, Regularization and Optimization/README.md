# 2. Improving Deep Neural Networks : Hyperparameter tuning, Regularization and Optimization  

## 1. Practical aspects of Deep Learning

### Normalization, Standardization, Regularization

#### Normalization  
![다운로드](https://user-images.githubusercontent.com/32921115/99683281-85665e80-2ac3-11eb-8b41-ea7d2be14304.png)

- 값의 범위(Scale)를 0~1 사이로 바꿔 데이터의 분포를 조정    
- 머신 러닝에서 **scale이 큰 feature의 영향이 비대해지는 것**을 방지  
- 딥 러닝에서 Local Minima에 빠질 위험 감소 (학습 속도 향상)  

#### Standardization  
![2304E84656B1B53A07](https://user-images.githubusercontent.com/32921115/99683358-99aa5b80-2ac3-11eb-88ed-e8cfa3d3aea6.png)

- 값의 범위(Scale)를 평균 0, 분산 1이 되도록 변환, 데이터의 분포를 조정  
- 머신 러닝에서 **scale이 큰 feature의 영향이 비대해지는 것**을 방지  
- 딥 러닝에서 Local Minima에 빠질 위험 감소 (학습 속도 향상)  
- 정규 분포를 표준정규분포로 변환하는 것과 같다.  

#### Regularization  
![다운로드](https://user-images.githubusercontent.com/32921115/99683950-408ef780-2ac4-11eb-9009-9f061020cfb1.png)
- Weight를 조정하는데에 제약을 거는 기법, 보통 Cost function에 L2 Norm을 더함
- Overfitting을 막기위해 사용  
- L1, L2 regularization 등 종류가 있음.

### Bias / Variance     
(사진)

- 첫 번째 경우 : Train set에서는 오류가 적고 Dev Set에서는 오류가 큼  
  -> Overfitting (High Variance)  

- 두 번째 경우 : 둘다 오류가 높음  
  -> Underfitting (High Bias)  

- 세 번째 경우 : Train Set에서는 15% Dev Set은 30%  
  -> 매우 최악 (오버피팅, 언더피팅 현상이 둘다 있음)  

- 네 번째 경우 : 둘다 1% 미만으로 낮음  
  -> low bias, low variance  

Human's Error : 0%  
Optimal(Base) Error : 거의 0% -> 만약 흐릿한 이미지, 사람도 잘 못맞추는 데이터일 경우 이 Error율이 올라간다.  

- 만약 High bias (Underfitting)인 경우, **Network를 더 크게 만들거나 학습을 더 길게한다.**    
- 만약 High Variance (Overfitting)인 경우, **더 많은 데이터를 수집**한다.  

### Regularization  
- Weight를 조정하는데에 제약을 거는 기법. Overfitting을 방지하는데 사용한다.
- 일반적으로 L2 (Norm 2) Regularization을 사용 (**Weight decay**)  

![9933B6495B79164D0E](https://user-images.githubusercontent.com/32921115/99686703-315d7900-2ac7-11eb-9571-e427871643a4.png)


#### 왜 Regularization을 하면 Overfitting이 방지가 될까?
- Tanh 함수를 예로 들면  
- 람다가 커짐 -> W 매개변수 작아짐 (Cost Function를 최소하하기 위해) -> Z값의 범위가 좁아짐 -> 함수가 선형인 함수처럼 변함 -> 덜 overfitting 해짐  

### Dropout Regularization   
- 뉴런의 연결을 임의로 삭제  
- 1 - keep_prob = hidden unit을 삭제할 확률  
- keep_prob은 각 층마다 설정 가능  

### Normalizing Input  
- 전체 구간을 0~1 사이의 값으로 맞춘다.

#### Why normailize Input?  
- 아래 사진과 같이 Unnormalized 한 그래프에서는 Learning Rate를 매우 작게 설정해야 정상적인 학습이 되는데 이유는 cost 그래프가 너무 길쭉 (elongaed) 하기 때문이다. 즉 Input의 Range가 서로 다르면 **경사 하강법**을 적용하는 것이 매우 까다롭다.  
- Normalization을 적용해 **Spherical**한 그래프를 만들자! 그러면 더 빠르게 경사 하강법으로 최적화 지점을 찾게 된다.  

### Vanishing/Exploding Gradients  
- Vanishing Gradient : Backprob시 Gradients가 너무 작아지는 현상  
- Exploding Gradient : Backprob시 Gradients가 너무 커지는 현상  
- 두 경우 다 학습이 제대로 이뤄지지 않는다.  

#### 해결책  
- Single Neuron에서 예를 들자  
- Z = w1x1 + w2x2 +... + WnXn에서 N이 커질 수록 Wi는 작아지는 것을 원한다  
- Wi의 편차를 1/n으로 설정  
- 만약 ReLu를 쓸 경우 편차를 2/n으로 설정하는 것이 더 작동을 잘함.  
- TanH를 이용하는 경우 root(1/n[l-1])를 사용하면 작동을 잘한다. (Xavier Intialization)
(사진)  

### Gradient Checking  
- Backprogation ALgorithm은 구현이 복잡하고 여러 방법으로 구현 가능  
- 이를 디버그 하기 위한 방법이 **Gradient Checking**  
- Gradient Checking은 계산할 때 굉장히 Expensive하기 때문에 학습할 때는 무조건 꺼야한다.  
- 파란 선은 실제 미분 했을 때 구해지는 접선의 기울기, **GradApprox 공식**을 이용해 구한 선과 근사하다. 
![687474703a2f2f692e696d6775722e636f6d2f5846556d384d472e6a7067](https://user-images.githubusercontent.com/32921115/99241947-df57f180-2841-11eb-818f-28615872c19c.jpg)

### Mini-batch gradient descent  
- 기존의 Gradient Decent의 문제점은 최적값을 찾을 때 마다 모든 data set을 넣어야 해서 학습하는데 시간이 오래 걸림.  
- Stochastic Gradient Descent : batch size를 1로 지정한다.  
![image](https://user-images.githubusercontent.com/32921115/100493729-354f6200-317d-11eb-9124-22cf151a4441.png)

- 이 2개의 장점을 섞은 기법이 **Mini-batch gradient**  
- Mini-batch size는 보통 2의 지수로 설정한다.  
![image](https://user-images.githubusercontent.com/32921115/100493758-8b240a00-317d-11eb-8827-f1dea27d554f.png)

#### batch를 정하는 방법  
- Training set가 작을 때 (m <=2000)이면 그냥 batch size로 학습하고, 이것보다 크면 2의 지수 (64,128,256,512)로 한다.

### Understanding exponentially weighted averages  
- 데이터의 이동 평균을 구할 때, 오래된 데이터가 미치는 영향을 지수적으로 감쇠(exponential decay) 하도록 만들어 주는 방법.  
- 손실함수의 최저점을 찾기 위한 최적화 알고리즘을 이해하기 위해서 필요한 사전 지식 중의 하나.  
![image](https://user-images.githubusercontent.com/32921115/100493952-f4a51800-317f-11eb-8a2d-f497a64c903b.png)
- 베타 : 0 ~ 1 사이의 값을 갖는 hyperparameter  
- 세타 : 새로 들어온 데이터  
- V : 현재의 경향을 나타내는 값  
![image](https://user-images.githubusercontent.com/32921115/100493963-169e9a80-3180-11eb-80e0-09163086b799.png)
- 이렇게 대입하면 베타는 더 작아지고 베타를 계수로 가진 Vn-2가 작아짐. -> **오래된 데이터일수록 현재의 경향을 표현하는데 더 적은 영향을 미친다.**  
- 이 V 값을 근사적으로 1/1-베타d일 간의 데이터를 사용해 평균을 취하는 것과 같음.  
ex) B = 0.98이라하면 위의 식에서 50이라는 값이 나오고, 이 50일 간의 데이터를 가지고 가중 평균을 구한 것. ![image](https://user-images.githubusercontent.com/32921115/100494002-96c50000-3180-11eb-8236-fc173a8421a9.png)
- 위의 그림에서 녹색선은 B가 0.98일 때 (50일간의 가중평균), 붉은선은 B가 0.9 (10일간의 가중평균)일 때다.  - 녹색은 50일간 데이터를 모두 반영하다보니 **최신의 동향을 반영하기보다는 과거의 데이터에 치중한 경향을 보이고**, 붉은색은 **상대적으로 최신의 데이터를 반영하므로 녹색보단 가볍게 최신 경향을 캐치한다 볼수 있음.**  

### Bias Correction in Exponentially Weighted Average  
- Gradient decent를 조금 더 빠르게 해주는 알고리즘  
![image](https://user-images.githubusercontent.com/32921115/100494063-2f5b8000-3181-11eb-9c6e-8b68b6162da5.png)
- 보라색 선을 보면 처음에 값이 매우 낮음. bias correction을 이용해 보라색의 낮은 위치를 초록선으로 올려줌

### Momentum Gradient Descent 
- 기존 Gradient Descent는 Global minimun이 아닌 local minimum에서 멈추는 경우가 있음.  
- 이 속도에 운동량을 곱해주는 것이 Momentum  
- dW와 db의 가중평균을 구해 learning rate에 곱한다.  
- 여기서 하이퍼파라미터의 베타는 보통 0.9 사용. 베타가 0이면 높은 마찰 저항, 1이면 마찰 저항 X  
- 왼쪽 공식 사용할 것.  
![image](https://user-images.githubusercontent.com/32921115/100494402-9a5a8600-3184-11eb-9f4c-af1fe2cd50d3.png)
