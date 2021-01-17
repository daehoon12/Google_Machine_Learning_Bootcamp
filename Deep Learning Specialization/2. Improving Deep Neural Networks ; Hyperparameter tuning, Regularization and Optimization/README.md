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

![image](https://user-images.githubusercontent.com/32921115/103342008-bfb41a00-4acb-11eb-9eb5-9cc49a81a3ad.png)


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
### Norm  
- Vector의 크기 (또는 길이)를 측정하는 방법. 두 벡터 사이의 거리를 측정하는 방법  

### L1 Norm  

![image](https://user-images.githubusercontent.com/32921115/104831594-c1ece580-58cd-11eb-8514-754453f307cb.png)

- 벡터 p,q의 **각 원소들의 차이의 절대값의 합.** (맨해튼 거리)  
- p=(3,1,-3), q=(5,0,7)이라면 p,q의 L1 Norm은 |3-5| + |1-0| + |-3-7| = 13이 됨.  

### L2 Norm  

![image](https://user-images.githubusercontent.com/32921115/104831612-f3fe4780-58cd-11eb-9685-cdcdbe39bc60.png)

- 직선의 거리 (유클리디안 거리)  
- p=(3,1,-3), q=(5,0,7)이라면 p,q의 L2 Norm은 (5-3)^2 + (1-0)^2 + (-3-7)^2에 루트를 씌운 값이 된다.  

### Regularization  
- Weight를 조정하는데에 제약을 거는 기법. Overfitting을 방지하는데 사용한다.
- 일반적으로 L2 (Norm 2) Regularization을 사용 (**Weight decay**)  

![image](https://user-images.githubusercontent.com/32921115/104831698-9a4a4d00-58ce-11eb-93c4-565a792c5594.png)

- 단순하게 loss fucntion의 값이 작아지는 방향으로 learning을 하면, 특정 weight가 너무 큰 값을 가지게 되어 **Overfitting 현상**이 발생한다.  

![image](https://user-images.githubusercontent.com/32921115/104831717-c2d24700-58ce-11eb-90fd-45ab0afe4746.png)

- Regularization을 통해 Overfitting을 방지한 모델  

### L1 Regularization  

![image](https://user-images.githubusercontent.com/32921115/104831883-fcf01880-58cf-11eb-94de-b4ad0cc077fa.png)

- L1 Regularization을 사용하는 regression model을 Lasso Regression이라고 한다.  

### L2 Regularization  

![image](https://user-images.githubusercontent.com/32921115/104831892-1abd7d80-58d0-11eb-8e5b-d2dd9c9b334b.png)

- L2 Regularization을 사용하는 regression model을 Ridge Regression이라고 한다.  

### 선택의 기준  
- 가중치 w 가 작아지도록 학습한 다는 것은 결국 Local noise 에 영향을 덜 받도록 하겠다는 것이며 이는 평균 데이터와 크게 다른 데이터 (Outlier)의 영향을 더 적게 받도록 하겠다는 것  

![image](https://user-images.githubusercontent.com/32921115/104831940-9d463d00-58d0-11eb-95c0-b7ad3b240163.png)

- 위의 그림에서 L1은 빨간 or 파란선을 사용해 특정 feature를 0으로 처리하는 것이 가능함. 즉 **Feature Selection**이 가능  
- 하지만 미분 불가능한 점이 있기에 Gradient-base learning에서 사용하지 않음. (L2를 많이 사용하는 이유)  

#### 왜 Regularization을 하면 Overfitting이 방지가 될까?
- Tanh 함수를 예로 들면  
- 람다가 커짐 -> Backpropagation 과정에서 Regularization 매개변수까지 빼 W의 값이 작아짐-> Z값의 범위가 좁아짐 -> 함수가 선형인 함수처럼 변함 -> 덜 overfitting 해짐  

### Dropout Regularization   
- 뉴런의 연결을 임의로 삭제해 Regularization 효과를 얻는 방법  
- 1 - keep_prob = hidden unit을 삭제할 확률  
- keep_prob은 각 층마다 설정 가능  

#### 효과  
1. Voting 효과 : 무작위로 neuron을 삭제하고 learning을 반복하면, 모든 neuron들이 골구로 fitting이 되어 평균적으로 잘 예측해 어떤 데이터든지 다 분류할 수 있게 된다. 이를 Voting 효과라고 한다.  

2. Co-adaptation 회피 효과 : 특정 neuron의 weight, bias가 큰 값을 갖게 되면, 그 neuron의 영향이 커지면서 다른 neuron들이 제대로 learning을 할 수 없다. 쉽게 말하면 **두개 이상의 neuron이 하나의 neuron처럼 활동하는 것**이라고 생각하면 되는데, 이는 **2개 이상의 neuron이 동일한 feature만 구분할 것이고 효율은 떨어지면서 계산량만 증가하는 현상**이 발생할 것이다. dropout을 통해 learning을 하면서 뉴런들이 서로 동조화(co-adaptation)을 피하고, 각각의 neuron이 **다양한 feature를 감지하면서** 더욱 강건(robust)한 network를 구성할 수가 있다.  

- 마치 오랜 시간동안 지구상에 존재하는 생명체들이 유전자 복제가 아닌 양성 생식을 통해 유전자를 결합하고, **강인한 유전자들이 살아남는 것**과 마찬가지다.  

3. Ensemble 기법 (모든 Model의 평균값)과 비슷한 효과를 발휘함.  
### Normalizing Input  

![image](https://user-images.githubusercontent.com/32921115/104832088-bbf90380-58d1-11eb-8f76-d375734431e1.png)

- 값의 Scale을 0~1 사이로 맞춰 데이터의 분포를 조정한다.  
- 데이터의 단위가 다르거나 (cm와 km를 비교할 때), 단위가 같더라도 값의 범위가 크게 차이나 비교하기 힘든 상황일 때 (1000점 만점에 90점 vs 100점 만점에 80점 비교), Normalization이나 Standardization을 사용하면 비교하기가 더 수월해진다.  

#### Why normailize Input?  

![image](https://user-images.githubusercontent.com/32921115/103342188-39e49e80-4acc-11eb-853d-5ae4179ebad6.png)

- 위 사진과 같이 Unnormalized 한 그래프에서는 Learning Rate를 매우 작게 설정해야 정상적인 학습이 되는데 이유는 cost 그래프가 너무 길쭉 (elongaed) 하기 때문이다. 즉 Input의 Range가 서로 다르면 **경사 하강법**을 적용하는 것이 매우 까다롭다.  
- Normalization을 적용해 **Spherical**한 그래프를 만들자! 그러면 더 빠르게 경사 하강법으로 최적화 지점을 찾게 된다.  

### Vanishing/Exploding Gradients  
- Vanishing Gradient : Backprob시 Gradients가 너무 작아지는 현상  
- Exploding Gradient : Backprob시 Gradients가 너무 커지는 현상  
- 두 경우 다 학습이 제대로 이뤄지지 않는다.  

#### 해결책  
- **초기에 W를 어떻게 초기화 할 것인가?**  
- Single Neuron에서 예를 들자  
- Z = w1x1 + w2x2 +... + WnXn에서 N이 커질 수록 Wi는 작아지는 것을 원한다  
- Wi의 편차를 1/n으로 설정  
- 만약 ReLu를 쓸 경우 편차를 2/n으로 설정하는 것이 더 작동을 잘함.  
- TanH를 이용하는 경우 root(1/n[l-1])를 사용하면 작동을 잘한다. (Xavier Intialization)
### Weight Initialization for Deep Networks  
N(in) : 이전 Node의 수, N(out) : 다음 Layer의 Node 수  

### 1. Xavier Intialization  

![image](https://user-images.githubusercontent.com/32921115/103344291-8383b800-4ad1-11eb-86a9-1c8cac4bafdc.png)

- TanH, Sigmoid Activation 함수를 사용할 때 사용하는 초기화 방법. (np.random.rand(shape) * Xavier)  

### 2. He Initialization

![image](https://user-images.githubusercontent.com/32921115/103344317-99917880-4ad1-11eb-829f-77af7685ea72.png)

- Relu 함수를 사용할 때 사용하는 Weight 초기화 방법.  

### Gradient Checking  
- Backprogation ALgorithm은 구현이 복잡하고 여러 방법으로 구현 가능  
- 이를 디버그 하기 위한 방법이 **Gradient Checking**  
- Gradient Checking은 계산할 때 굉장히 Expensive하기 때문에 학습할 때는 무조건 꺼야한다.  
- 파란 선은 실제 미분 했을 때 구해지는 접선의 기울기, **GradApprox 공식**을 이용해 구한 선과 근사하다. 
![687474703a2f2f692e696d6775722e636f6d2f5846556d384d472e6a7067](https://user-images.githubusercontent.com/32921115/99241947-df57f180-2841-11eb-818f-28615872c19c.jpg)

## 2. Optimization Algorithms  
### Gradient Descent  
- **Loss Fucntion의 Loss를 줄이기 위한 Optimizer 방법 중 하나.** 미분 값(기울기)이 최소가 되는 점을 찾아 알맞은 W와 b를 찾아냄  

### 과정  
1. W를 설정한다. (보통은 임의의 값, or activation 함수에 따라 Xhavier나 He initializaion을 써도 됨)  
2. W에서 Gradient를 계산한다.  
3. learning rate와 곱해 W를 조정한다.  

### 단점  
- 한번에 모든 데이터 셋을 사용해 학습 속도가 느림.  

### Stochastic Gradient Descent  
- Batch Size가 1인 Gradient Descent  
- Iteration이 dataset의 개수  
- 하나의 Data가 무작위로 선택되어 learning  

### 단점  
- 노이즈가 매우 심하다.  

### Mini-batch Gradient Descent  
- 일반적으로 2의 n승으로 선택  
- SDG의 노이즈를 줄이면서 전체 batch보다 효율적  

### 셋의 공통적인 단점  
- Local Optima, Plateau에 빠지기 쉬움.  

### Local Optima  

![image](https://user-images.githubusercontent.com/32921115/104686614-e8c9e100-5740-11eb-8e87-91e42c3f1c61.png)

- 위의 Loss function을 보면 최소 지점에 도달하지 않았음에도 불구하고, 어느 지점에 들어가니까 평평하고, 더 진행하려보니까 loss가 늘어나므로 w를 멈추게 된다. 이러한 지점을 **local optima(minumum)**이라 한다.

### Plateau  

![image](https://user-images.githubusercontent.com/32921115/104686978-ace34b80-5741-11eb-8ea7-24114c9760e0.png)

- Global Optima를 향해 나아가는데, 평지가 생겨서 더 이상 loss가 업데이트 되지 않는 현상  

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
- 여기서 하이퍼파라미터의 베타는 보통 0.9 사용. 베타가 0이면 마찰 저항X , 1이면 높은 마찰 저항  
- 왼쪽 공식 사용할 것.  
![image](https://user-images.githubusercontent.com/32921115/100494402-9a5a8600-3184-11eb-9f4c-af1fe2cd50d3.png)

### RMSprop
![image](https://user-images.githubusercontent.com/32921115/100494549-2ae59600-3186-11eb-9091-81b4c1aab6b0.png)
- W 방향으로 learning이 빨리 되기를 바람.  
- W는 크게, b는 작게 만들어야 learning이 빨리 된다.  
- 즉 dw의 값을 작게, db의 값을 크게 만들어 학습을 하면 된다.

### Adam  
![image](https://user-images.githubusercontent.com/32921115/100494634-bc550800-3186-11eb-9179-8de3fa15693e.png)
- Momentum과 RMSProb을 합침. 성능이 제일 좋으므로 Gradient Optimization 알고리즘을 선택할 때 항상 Default로 할 것!  
- 기존 RMSprop과 Momentum 방식에 Bias 보정식을 사용헀다.  
- Hyperparameter는 보통 B1 (Momentum)에 0.9, B2 (RMSProp)에 0.999, 입실론은 10^-8을 함.

### Learning rate decay (학습 속도 감소법)
![image](https://user-images.githubusercontent.com/32921115/100494745-f2df5280-3187-11eb-9e8c-e0b3f255cc57.png)
- 학습 초기에는 큰 스텝으로 움직이고, 학습이 점점 다 되면 학습 속도를 느려지게 한다.  

**수식**  
![image](https://user-images.githubusercontent.com/32921115/100494769-38038480-3188-11eb-9ea9-f1119ac3cf20.png)
ex) 알파0이 0.2일 때  
![image](https://user-images.githubusercontent.com/32921115/100494779-5b2e3400-3188-11eb-8e8e-971cda3026ef.png)

## 3. Hyperparameter tuning, Batch Normalization and Programming Frameworks  

## 1) Hyperparameter Tuning  

### Tuning Process  
- 하이퍼파라미터 튜닝 중 가장 중요한 것 : learning rate (알파) > hidden units, mini-batch size, 베타 > layers, learning rate decay  

### Hyperparameters tuning in practice : Pandas vs. Caviar  

![image](https://user-images.githubusercontent.com/32921115/103517033-d46e2480-4eb4-11eb-9095-ead4eac30684.png)  

## 2) Batch Normalization  
 
### Batch Normalization  

![image](https://user-images.githubusercontent.com/32921115/104681037-b5815500-5734-11eb-95de-9b28d1e4d467.png)


- **Mini Batch의 평균과 분산**을 이용해서 Normalization 한 뒤, Scale 및 shift 를 감마(γ) 값, 베타(β) 값을 통해 실행. (감마와 베타는 실행 가능한 변수), **Backpropagation**을 통해 Learning 됨.  
- 기존 output = g(Z), Z = WX + b 식은 output = g(BN(Z)), Z = WX + b, 즉 기존 **Activation Function의 Input에 Batch Normalization**을 적용한다.  
- 입실론 값은 계산할 때 0으로 나눠지는 문제가 발생하는 것을 막기위한 값, 감마는 Scale 값, 베타는 Shift transform에 대한 값이다. 이들은 **데이터를 계속 Normalization을 할 때, Activation이 Non-linear한 성질을 잃게 되는 것을 막아준다.**  
- 감마와 베타도 learning 값이므로, **Optimizer 기법 사용이 가능하다.**  

### Why does Batch Norm work?  
### 1) Covariate Shift 방지  

![image](https://user-images.githubusercontent.com/32921115/103521312-e0111980-4ebb-11eb-8be9-acc7734b915e.png)

- Covariate Shift : 이전 Layer의 파라미터 변화로 인해 현재 **Layer의 분포가 바뀌는 현상**  
- **Data의 분포가 바뀌면?** -> 우리가 찾아야하는 분포를 제대로 찾기 어려워짐.  
### 2) Gradient Vanishing/ Exploding 방지  
- 기존 DNN에서는 learning rate를 너무 높게 잡을 경우, Gradient가 explode/vanish하거나 local optima 현상에 빠지는 경우가 있었다. 이는 **Parameter의 scale 때문인데, batch normalization을 사용할 경우 propagation할 때 parameter의 scale에 영향을 받지 않는다.** 즉 **learning rate를 크게 잡을 수 있고 이는 빠른 learning이 가능하다!**

### 3) Regularization 효과  
- 미니 배치마다 평균과 표준편차를 계산하여 사용하므로 training data에 일종의 잡음 주입 효과로 Overfitting 방지하는 효과 발생 (Dropout 효과와 비슷함)  

## 3) Multi-class classfication  

![image](https://user-images.githubusercontent.com/32921115/103522203-60844a00-4ebd-11eb-97bc-7fc6f38c3fa3.png)  

### Softmax Function  

![image](https://user-images.githubusercontent.com/32921115/103523146-cb825080-4ebe-11eb-9f2b-a8850c8b852d.png)

- Neuron Network의 output layer에서 사용하는 Activation Function, **Output의 총 합이 1이며** **Multi-class classfication에서 사용**  

### Loss Function  

![image](https://user-images.githubusercontent.com/32921115/103523522-4f3c3d00-4ebf-11eb-9d47-809f5497f9c4.png)

- **Categorical Cross-Entropy Loss 사용**  

### Cross Entropy Example  

![image](https://user-images.githubusercontent.com/32921115/103523464-3a5fa980-4ebf-11eb-9a32-8426eb6bcbb5.png)
