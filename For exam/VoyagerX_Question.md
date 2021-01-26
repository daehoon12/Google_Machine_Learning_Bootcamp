# Q

## 1. Gradient Descent란?  
  - **Loss Fucntion의 Loss를 줄이기 위한 Optimizer 방법 중 하나.** 미분 값(기울기)을 통해 값이 제일 낮은 곳을 찾아 알맞은 W와 b를 찾아냄  

### 과정  
1. W를 설정한다. (보통은 임의의 값, or activation 함수에 따라 Xhavier나 He initializaion을 써도 됨)  
2. W에서 Gradient를 계산한다.  
3. learning rate와 곱해 W를 조정한다.  

### 단점  
- 한번에 모든 데이터 셋을 사용해 학습 속도가 느림.  

## Stochastic Gradient Descent  
- Batch Size가 1인 Gradient Descent  
- Iteration이 dataset의 개수  
- 하나의 Data가 무작위로 선택되어 learning  

### 단점  
- 노이즈가 매우 심하다.  

## Mini-batch Gradient Descent  
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

## 2. Sigmoid의 단점은?  
- Gradient Vanishing 현상 발생 -> 함수의 끝으로 갈 수록 미분 값이 0으로 수렴함. -> Weight를 업데이트가 잘 안됨.  
- 함수값이 0~1 사이, 모든 x값이 같은 부호만 있다고 가정하면 backpropagation zigzag 현상이 발생  

![image](https://user-images.githubusercontent.com/32921115/104579027-c7b0b400-569e-11eb-98da-472d81419730.png)

참고 : https://reniew.github.io/12/

![image](https://user-images.githubusercontent.com/32921115/104573065-cfb92580-5697-11eb-9940-c2d18979bbe6.png)

- exp연산은 너무 큰 연산  

## 3. Validation, Test Set의 각각 역할은?  
- Validation Set : Learning을 하지는 않지만 learning에 관여는 한다. epoch당 **learning 되지 않은 data를 통해**  train data와 성능을 비교한다. 그리고 그래프 같은 것 그려 어느 지점에서 가장 모델의 성능이 좋은지 찾을 수 있는 지표가 된다. 만약 **train data에서는 좋은 accuracy를 가지고 있으나, validation data에서는 accuracy가 낮으면 Overfitting 문제가 발생**한다는 것을 알 수 있다. 즉, Overfitting을 방지하고, learning되지 않은 data를 통해 Model의 성능을 평가하면서 적절한 hyperparameter, epochs를 찾기 위해 사용된다.  

- Test Set : learning이 다 된 Model의 **최종 성능을 평가하기 위해 사용.**  

## 4. Auto Encoder란?  
- **비지도 방식**으로 **효율적인 데이터 압축을 학습**하는 데 사용되는 인공신경망의 한 종류이다.

## 나오게 된 계기  
- 더 큰 input vector일 수록 많은 파라미터를 가짐.  
- but learning data에 굶주림도 증가해, 충분한 양의 데이터가 없으면 overfitting이 일어남.  
- convolutional 구조로 모델의 parameter를 줄이긴 했지만, 대량의 label이 필요한 데이터를 요구함. 이 데이터들은 희소하고 만드는데 비싸다.  
- 즉, Unsupervised 방식인 Embedding learning or 저차원 표현으로 이 문제를 해결  

## 원리 및 아키텍처  

![image](https://user-images.githubusercontent.com/32921115/104697759-9219d280-5753-11eb-80d4-351ea4763694.png)

- Encoder : 데이터 입력을 받고 **이를 저차원 벡터로 압축한다. (일종의 손실 압축)**, 중요한 feature를 최대한 보존하는 것을 목표로 하며, 둔감한 feature는 손실시킴.  
- Decoder : 임의의 레이블에 임베딩을 맞추는 대신 계산을 반전해 원래 입력으로 재구성  
- 입력과 출력이 동일함.  

## 효과  
- 데이터 압축 : feature의 수가 줄어 dimension이 줄어 memory가 적어짐.  
- curse of dimensionality 회피 : 데이터의 dimension이 증가할 수록 모델 추청에 필요한 data의 개수가 기하급수적으로 증가하는데, feature 수를 줄여줌으로써, 데이터의 dimension을 감소시킴.  
- 

## 5. Dropout의 효과는?  
1. Voting 효과 : 무작위로 neuron을 삭제하고 learning을 반복하면, 모든 neuron들이 골구로 fitting이 되어 평균적으로 잘 예측해 어떤 데이터든지 다 분류할 수 있게 된다. 이를 Voting 효과라고 한다.  

2. Co-adaptation 회피 효과 : 특정 neuron의 weight, bias가 큰 값을 갖게 되면, 그 neuron의 영향이 커지면서 다른 neuron들이 제대로 learning을 할 수 없다. 쉽게 말하면 **두개 이상의 neuron이 하나의 neuron처럼 활동하는 것**이라고 생각하면 되는데, 이는 **2개 이상의 neuron이 동일한 feature만 구분할 것이고 효율은 떨어지면서 계산량만 증가하는 현상**이 발생할 것이다. dropout을 통해 learning을 하면서 뉴런들이 서로 동조화(co-adaptation)을 피하고, 각각의 neuron이 **다양한 feature를 감지하면서** 더욱 강건(robust)한 network를 구성할 수가 있다.  

- 마치 오랜 시간동안 지구상에 존재하는 생명체들이 유전자 복제가 아닌 양성 생식을 통해 유전자를 결합하고, **강인한 유전자들이 살아남는 것**과 마찬가지다.  

3. Ensemble 기법 (모든 Model의 평균값)과 비슷한 효과를 발휘함.  

## 6. CNN의 장점은?  
- 기존 DNN의 문제점 : 기존 DNN에서의 이미지를 학습할 때의 Weight의 개수는 **input 차원 수  * layer node 개수**로 parameter가 증가하면 learning 시간 증가와 overfitting을 야기  

1. Parameter Sharing : 3x3 filter를 사용한다 가정할 때 filter 1개당 9개의 weight가 존재, 모든 input data는 이 filter를 사용함으로 동일한 parameter를 공유. **계산량이 현저하게 준다.**  

2. Sparse Connection : Parameter sharing을 통해 모든 image data를 넣는 것이 아니라, filter와 똑같은 size의 data를 input으로 넣어서 learning. **Overfitting 방지, neuron의 weight가 너무 커지는 것을 방지**  

![image](https://user-images.githubusercontent.com/32921115/104586661-1b27ff80-56a9-11eb-9139-c3db96246b2c.png)

## 7. Word2Vec의 원리는?  
- Word2Vec : word embedding을 생성하기 위한 프레임워크, CBOW 모델, Skip-Gram 모델이 있음.  

### 1. CBOW Model  
- 전체 Context에서 Embedding을 만들고 Target 단어를 예측하기 위해 Encoder를 사용. 

### 2. Skip-Gram Model  
- 중심이 되는 단어를 무작위로 선택하고 주변 단어를 예측.  
- 중심 단어가 Context(input)이 되고 주변 단어를 선택해서 Target(prediction)이 되도록 하는 **superivised learning**   
- example : "the boy went to the bank." 라는 문장을 (문맥, target) 순서로 분리하면 ([the, went], boy), ([boy, to], boy), ([went, the], boy), ([to, bank], boy)로 나뉜다. 여기서 input은 target output은 문맥의 단어중 하나다. 첫 번째 쌍에서 두 개의 데이터 (boy, the), (boy, went)를 생성.  

### 3. Encoder  
- dictionary가 V개의 단어를 가지고 있다면, 인코더는 V개의 row를 가진 Lookup Table, 여기서 i번째 행은 i번째 어휘 단어에 대응하는 embedding vector.  
- 즉 Lookup Table (코세라 강의에서는 Embedding Matrix라고도 함)과 i번째의 one-hot vector의의 내적을 통해 Embedding Vector를 얻어 단어를 예측할 수 있다.  
- Lookup Table은 초기에 무작위로 초기화되고 우리가 learning해야 하는 것이 Lookup Table이 된다.  

### Word Embedding  
- word를 사용자가 지정한 크기만큼 숫자로 된 vector로 만듬. 특정한 Label이 있는 것이 아니라 Unsupervised Learning 방법      
- one-hot representation의 단점은 **단어의 개수가 늘어나면 vector의 dimension도 늘어나고**, 각 단어를 하나의 Object로 여기기 때문에 단어 간의 관계를 추론할 수 없는 단점이 존재.  
- **단어 간 유사도를 반영할 수 있도록**하기 위해 word를 Vector로 바꾸어주는 Algorithm.    
- 유사도를 체크하는 방법으로 각 단어의 Cosine Similarity 방식을 사용하는데, Embedding Vector를 통해 유사성을 체크함.(One hot vector는 항상 값이 0이 나옴) 

## 8. Adam Optimizer의 동작은?  
- Momentum + RMSProp  

### Momentum  

![image](https://user-images.githubusercontent.com/32921115/104603734-0d31a900-56c0-11eb-9b0c-3c5383b44954.png)

- 파란선 : 일반 mini-batch gradient desent 방법, **진동**을 하면서 빨간 점에 도달  
- 보라선 : learning rate를 높게 주면 진폭이 커짐  
- 빨간선 : momentum을 이용해 진폭은 줄이고, 오른쪽으로는 더 빨리 이동함  

### 동작  
- 먼저 각 iteration에 대한 dW, dB를 구한다.  
- 지수 가중 평균을 구함  

![image](https://user-images.githubusercontent.com/32921115/104604348-9943d080-56c0-11eb-9b53-d02975723db4.png)

![image](https://user-images.githubusercontent.com/32921115/104604414-ae206400-56c0-11eb-94a4-6767e093a1a6.png)

- 이 값으로 weight와 bias를 업데이트 한다.  
- 모맨텀에서 가한 연산이 지수 가중 평균이기 때문에, **+, -로 진동을 하는 수직값는 평균이 0에 가깝게 되어** 진동 폭이 준다.  
- 반대로 **수평 값은 한쪽으로만 가 서로의 부호가 같아 값이 커진다** (Momentum이라고 하는 이유)  
- 모맨텀을 물리적으로 바라보면 등고선 형태는 밥그릇 같은 모양이 되고 빨간 점은 밥그릇의 가장 아래 가운데 부분이 됨.  
- Vdw, Vdb는 공이 밥그릇 아래로 내려가는 속도라 보면, dW, db는 속도의 변화율인 가속도로 해석  
- 베타는 0과 1 사이의 값이므로 속도를 줄어들게 하는 **마찰력**이라고 볼 수 있다.  

![image](https://user-images.githubusercontent.com/32921115/104605131-749c2880-56c1-11eb-9fc1-6c51e95dd12c.png)

- 정리하면 모멘텀은 **그래디언트 디센트에 지수가중평균 개념을 도입**하여 어떤 축으로 진동하는 값을 줄여주어서 학습 속도를 빠르게 하고 빠르게 이동해야 하는 축으로는 좀 더 빠르게 이동할 수 있도록 만들어 주는 방법이라고 볼 수 있습니다.
### RMSProp (Root Mean Square Prop)  
![image](https://user-images.githubusercontent.com/32921115/104601765-f722e900-56bd-11eb-9de7-5eef09401277.png)

- 현재 파란색 선으로 이루어진 그래디언트 디센트에서는 수직축으로의 이동 속도를 늦추고 수평축으로의 이동 속도를 빠르게 한다면 그래디언트 디센트 문제를 개선할 수 있음.  
- 각 Iteration 마다 배치에 대한 dw, db를 계산  
- 마찬가지로 가중평균을 이용함. 대신 미분계수를 제곱함. **도함수의 제곱을 지수 가중 평균**  

![image](https://user-images.githubusercontent.com/32921115/104602020-4a953700-56be-11eb-8308-9694fa39aeef.png)  

![image](https://user-images.githubusercontent.com/32921115/104602047-5123ae80-56be-11eb-9e8a-7cb386ccdfab.png)  

![image](https://user-images.githubusercontent.com/32921115/104602089-5f71ca80-56be-11eb-8f72-12da5f28dc36.png)

- 위의 예에서 **w의 변화를 키우기 위해 Sdw는 작아져야 하고, b의 변화량을 줄이기 위해 Sdb는 커져야 한다.**  
- 실제로 도함수 값에 제곱을 해 **상대적으로 큰 값을 가지는 미분 값은 크지만 나눠지는 값도 커져 변화량이 줄고, 도함수 값이 작으면 나누는 값이 작아져 변화량이 커지게 된다.**  
- 초록 선과 같이 수직 방향의 진동을 억제해 learning을 빠르게 함.  
- Sdw 제곱근으로 나뉘기 때문에 learning rate를 조금 크게 잡아도 됨 -> **learning speed up!**  
- **0으로 나뉘지 않도록** 분모에 입실론을 더해줌.  
- 정리하면 RMSProp은 **모멘텀과 같이 진동을 줄이는 효과**도 있고 **더 큰 러닝 레이트를 사용하여 학습 속도를 증가**시킬 수 있다는 장점이 있습니다.
### Adam  
- Momentum + RMSProp  
- 성능이 이미 증명되어 Default로 쓰면 좋음  

### 동작  
- Momentum과 RMSProp에서 다룬 것과 같이 현재 배치를 대상으로 **dw, db**를 구함.  
- 똑같이 모멘텀과 RMSProp에서 가중 평균을 구한것처럼 Vdw, Vdb, Sdw, Sdb를 구한다.  
- Bias Correction을 사용한다.  

![image](https://user-images.githubusercontent.com/32921115/104603423-bb891e80-56bf-11eb-8c67-11b5d0adf94e.png)

![image](https://user-images.githubusercontent.com/32921115/104603476-cc399480-56bf-11eb-9152-ef75cf5b1f50.png)

-최종으로 업데이트 되는 식  

![image](https://user-images.githubusercontent.com/32921115/104603550-dfe4fb00-56bf-11eb-9b05-da911d712e5c.png)


## 9. Batch Normalization의 동작은?    

![image](https://user-images.githubusercontent.com/32921115/104681037-b5815500-5734-11eb-95de-9b28d1e4d467.png)


- **Mini Batch의 평균과 분산**을 이용해서 Normalization 한 뒤, Scale 및 shift 를 감마(γ) 값, 베타(β) 값을 통해 실행. (감마와 베타는 실행 가능한 변수), **Backpropagation**을 통해 Learning 됨.  
- 기존 output = g(Z), Z = WX + b 식은 output = g(BN(Z)), Z = WX + b, 즉 기존 **Activation Function의 Input에 Batch Normalization**을 적용한다.  
- 입실론 값은 계산할 때 0으로 나눠지는 문제가 발생하는 것을 막기위한 값, 감마는 Scale 값, 베타는 Shift transform에 대한 값이다. 이들은 **데이터를 계속 Normalization을 할 때, Activation이 Non-linear한 성질을 잃게 되는 것을 막아준다.** -> 시그모이드를 예로 들면 N(0,1)일 경우 95%의 input은 1.96<= x <=1.96 사이인데, 이 부분은 sigmoid의 linear한 부분이다.  
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

## 10. Cycle GAN이란?  

### 나오게 된 배경  
- Image-to-image traslation은 짝이 있는 Image Training Set를 이용해 Input과 Output 이미지를 매핑하는 것이 목표인 Computer Vision 분야 중 하나. But, 짝이 지어진 Training Data를 얻는 것은 쉽지 않고 만들기도 어렵다.  
- **짝 지어진 Data없이** X라는 도메인으로부터 얻은 이미지를 Target인 Y로 바꾸면 어떨까? 해서 나온 Model  

### 원리  
1. Domain X를 Domain Y로 바꿔주는 **Translator G : X->Y를 GAN**으로 구현.  
  - pair dataset이 필요없이 X를 넣었을 때, Y를 생성할 수 있음.  
  - But, X 분포와 Y 분포에 대응하는 관계가 매우 많기 때문에 x와 y가 pair인지 보장할 수 없음 (풍경사진을 넣을 때 그 사진과 관련없는 모네 그림이 나올 수 있음)  
  
2. 위의 단점을 개선시키기 위해 **Cycle Consistent**라는 개념을 이용  
  - Cycle Consistent : Domain X에서 Y로 바꿨을 때, **다시 Y에서 translate 할 때 X가 나오는** 일관성.  
  - 반대 방향으로 도메인을 바꿔주는 translator F :Y -> X를 정의  
  
3. G(x)에서 나오는 Output을 y' F(y)에서 나오는 output을 x'라 할 때, **F(y') = x, G(x') = y**를 만족하는 방향으로 learning을 한다.  

4. 점점 **G(x) = y'은 y와 F(y) = x'는 x와 근사**하게 된다.  

5. 이 과정에서 GAN 2개가 Cycle 구조로 사용되기 때문에 **CycleGAN으로 불린다.**  
