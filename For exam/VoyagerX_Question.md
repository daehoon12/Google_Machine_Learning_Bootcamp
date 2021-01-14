# Q

## Gradient Desent란?  
- Loss Fucntion의 Loss를 줄이기 위한 Optimizer 최적화 방법 중 하나. 미분 값(기울기)이 최소가 되는 점을 찾아 알맞은 W와 b를 찾아냄  

### 과정  
1. W를 설정한다. (보통은 임의의 값, or Xhavier나 He initializaion을 써도 됨)  
2. W에서 Gradient를 계산한다.  
3. learning rate와 곱해 W를 조정한다.  

## Sigmoid의 단점은?  
- Gradient Vanishing 현상 발생 -> 함수의 끝으로 갈 수록 미분 값이 0으로 수렴함. -> Weight를 업데이트가 잘 안됨.  
- 함수값이 0~1 사이, 모든 x값이 같은 부호만 있다고 가정하면 backpropagation zigzag 현상이 발생  

![image](https://user-images.githubusercontent.com/32921115/104579027-c7b0b400-569e-11eb-98da-472d81419730.png)

참고 : https://reniew.github.io/12/

![image](https://user-images.githubusercontent.com/32921115/104573065-cfb92580-5697-11eb-9940-c2d18979bbe6.png)

- exp연산은 너무 큰 연산  

## Validation, Test Set의 각각 역할은?  
- Validation Set : Learning을 하지는 않지만 learning에 관여는 한다. epoch당 **learning 되지 않은 data를 통해**  train data와 성능을 비교한다. 그리고 그래프 같은 것 그려 어느 지점에서 가장 모델의 성능이 좋은지 찾을 수 있는 지표가 된다. 만약 **train data에서는 좋은 accuracy를 가지고 있으나, validation data에서는 accuracy가 낮으면 Overfitting 문제가 발생**한다는 것을 알 수 있다. 즉, Overfitting을 방지하고, learning되지 않은 data를 통해 Model의 성능을 평가하면서 적절한 hyperparameter, epochs를 찾기 위해 사용된다.  

- Test Set : learning이 다 된 Model의 **최종 성능을 평가하기 위해 사용.**  

## Auto Encoder란?  

## Dropout의 효과는?  
1. Voting 효과 : 무작위로 neuron을 삭제하고 learning을 반복하면, 모든 neuron들이 골구로 fitting이 되어 평균적으로 잘 예측해 어떤 데이터든지 다 분류할 수 있게 된다. 이를 Voting 효과라고 한다.  

2. Co-adaptation 회피 효과 : 특정 neuron의 weight, bias가 큰 값을 갖게 되면, 그 neuron의 영향이 커지면서 다른 neuron들이 제대로 learning을 할 수 없다. 쉽게 말하면 **두개 이상의 neuron이 하나의 neuron처럼 활동하는 것**이라고 생각하면 되는데, 이는 **2개 이상의 neuron이 동일한 feature만 구분할 것이고 효율은 떨어지면서 계산량만 증가하는 현상**이 발생할 것이다. dropout을 통해 learning을 하면서 뉴런들이 서로 동조화(co-adaptation)을 피하고, 각각의 neuron이 **다양한 feature를 감지하면서** 더욱 강건(robust)한 network를 구성할 수가 있다.  

- 마치 오랜 시간동안 지구상에 존재하는 생명체들이 유전자 복제가 아닌 양성 생식을 통해 유전자를 결합하고, **강인한 유전자들이 살아남는 것**과 마찬가지다.  

3. Ensemble 기법 (모든 Model의 평균값)과 비슷한 효과를 발휘함.  

## CNN의 장점은?  
- 기존 DNN의 문제점 : 기존 DNN에서의 이미지를 학습할 때의 Weight의 개수는 **input 차원 수  * layer node 개수**로 parameter가 증가하면 learning 시간 증가와 overfitting을 야기  

1. Parameter Sharing : 3x3 filter를 사용한다 가정할 때 filter 1개당 9개의 weight가 존재, 모든 input data는 이 filter를 사용함으로 동일한 parameter를 공유. **계산량이 현저하게 준다.**  

2. Sparse Connection : Parameter sharing을 통해 모든 image data를 넣는 것이 아니라, filter와 똑같은 size의 data를 input으로 넣어서 learning. **Overfitting 방지, neuron의 weight가 너무 커지는 것을 방지**  

![image](https://user-images.githubusercontent.com/32921115/104586661-1b27ff80-56a9-11eb-9139-c3db96246b2c.png)

## Word2Vec의 원리는?  

## Adam Optimizer의 동작은?  
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
- 모맨텀에서 가한 연산이 지수 가중 평균이기 때문에, **+, -로 진동을 하는 bias는 평균이 0에 가깝게 되어** 진동 폭이 준다.  
- 반면 **오른쪽으로 값이 있던 W의 평균은 큰 값을 가지게 되어 더욱 빨리 업데이트**가 된다.  
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
