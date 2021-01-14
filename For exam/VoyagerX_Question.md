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
- local optima 현상을 막기 위해 
- dW와 db의 가중평균을 구해서 learning rate에 곱함.  
- 알파는 하이퍼파라미터, 베타는 마찰 저항  
