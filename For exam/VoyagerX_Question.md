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
