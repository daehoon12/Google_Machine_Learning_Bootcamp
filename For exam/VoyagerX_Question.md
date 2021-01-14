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

![image](https://user-images.githubusercontent.com/32921115/104573065-cfb92580-5697-11eb-9940-c2d18979bbe6.png)

- exp연산은 너무 큰 연산  

## Validation, Test Set의 각각 역할은?  

## Auto Encoder란?  
\frac{\partial{L}}{\partial{w}}=\frac{\partial{L}}{\partial{a}}\frac{\partial{a}}{\partial{w}}
