# Neural Networks and Deep Learning  
  
## Logistic Regression
- Regression (회귀) : 입력에 대한 출력값이 연속적인 실수값으로 출력  
- y =wx + b 인 선형 식을 Activation 함수에 대입해 Non-Linear하게 만듬.  
- **수치가 아닌 확률로 출력**하고 싶을 때 사용. Linear Regression 사용 시 확률이 1이 넘어가는 현상 발생.  
- **Binary Classification**을 구현하는 알고리즘  

**그림으로 보는 차이점**  
  
![image](https://user-images.githubusercontent.com/32921115/103337379-7c9f7a00-4abe-11eb-940b-f31894b99e37.png)

## Loss Function  
- Binary CrossEntropy 함수 사용  
- 아래 식은 이해하고 그려서 외우기  

![image](https://user-images.githubusercontent.com/32921115/103338324-62b36680-4ac1-11eb-8fc7-8ec2b138b095.png)

## Cost Function  
- 각 Train Set의 Loss Fucntion의 평균값  

## Gradient Descent  
- 최적의 Weight, bias 값을 찾기위한 Optimizer 방법

### 원리  

![image](https://user-images.githubusercontent.com/32921115/103338564-1ddbff80-4ac2-11eb-9d78-5591235d7453.png)

- 위의 그래프는 W와 Cost의 상관관계  
- Backpropagation에서 W와 B의 값에 미분 값을 더해줌.  
- 기울기가 음수일 경우 W가 늘고, 양수일 경우 W가 감소한다.  
- Cost가 최소 = 기울기가 0  
- 휴리스틱한 방식을 사용하기 때문에 Local Minumum을 가질 위험이 있음.  

## Activation Functions  
1. Sigmoid
