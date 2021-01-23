# Neural Networks and Deep Learning  
  
## Logistic Regression
- Regression (회귀) : 입력에 대한 출력값이 연속적인 실수값으로 출력  
- y =wx + b 인 선형 식을 Activation 함수에 대입해 Non-Linear하게 만듬.  
- **수치가 아닌 확률로 출력**하고 싶을 때 사용. Linear Regression 사용 시 확률이 1이 넘어가는 현상 발생.  
- **Binary Classification**을 구현하는 알고리즘  
- Sigmoid 함수 사용  

**그림으로 보는 차이점**  
  
![image](https://user-images.githubusercontent.com/32921115/103337379-7c9f7a00-4abe-11eb-940b-f31894b99e37.png)

## Loss Function  
- Binary CrossEntropy 함수 사용  
- 아래 식은 이해하고 그려서 외우기  

![image](https://user-images.githubusercontent.com/32921115/103338324-62b36680-4ac1-11eb-8fc7-8ec2b138b095.png)

## Cost Function  
- 각 Train Set의 Loss Fucntion의 평균값  

## Backpropagation  

![image](https://user-images.githubusercontent.com/32921115/105575767-45687400-5db1-11eb-8b26-1d446c97ad9f.png)  

- L = -log(a) (y가 0일 때) 

![image](https://user-images.githubusercontent.com/32921115/105575772-50bb9f80-5db1-11eb-8e0b-93b3970e3dd3.png)  

- L = -log(1-a) (y가 1일 때)  

- 위의 함수를 미분  

dL = -1/a (y가 0일 때) -> 항상 음수 -> 오른쪽으로 감.
dL = 1/(1-a) (y가 1일 때)  -> 항상 양수 -> 왼쪽으로 감.

## Gradient Descent  
- 최적의 Weight, bias 값을 찾기위한 Optimizer 방법

### 원리  

![image](https://user-images.githubusercontent.com/32921115/103338564-1ddbff80-4ac2-11eb-9d78-5591235d7453.png)

- 위의 그래프는 W와 Cost의 상관관계  
- Backpropagation에서 W와 B의 값에 미분 값을 더해줌.  
- 기울기가 음수일 경우 W가 늘고, 양수일 경우 W가 감소한다.   
- 휴리스틱한 방식을 사용하기 때문에 Local Minumum을 가질 위험이 있음.  

## Activation Functions  

### 1. Sigmoid

![image](https://user-images.githubusercontent.com/32921115/103338807-ec176880-4ac2-11eb-987d-f5befdfd9a0e.png)

- Logistic 함수라 불리기도 한다.  
- 선형인 멀티 퍼셉트론에서 Non linear한 값을 얻기 위해 사용  
- 중간 값이 0.5, [0,1] 사이의 값 

**단점**  
- Gradient Vanishing 현상 발생 : input 값이 일정 끝으로 가면 미분 값이 0으로 수렴해 업데이트가 잘 되지 않는 현상  
- 함수값 중심이 0이 아님 : 학습이 느려짐, x를 양수라 가정했을 때, gradient 값은 항상 양수가 나온다. 즉 **같은 방향으로 update**가 되는데, 학습을 지그재그 형태로 만들어 느리게 만드는 원인이 된다.  

**Graph**  

![image](https://user-images.githubusercontent.com/32921115/103339353-85934a00-4ac4-11eb-952e-55436570b208.png)

### 2. Tanh

![image](https://user-images.githubusercontent.com/32921115/103339082-c8085700-4ac3-11eb-9ff3-d07a0529dadf.png)

- 모든 면에서 Sigmoid보다 월등  
- 함수의 중심값을 0으로 옮겨 sigmoid의 단점을 해결  
- 여전히 gradient vanishing 문제는 남아있음  

**Graph**  

![image](https://user-images.githubusercontent.com/32921115/103339318-6bf20280-4ac4-11eb-9220-5129db496d6e.png)

### 3. ReLu  

![image](https://user-images.githubusercontent.com/32921115/103339178-0dc51f80-4ac4-11eb-9bf0-9e5b9e6a90cc.png)

- 가장 많이 사용되는 활성화 함수  
- x > 0 이면 기울기가 1, else 기울기 0  
- tanH, sigmoid 함수와 비교시 학습이 훨씬 빨라짐 
- x가 음수인 값들에 대해서는 뉴런이 죽을 수 있는 단점이 존재  

**Graph**  

![image](https://user-images.githubusercontent.com/32921115/103339288-5250bb00-4ac4-11eb-9eb0-189892e7f306.png)

### 4. Leakly Relu  

![image](https://user-images.githubusercontent.com/32921115/103339270-4107ae80-4ac4-11eb-81ee-f798e37cd1cf.png)

- Relu의 단점을 해결하기 위해 나온 함수.  

## Getting your matrix dimensions right  

![image](https://user-images.githubusercontent.com/32921115/103340668-0273f300-4ac8-11eb-97b5-b06aa37dad7a.png)

- W의 Dimension = (사용자가 지정한 Unit의 개수, 이전의 input의 개수)  
- b의 Dimension = (사용자가 지정한 Unit의 개수, 1)  
- layer의 Parameter 수 = 위의 행렬값 곱한것들을 다 더한다.  

