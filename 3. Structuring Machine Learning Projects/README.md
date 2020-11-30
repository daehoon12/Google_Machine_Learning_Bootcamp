# 3. Structuring Machine Learning Projects  

## 1. Introduction to ML Strategy  

### Orthogonalization  
- 주어진 벡터들을 이용해 서로 수직인 벡터를 만드는 방법.  
- In Marchine Learning, 원하는 효과를 위해 변수를 조정하는 과정에서 **하나의 버튼이 하나의 동작**만을 하게끔 만드는 것.  

#### TV tuning example
![image](https://user-images.githubusercontent.com/32921115/100571339-42ec1f80-3316-11eb-8368-16a84010a032.png)

TV를 튜닝할 수 있는 5개의 버튼과 다이얼이 있다.  
각각의 버튼과 다이얼은 채널, 볼륨, 화면 가로폭, 세로폭, 화면 회전각 조절 기능을 한다.  
만약 한 버튼이 채널, 볼륨 조절 기능을 하고, 다른 한 버튼이 가로폭, 세로폭, 회전각 조절 기능을 하면 tv를 원하는 대로 튜닝하는 것이 거의 불가능하다.  
- 다시 말해 이 예제에서 Orthogonalization이란 **각 버튼이 한 가지 일만 하도록 설계하는 것을 의미**한다.  
#### Chain of assumptions in ML  
- 머신러닝이 잘 운영되기 본인의 시스템을 튜닝해 4가지가 잘 유지되도록 해야함.  
	1. Training set에서 잘 작동 (작동 잘 안될 경우 network 크기나 adam으로 최적화)  
	2. Dev set에서도 잘 작동 (overfitting 해결방법이랑 같음 -> regulaization or bigger train set)  
	3. Test set에서도 잘 작동 (dev set을 과하게 튜닝했을 경우가 높으므로 bigger dev set), Overfitting 방지  
	4. Cost function이 적용된 Test set에서 잘 작동해야 함. (dev set을 바꾸거나 cost function을 바꿔야 함)  

- 각각의 단계에서 쓸 수 있는 **Solution들은 Orthogonalization** 해아 한다. 

### Single number evaluation metric  

#### Another cat classification example  
![image](https://user-images.githubusercontent.com/32921115/100572476-e9392480-3318-11eb-9289-347ec77e6068.png)
- 여기서 최적의 cat classifier를 선택하는 방법은 2가지가 있다.  
1. 선형적으로 두 척도를 계산   
  - Cost = (Accuracy) - 0.5*(running time)  
  - 너무 인위적이고 적합하지 않음.  
2. Optimizing and Satisficing Metrics (최적화와 조건 척도)  
  - Accuracy가 최대인 Classifier를 선택 (Optimizing Metric)  
  - 그리고 Running time이 100ms 보다 작은 것중에 고름. (Satisficing Metric)  
  - 최고의 Classifier를 선택하기 더 편해진다!


### Train/dev/test distributions  
![image](https://user-images.githubusercontent.com/32921115/100573474-3d450880-331b-11eb-922a-b1fb9426ffa7.png)
- 8개국이 Cat classfication을 만든다 가정.  
- US, UK, Europ, South America는 Dev 나머지는 test set -> dev set과 test set이 다른 분포도를 가짐.  
- 이렇게 나누지 말고 8개국이 모은 데이터를 무작위로 dev와 test set에 넣자.  

#### 결론
- Dev/Test Set은 같은 분포도를 가져야 한다.    
- 미래에 어떤 데이터를 반영 시킬지 생각해 그 데이터를 dev, test에 넣어라.  

### Size of dev and test sets  
- 데이터 사이즈가 작을 때 (10,000 이하) 일때는 train, test 비율이 (7:3)이 맞으나, 빅 데이터에선 더 이상 맞지 않음.  
- 보통 빅 데이터에선, train : dev : test 비율이 98:1:1이다.  
- Test Set의 목적은 최종 Cost를 평가하는 것이므로, 이 목적에 맞게 적당한 크기의 Test Set을 설정 해야한다.
