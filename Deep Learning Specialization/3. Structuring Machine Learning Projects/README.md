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
	3. Test set에서도 잘 작동 (dev set을 과하게 튜닝했을 경우가 높으므로 bigger dev set) 
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

### When to change dev/test sets and metrics  

#### Cat dataset examples
![image](https://user-images.githubusercontent.com/32921115/100574913-5dc29200-331e-11eb-8de5-ce512c3ff98c.png)
- A는 3%의 error가 발생함과 동시에 pornographic 이미지를 보여줌  
- B는 5%의 error가 발생하지만, A같은 상황은 발생하지 않음.  
- 이 예에서 Metric과 Dev Set은 A를 더 선호하지만, User들은 포르노 이미지를 걸러낼 수 있는 B를 더 선호함.  

#### Orthogonalization for cat pictures : anit-porn  
1. evaluate classifiers에 대한 metric을 정의하는 방법을 찾는다. (목표를 겨냥) 
2. 이 metric이 어떻게 해야 성능이 좋아지는지 생각한다. (목표를 실행)

#### Another example  
![image](https://user-images.githubusercontent.com/32921115/100575797-366cc480-3320-11eb-8a6f-e18db3a1c6a3.png)
- 이 알고리즘을 토대로 제품을 만들면 B 알고리즘이 더 뛰어난 성능을 발휘하는 것처럼 보여지는 상황.  
- dev set은 고화질의 고프레임 사진이고 user의 사진은 다양한 고양이 사진 (저프레임, 흐릿한, 저화질 등)  
- dev set 또는 dev와 test set에서만 잘 작동하므로 **Metric과 dev/test set을 바꿔야 한다.**

### Why human-level performance?  

#### Comparing to human-level performance
![image](https://user-images.githubusercontent.com/32921115/100576170-e80bf580-3320-11eb-89e3-f3565ac1b78e.png)
- human-level performance보다 accuracy가 높아지면 증가속도가 점점 줄어든다.  
- accuracy는 절대 bayes optimal error에 도달 할수 없다.  
※ Bayes Optimal Error : 성능은 이론적인 한계에 절대 도달하지 못하는 경우.  

#### Why compare to human-level performance    
- 인간은 사물 판단, 언어 읽기 등 자연적인 데이터의 업무를 기계보다 잘 처리한다.  
- 만약에 ML이 인간보다 못한 Performance를 보여준다면  
   1) 인간을 통해 label을 얻고  
   2) 알고리즘이 틀린 부분을 인간은 어떻게 맞췄고 알고리즘은 왜 틀렸는지에 대한 insight를 얻는다.  
   3) 더 나은 bias/variance를 분석한다.  
   
### Avoidable bias  

#### 복습
![image](https://user-images.githubusercontent.com/32921115/100576616-d2e39680-3321-11eb-9969-35ab31611400.png)  

#### Cat classification example  
![image](https://user-images.githubusercontent.com/32921115/100576778-28b83e80-3322-11eb-87eb-0ba1b5015fd8.png)
- Avoidable Bias : Bayes Error와 Training Error의 차이  
- 위의 그림에서 왼쪽의 avoidable bias는 7%, 오른쪽은 0.5%다. 즉 왼쪽 사례가 avoidable bias를 줄이면 더 큰 효과를 볼 수 있다.

### Understanding human-level performance  
![image](https://user-images.githubusercontent.com/32921115/100577146-f0653000-3322-11eb-8ecb-b2ed217b54ab.png)  
- Bayes error로 지정할 "Human-level error는 **error가 가장 낮은 것**으로 골라야 한다.
#### Error analysis example  
![image](https://user-images.githubusercontent.com/32921115/100577348-679ac400-3323-11eb-8617-2998a58af9e1.png)
- 첫 번째 경우는 Bayes Error와 Training Error간 차이가 크므로 bias에 중점을 둔다.  
- 두 번째 경우는 Train Error와 dev error간 차이가 크므로 Variance에 중점을 둔다.  
- 세 번째 경우에서는 인간의 성능이랑 비슷해 high bias와 high variance 효과를 제거하기 어렵다.  

#### 요약  
![image](https://user-images.githubusercontent.com/32921115/100577256-2a363680-3323-11eb-9f3d-53f0b5bd7800.png)
- human level error를 알면 bayes error를 정의할 수 있고, 이후 알고리즘에서 bias나 variance를 줄일 것인지 알 수 있다.  

### Improving your model performance  

#### Assumptions  
![image](https://user-images.githubusercontent.com/32921115/100578020-c3198180-3324-11eb-80c4-5e1bda91804e.png)
- 1번 경우는 낮은 Avoidable bias를 얻을 수 있다.  
- 2번 경우는 Variance가 나쁘지 않다는 것을 알 수 있다.

#### Reducing bias and variance  
![image](https://user-images.githubusercontent.com/32921115/100578119-f52ae380-3324-11eb-933f-660242298b6e.png)

