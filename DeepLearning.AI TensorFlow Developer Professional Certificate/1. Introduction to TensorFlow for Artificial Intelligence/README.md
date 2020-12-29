## 개념 
- 일반적인 프로그래밍은, Rule과 Data를 통해 Answer를 찾는다. 반면 Machine Learning에서는 A**nswer와 Data를 통해 Rule을 찾는다.**  
- Dense : A layer of connected neurons, 만약, 입력 뉴런의 가중치가 4개이고, 출력 뉴런의 가중치가 8개라면 Dense Layer는 이를 4×8로 총 32개의 가중치를 만든다.  
- Loss Fucntion : 내 모델을 통해 생성된 결과 값과 실제로 발생하기를 원했던 값간의 차이를 계산하는 함수.  
(※ 컴퓨터는 기본적으로 값을 대입해서 문제를 풀어본다. 그래서 대입한 결과와 실제 정답간의 간격 즉, **차이를 최대한 줄이는 방향으로 값을 대입하게 된다.** 이 값의 차이를 **loss**라고 하며, **이 loss를 줄이는 방향으로 학습이 진행이 됨.**) 
- Optimizer : Deep Learning에서는 Learning 속도를 빠르고 안정적이게 하는 기법.  
- Convergence : 정답에 매우 근접하는 과정.  

## Keras  
![image](https://user-images.githubusercontent.com/32921115/103254375-6adbab00-49c8-11eb-91de-0fed40942092.png)  
- Sequential : Layer를 linear로 연결해 구성.  
- add() : 레이어를 추가  
- compile() : Learning 방식을 설정  
- fit() : Model을 학습  
- predict() : Test Data를 통해 Model을 output을 얻음.  

## Pipeline  
1. Dataset 생성  
  - Data를 불러오거나 시뮬레이션을 통해 데이터 생성  
  - Train, Validation, Test set 생성  
2. Model 구성  
  - Sequence Model을 생성한 뒤 필요한 Layer를 add()로 추가해 구성  
  - 복잡한 모델이 필요할 때는 케라스 함수 API를 사용  
3. 모델 학습과정 설정  
  - Loss fucntion, Optimizer, Callback 함수 정의  
  - compile() 메소드 사용  
4. Model 학습시키기  
  - Tarin set을 이용해 Model 학습  
  - fit() 메소드 사용  
5. 학습과정 살펴보기  
  -  Epoch마다 Loss, Acc 측정  
6. Model 평가하기  
  - Validation Set으로 학습한 Model을 평가  
  - evaluate()  사용  
7. Model 사용하기  
  - 임의의 입력으로 모델의 Output을 얻음   
  - predict() 메소드 사용  

