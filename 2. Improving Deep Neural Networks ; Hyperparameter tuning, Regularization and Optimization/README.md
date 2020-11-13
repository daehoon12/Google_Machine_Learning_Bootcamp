# 2. Improving Deep Neural Networks : Hyperparameter tuning, Regularization and Optimization  

## 1. Practical aspects of Deep Learning

### Normalization, Standardization, Regularization

#### Normalization  
- 값의 범위(Scale)를 0~1 사이로 바꾸는 것  
- 머신 러닝에서 **scale이 큰 feature의 영향이 비대해지는 것**을 방지  
- 딥 러닝에서 Local Minima에 빠질 위험 감소 (학습 속도 향상)  

#### Standardization  
- 값의 범위(Scale)를 평균 0, 분산 1이 되도록 변환  
- 머신 러닝에서 **scale이 큰 feature의 영향이 비대해지는 것**을 방지  
- 딥 러닝에서 Local Minima에 빠질 위험 감소 (학습 속도 향상)  
- 정규 분포를 표준정규분포로 변환하는 것과 같다.  

#### Regularization  
- Weight를 조정하는데에 제약을 거는 기법  
- Overfitting을 막기위해 사용  
- L1, L2 regularization 등 종류가 있음.

### Bias / Variance     
(사진)

- 첫 번째 경우 : Train set에서는 오류가 적고 Dev Set에서는 오류가 큼  
  -> Overfitting (High Variance)  

- 두 번째 경우 : 둘다 오류가 높음  
  -> Underfitting (High Bias)  

- 세 번째 경우 : Train Set에서는 15% Dev Set은 30%  
  -> 매우 최악 (오버피팅, 언더피팅 현상이 둘다 있음)  

- 네 번째 경우 : 둘다 1% 미만으로 낮음  
  -> low bias, low variance  

Human's Error : 0%  
Optimal(Base) Error : 거의 0% -> 만약 흐릿한 이미지, 사람도 잘 못맞추는 데이터일 경우 이 Error율이 올라간다.  

- 만약 High bias (Underfitting)인 경우, **Network를 더 크게 만들거나 학습을 더 길게한다.**    
- 만약 High Variance (Overfitting)인 경우, **더 많은 데이터를 수집**한다.  

### Regularization  
- Weight를 조정하는데에 제약을 거는 기법. Overfitting을 방지하는데 사용한다.
- 일반적으로 L2 (Norm 2) Regularization을 사용 (Weight decay)  
(사진)
#### 왜 Regularization을 하면 Overfitting이 방지가 될까?
- Tanh 함수를 예로 들면  
- 람다가 커짐 -> W 매개변수 작아짐 (Cost Function를 최소하하기 위해) -> Z값의 범위가 좁아짐 -> 함수가 선형인 함수처럼 변함 -> 덜 overfitting 해짐  
