# Convolutional Neural Networks  

## 1. Foundations of Convolutional Neural Networks  

## Computer Vision  
- Marchine의 **시각에 해당하는 부분을 연구하는 컴퓨터 과학**의 최신 연구 분야 중 하나  
- Image Classification, Object Detection, Nerual Style Transfer 등 다양한 Computer Vision Problem 존재  

## Edge Detection 
- Image의 **가장자리를 탐지**  

## Example  

### 1) Vertical  

![image](https://user-images.githubusercontent.com/32921115/103525441-90821c00-4ec2-11eb-9ced-ada8a8126897.png)

- 가운데에 있는 Matric를 **Filter**라고 한다.  
- Output에서 흰색에 해당 되는 것이 Edge에 해당된다. (수직으로 검출)  

### 2) Horizontal  

![image](https://user-images.githubusercontent.com/32921115/103525753-1e5e0700-4ec3-11eb-9104-32eee801a87a.png)

**딥 러닝에서는 이 Filter를 Parameter W로 둔다 -> 즉 Backpropagation을 통해 최적의 값으로 변한다!!!!!!**  
**Filter Size ↓, 계산 속도 ↑, Learning 해야하는 Weight의 숫자가 줄어들기 때문!**  

## Padding  
- 코너나 모서리 쪽의 픽셀들은 Output에서 적게 사용 된다. 이는 **가장자리 쪽의 많은 정보를 버린다.**  
- 그래서 **Padding**을 이용해 외곽쪽을 채워 filter와 곱을 할때, 조금 더 사용할 수 있게 해준다.  

### Valid and Same Convolutions  
- Valid : No Padding  
- Same : Use Padding  

## Stride  
- Filter가 한 번에 얼마나 움직이는 지의 값  

## Summary of Convolutions  

![image](https://user-images.githubusercontent.com/32921115/103526299-0aff6b80-4ec4-11eb-8aa2-81ecbcbf66bb.png)

- 외우지 말고 유도할 것.  

### Technical note on cross-correlation vs. convolution  

![image](https://user-images.githubusercontent.com/32921115/103526518-65003100-4ec4-11eb-8a41-abe8eb3c4e8e.png)

- 위에서 했던 계산은 다 **cross-correlation이지만, 결합법칙 때문에 결과 값이 같으므로 Deep learning에서는 그냥 convolution이라 부른다.**  

## Convolutions over volumes  

![image](https://user-images.githubusercontent.com/32921115/103527007-0c7d6380-4ec5-11eb-8e1d-b02dce09e4e4.png)

- **이미지 채널의 개수는 필터의 채널 수와 일치해야 한다.**  

![image](https://user-images.githubusercontent.com/32921115/103527620-ff14a900-4ec5-11eb-8c14-bd8fddc59180.png)

- **Filter의 수만큼 Convolution 된 Output이 되고 곧 Channel이 된다.  
- ** Convolution Layer를 거치면서 width, hegiht는 줄고 Channal의 수는 느는 경향이 있다.  

### Types of Layer in convolutional network  
- Convolution  
- Pooling  
- Fully connected  

## Pooling Layers  
-표현되는 크기 ↓, 계산 속도 ↑, 감지 능력 ↑ 해주는 Layer  

### Max pooling  

![image](https://user-images.githubusercontent.com/32921115/103528070-b14c7080-4ec6-11eb-806e-f193ca58de82.png)

- Feature들의 Max값을 감지. 

### Average Pooling  

![image](https://user-images.githubusercontent.com/32921115/103528039-a396eb00-4ec6-11eb-959e-2cdd7df649cd.png)

- Feature들의 평균 값  

**보통 Max >Average** 

### Hyperparameters  
- f : filter size, s : stride, max or average pooling  
- 스스로 learning되지 않음.  

## Neual Network Example  

![image](https://user-images.githubusercontent.com/32921115/103528196-e062e200-4ec6-11eb-9f53-c324786ce753.png)

- 위의 아키텍처를 다 이해할 것.  

## 2. Deep convolutional models: case studies  


## Classic Network  

### 1) LeNet-5  

![image](https://user-images.githubusercontent.com/32921115/103610530-bdd2d680-4f63-11eb-87cd-73ea05c6ce5a.png)

- 손글씨를 분류하기 위해 만들어진 Network, **Input이 GreyScale**
- Yann Lecun과 그의 동료들이 1990년에 만든 최초의 CNN LeNet-1을 발명  
- 1998년 LeNet-5 발명  
- 2개의 Convolutional Layer와 2개의 pooling layer. 2개의 Fully Connected Layer, 0~9를 출력해주는 Softmax 함수  
- **Sigmoid** Activation Function 사용  

### 2) AlexNet  

![image](https://user-images.githubusercontent.com/32921115/103610886-739e2500-4f64-11eb-8903-566699ca9a5d.png)

- 2012년에 Krizhevsky에 발표된 Network
- **병렬 처리에 특화된 GPU**의 보급으로 대량 연산을 고속으로 수행이 가능하면서 AlexNet을 시작으로 Deep Learning이 주목을 받기 시작함.  

### LeNet과의 차이점  
- Relu 사용  
- LRN이라는 국소적 정규화를 실시하는 Layer 사용  
- Drop Out 사용
  
### GPU를 사용해 병렬 처리  

![image](https://user-images.githubusercontent.com/32921115/103610926-8add1280-4f64-11eb-8882-db410d694f1c.png)

### 3) VGG-16 (Visual Geometry Group 16 Weight layers)  

![image](https://user-images.githubusercontent.com/32921115/103611745-3d61a500-4f66-11eb-8fc7-796241eeb359.png)
- Karen Simonyan과 Andrew Zisserman에 의해 2015 ICLR에 게재  
- **네트워크의 깊이를 깊게 만드는 것**이 성능에 어떤 영향을 미치는지를 확인하고자 한 것
- Conv Layer 개수 + FC Layer 개수 = 16  
- CONV = 3 X 3 Filter, Stride = 1, Padding 사용, Max-Pooling = 2 X 2, Stride = 2  

## Residal Network (ResNets)  
- Vanishing, Exploding Gradient 때문에 매우 깊은 Layer에서의 Learning의 어려움 있음.  
- 이 문제를 해결하기 위해 **Skip Connection** 개념이 나옴.

### Skip Connection (Shortcut)  

![image](https://user-images.githubusercontent.com/32921115/103613213-76e7df80-4f69-11eb-8f29-d3745e2d4df9.png)

- 위 그림과 같이 A_l에 Nonliearlity (Relu)를 적용하기 전에, 더 깊은 Layer에 더해줌

### Residual Block  

![image](https://user-images.githubusercontent.com/32921115/103613052-13f64880-4f69-11eb-94b5-8e9330e600ac.png)

![image](https://user-images.githubusercontent.com/32921115/103614349-bb747a80-4f6b-11eb-8f82-2410a6bae26d.png)

- Direct로 Learning하는 것 대신 Skip Connection을 통해 **기존의 Learning 정보 보존해 Activation Function에 대입**  
- 수식이 **A_[l+2] = g(z_[l+2] + a_[l])** 로 바뀜  

### Plane vs. Res  

![image](https://user-images.githubusercontent.com/32921115/103614223-6df80d80-4f6b-11eb-90b3-a8458a1eee3f.png)

- Plane에서는 Layer가 깊어질수록 Optimization 알고리즘의 learning이 힘들어 Training Error가 커지게 됨.  
- Resnet에서는 training error가 점점 줄어든다.  

## Why ResNets work?  

![image](https://user-images.githubusercontent.com/32921115/103614590-46ee0b80-4f6c-11eb-8598-cf2cf1de9394.png)

1) a[l+2] = g(z[l+2] + a[l])에서 **z[l+2] = w[l+2] x a[l+1] + b[l_2]**을 대입  
2) 여기서 **L2 Regularization or Weight decay**를 적용하면 W[l+2]의 값이 점점 줄게 됨.  
3) W와 b의 값이 0으로 수렴하면서 z[l+2]가 0이 되고, g(a[l])에서 a[l]은 이미 Non-negative한 값이므로 **g(a[l]) = a[l]**이 됨.  
4) 따라서 **a[l+2]= a[l]** 성립하고 이 성질이 ResNet의 핵심이 된다.  

**Residual block의 input과 output의 Dimension은 같아야 한다.**  





