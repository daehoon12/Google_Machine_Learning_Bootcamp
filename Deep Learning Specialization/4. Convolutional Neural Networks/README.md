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

## Why Convolutions?  
- 기존 DNN의 문제점 : 기존 DNN에서의 이미지를 학습할 때의 Weight의 개수는 **input 차원 수  * layer node 개수**로 parameter가 증가하면 learning 시간 증가와 overfitting을 야기  

1. Parameter Sharing : 3x3 filter를 사용한다 가정할 때 filter 1개당 9개의 weight가 존재, 모든 input data는 이 filter를 사용함으로 동일한 parameter를 공유. **계산량이 현저하게 준다.**  

2. Sparsity of Connection : Parameter sharing을 통해 모든 image data를 넣는 것이 아니라, filter와 똑같은 size의 data를 input으로 넣어서 learning. **Overfitting 방지, neuron의 weight가 너무 커지는 것을 방지**  

![image](https://user-images.githubusercontent.com/32921115/104586661-1b27ff80-56a9-11eb-9139-c3db96246b2c.png)


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

## Network in Newtork and 1x1 Convolutions  

![image](https://user-images.githubusercontent.com/32921115/103617326-6b98b200-4f71-11eb-8a64-5befeb359f6f.png)

- non-liearity 성질(Relu)을 더하면서 output의 volume을 증가/유지/감소 할 수 있음. 

### Using 1x1 Convolutions  

![image](https://user-images.githubusercontent.com/32921115/103617286-5a4fa580-4f71-11eb-83c2-7ae381bb22a7.png)

- height와 width를 줄이려면 pooling layer를 사용, 그러면 channel을 줄이고 싶으면?  
- 이때 1x1 convolution 사용. 
- 28x28x192의 값이 filter를 거쳐 28x28x32로 변함. width, height는 유지되고 volume은 축소됨.  
- 똑같이 channel을 줄일 수 있는 **padding ='same', stride = 1, filter 3x3**과의 차이는 Network의 연산량에서 차이가 난다.  

## Inception Network  

### Inception Module  

![image](https://user-images.githubusercontent.com/32921115/103621338-65f29a80-4f78-11eb-895a-4cd95551769f.png)

- 3x3, 5x5 filter를 통과하기 전 1x1 filter를 통과하여 channel의 수를 줄여 연산량을 줄임  
- output은 마지막 layer의 채널의 개수의 합 (64 + 128 + 32 +32)  

### GoogleNet  

![image](https://user-images.githubusercontent.com/32921115/103621530-b2d67100-4f78-11eb-9b84-828f5be57918.png)


## 3. Object Detection  

## Object Localization  

### What are localization and detection?  
- Image Classification : 특정 Algorithm을 이용해 Classfication 하는 것. Car라고 정답을 출력  
- Classfication with localization : Image Classification + bounding box Object 표시 -> Car라고 예측 + car의 위치까지 표시  
- Detection : Multiple Object를 Classfication 및 Localization  

### Classfication with localization  

![image](https://user-images.githubusercontent.com/32921115/103628324-2f218200-4f82-11eb-9b19-7416e2797d4c.png)

- Classfication Problem : Image를 Multiple-layer ConvNet에 Feed하고 마지막 Layer에서 softmax 함수 적용후 output vector를 통해 prediction 가능  
- bx, by, bh, bw는 위의 이미지에서 빨간색 Bounding box를 나타냄. (by, bx)는 bounding box의 중점, bh, bw는 각각 높이와 너비  
- bounding box 좌표 4개를 받고 **Supervised Learning 알고리즘을 통해 Classfication과 Localization** 수행 가능  
- 위의 예에서 Car의 bounding box는 (bx, by, bh, bw) = (0.5, 0.7, 0.3, 0.4). by는 y축의 좌표가 아래쪽에 있으므로 0.5보다 큰 값  

### Defining the target label y  

![image](https://user-images.githubusercontent.com/32921115/103628258-16b16780-4f82-11eb-89f9-3a9d1054e734.png)

- target label y = [Pc, bx, by, bh, bw, c1, c2, c3]  
- Pc (Probability of Class) : Class가 Object인가 아닌가? **Pc =1 -> Object 존재, Pc = 0 -> Object는 Background**  
- bx, by, bh, bw : Bounding Box의 좌표  
- c1,c2,c3 : 각 Class (위의 예에서는 Pedstrian, car, motorcycle)  
- 위의 예에서 왼쪽 사진은 Car를 분류해서 c2=1이고, 오른쪽 사진은 Pc=0이므로 나머지 요소들이 다 ? 이다.  
- Loss Function은 Squared Error 사용 (Cross Entropy도 가능), Pc = 1일 때는 모든 Y의 원소에 대해 squared Error를 구하고, 0일 때는 Pc값 즉 y1 원소에 대해서만 구한다.  

## Landmark Detection  
- Landmark : Image 내의 중요 부분이라 판단되는 좌표 Y  

### Example  

![image](https://user-images.githubusercontent.com/32921115/103629775-1619d080-4f84-11eb-8062-5c47c60e839f.png)

- 사람 얼굴이 ConvNet을 거쳐 129개의 output unit을 가짐 (1개는 face or not, 나머지는 128개는 landmark 좌표)  
- landmark의 개수는 지정해줘야 한다.  
- Train set에 Landmark 정보가 있어야 한다.  
- Pose Detection에서는 Key Position을 Landmark로 잡아서 구함. (위의 예에서는 어깨 팔꿈치 등)  

landmark detection을 위해서는 **모든 Training set에 대하여 좌표가 일관성 있게 준비**되어야 한다.  
예를 들어 1번 training image에 1번 좌표가 사람 왼쪽 눈의 왼쪽 코너라고 하면 2번 image의 1번 좌표 또한 왼쪽 눈의 왼쪽 코너여야 한다.  

## Object Detection  

### Car detection example  

![image](https://user-images.githubusercontent.com/32921115/103630609-431ab300-4f85-11eb-8bbb-357c8e98d2ad.png)

- x = 정답인 자동차만 crop한 이미지  
- y = 이미지가 자동차인지 아닌지에 대한 값  
- Object는 이미지 중앙에 위치 and 이미지 전체를 차지  
- Car (1) or not (0)  

### Sliding windows etection  

![image](https://user-images.githubusercontent.com/32921115/103630966-c2a88200-4f85-11eb-8d28-7964693e4abf.png)

- 왼쪽 이미지 상단부터 window size만큼 이미지를 Learning된 ConvNet에 Feed  
- Computational Cost가 크다.  
- 작은 stride, 작은 window size -> cost 증가  
- 높은 stride, window size -> 성능 저하  

## Convolutional Implementation of sliding windows  

### Turning FC layer into convolutional layers  

![image](https://user-images.githubusercontent.com/32921115/103638772-b544c500-4f90-11eb-971e-ff894639ffc6.png)

- 기존의 vector 였던 FC를 convolutinal 하게 바꿈  
- 5X5X16을 vector로 변환 x fully connected 개수 (400x400), 5X5X16 volume에 1x1x16인 filter 400개 convolution  (5x5x16x400) -> 둘다 같다.  

### Convolution implementation of sliding windows  

![image](https://user-images.githubusercontent.com/32921115/103638912-e7562700-4f90-11eb-8317-79e207162fc4.png)

- 14x14x3 train image를 16x16x3 test image에 적용, stride=2로 적용하면 2x2x4의 output을 얻음.  

![image](https://user-images.githubusercontent.com/32921115/103639148-45830a00-4f91-11eb-8a26-9b80121e1c46.png)

- 16x16x3 image에서 4번의 sliding window를 가지게 되어 4개의 label을 갖게 됨.  

![image](https://user-images.githubusercontent.com/32921115/103639281-78c59900-4f91-11eb-9984-b9c80a03cfed.png)

- test image = 28 x 28 x 3, window size = 14 x 14 x 3  
- 왼쪽 상단부터 오른쪽 하단까지 sliding window를 통하여 탐색  
- 이 때 window에 맞게 object가 있으면 Neural Network를 이용해 Detection  

## Bounding Box Predictions  

![image](https://user-images.githubusercontent.com/32921115/103640314-21c0c380-4f93-11eb-9ba7-269e84371f04.png)

- 위의 사진처럼 정확히 일치하는 box가 없어 Detection을 못하는 경우 좀 더 정확하게 할수 있는 방법이 있을까?  
- **YOLO Algorithm**  

### YOLO Algorithm  

![image](https://user-images.githubusercontent.com/32921115/103641131-99432280-4f94-11eb-898a-15bd8cd22e0d.png)

- 100 x 100 input image가 있다고 가정, 위에서는 3 x 3 grid를 사용하나, 실제는 더 정교한 grid를 사용함.  
- 각각의 grid에서 Object가 없는 보라색은 Pc = 0, 있는 나머지 색깔은 Pc =1  
- 위의 3x3 grid를 이용한 YOLO 알고리즘을 기준으로 3 x 3 x 8 output volume가 필요, Y가 8개의 원소를 가지기 때문.  
- 위의 예제에서 X는 100 x 100 x 3의 Image. input을 **ConvNet을 이용해 처리**하고 마지막 output을 grid와 똑같은 size인 3 x 3 x 8을 갖도록 Network를 구성해 Training 해야함. (실제는 19 x 19 x 8과 같이 만드는 것이 좋음)  
- grid의 개수가 많아질 수록 한 개의 grid에서 여러개의 object가 검출될 확률이 줄어든다.  
- Input 하나를 ConvNet에 적용하고 출력을 grid로 나누어 한번에 Object에 대한 Classification 과 Localization을 함 -> **처리 속도가 빨라서 real time object**에 적용 가능  

### Specify the bounding boxes  

![image](https://user-images.githubusercontent.com/32921115/103641471-24bcb380-4f95-11eb-9104-5462dcc5dd32.png)

- grid 내의 midpoint를 기준으로 height와 width로 그려짐  
- 각 grid의 좌측 상단 (0,0), 우측 하단 (1,1)로 정의함.  
- bh, bw는 1보다 커질 수 있다. -> 큰 Object가 여러 grid에 걸칠 경우  

## Intersection over union  

![image](https://user-images.githubusercontent.com/32921115/103641971-deb41f80-4f95-11eb-8e81-9e706cf73c2b.png)

- Prediction과 Label 각각의 Bounding Box가 얼마나 유사한지 확인하는 기준  
- IOU : **(보라색 ∩ 빨간색) / (보라색 ∪ 빨간색 - 보라색 ∩ 빨간색)**  
- 일반적으로 0.5보다 크면 예측이 Correct한 것으로 판정, 좀 더 정확하게 할려면 더 큰 값으로 설정하면 된다.  

## Non-max Suppression  

![image](https://user-images.githubusercontent.com/32921115/103642565-a95c0180-4f96-11eb-85ed-7e7bdc0b3cca.png)

- 가장 이상적인 Detection : 적절한 grid 내에 하나의 mid point가 있는 경우  
- 그러면 **mid-point가 여러 개일 경우 어떤 grid의 mid point를 선택하야 할까?** -> **Non-Max Suppression**  

![image](https://user-images.githubusercontent.com/32921115/103642714-ec1dd980-4f96-11eb-8234-ec48ea07b1d2.png)

- 위의 Object에서 Pc값이 가장 큰 값을 선택하고 나머지는 버린다.  
- output =[Pc, bx,by,bh,bw]  

### Algorithm  
1. CNN을 이용해 Output을 출력  
2. Pc <= Threshold (여기서는 0.6) 이하의 grid box를 제거 
while gird box:  
3. 남아있는 박스 중에서 가장 큰 값을 가지는 grid 선택  
4. grid의 box와 맨 처음 골랐던 box와의 IOU가 0.5 이상인 box는 모두 제거 (많이 겹치는 것 박스를 없애주기 위함)  

**Object 당 1 개의 Bounding Box를 검출할 수 있다.** 

## Anchor boxes  

![image](https://user-images.githubusercontent.com/32921115/103644172-4d46ac80-4f99-11eb-83be-64a43dc51594.png)

- 각 grid에서 여러개의 Object를 detect 하는 방법  
- y는 anchor box의 수 * object 정보를 가지고 있음 (y = [Pc, bx, by, bh, bw, c1,c2,c3, Pc, bx, by, bh, bw, c1,c2,c3]  
- Anchor box를 여러개 적용한 경우에서 각각의 Object는 midpoint를 포함하는 grid 중 **Anchor box들과 가장 큰 IOU를 가지는 grid에 Object가 할당**된다.  
- 정사각형에 가깝지 않은 Object들을 잘 검출할 수 있음. 위의 예에서는 Car, Pedistian
### Example  

![image](https://user-images.githubusercontent.com/32921115/103645579-939d0b00-4f9b-11eb-8753-93ae13e1b39e.png)

- Pedestian은 Anchor box1, Car는 Anchor box2에 더 가까움  
- (3,2) grid에서 anchor box1, anchor box2에 대해 Detection한 결과를 보여줌.  
- 만약 Pedestian은 없고 Car만 있다고 가정하면, Anchor Box2가 Anchor Box1보다 IOU가 높아 Anchor box1의 Pc는 0, 나머지 값은 ?로 바뀌고 Anchor box2의 정보만 저장이 된다.

### 한계  
1. **Anchor box 개수보다 많은 Object가 있을 경우** 다 검출하기 어려움  
2. **같은 Anchor box에 해당하는 2개 이상의 Object**가 있으면 다 검출하기 어려움 

## 선택하는 방법?  
- 사이즈를 직접 정하는 방법  
- K-means 알고리즘을 이용해 Object의 shape끼리 그룹화 뒤, 그룹에 알맞는 Anchor box를 사용  

## YOLO Algorithm  

### Training set 구성  

![image](https://user-images.githubusercontent.com/32921115/103645729-da8b0080-4f9b-11eb-8197-54ead45a92a1.png)

- 3 x 3 grid 사용, 2개의 anchor box 사용, 3개의 class  
- 위에서 3 x 3은 grid, 2는 anchor box 개수, 8은 Pc + bounding 좌표 + class 개수  
- 왼쪽 상단에는 Object가 없어 Pc 값이 0이고 나머지 원소들이 다 ?로 되있음  
- (3,2) grid에는 자동차를 감지하고, **anchor box2의 IOU가 더 높아** 아래의 원소에만 값이 들어가고 anchor box1에 해당되는 원소는 0과 ?가 되었다.  
- output vector를 이용하여 Detection 할 시 3x3x16으로 나타내고 비교한다.  

### Making Predictions  

![image](https://user-images.githubusercontent.com/32921115/103646432-01960200-4f9d-11eb-8196-b0f4d22aa943.png)

### Outputting the non-max supressed outputs  

![image](https://user-images.githubusercontent.com/32921115/103646470-12467800-4f9d-11eb-817b-e59c2158fbe7.png)

- 나온 Output에서 중복 Detection된 결과들을 제거해야 함 (non-max supresseion)  
- YOLO Algorithm 완성  


**현실적으로는 정교한 grid가 필요할 것 (19 x 19 x number of ahchor box x 8)**  

## 4. Face Recognition  

## Face Verification vs. Face Recognition  
- Face Verification : Image, name or ID를 Input으로 주고 Output으로 input의 정보에 해당하는 사람이 맞는지 확인  
- Face Recognition : K명의 사람을 비교함. ex) 100명의 사람의 DB에서 특정 사람을 찾으려면 Verification 100번 수행 -> (99.9)^100 -> 약 90%로 정확도가 줄어든다.  

## One Shot Learning  
- One-Shot Learning : 한 사람의 사진을 이용해 그 사람을 Recognition하도록 Learning 해야 한다.  

### Example  
- 4개의 사람 Image DB를 가지고 있고 Image 하나를 받아 CNN을 거친 후 Output에 softmax를 취한다 가정.  
- Output은 person1,2,3,4 그리고 None 총 5개  

But 여기서는 2가지 한계가 있음  
1. Image data가 너무 적어 learning이 잘 안됨.  
2. 새로운 사람이 추가되면 다시 learning 해야 한다.  

### Learning a similarity fucntion  
- Image의 difference를 구함.  
- d(img1, img2)에서 같은 사람이면 output은 작을 것이고, 다른 사람이면 output의 값은 클 것이다.  
- 3가지 기능을 지원해야 한다.  
  1. 가장 유사한 이미지 찾기.  
  2. 기존 DB에 없는 이미지는 없다고 판별  
  3. 새로운 이미지 등록  
  
## Siamese Network  
- DeepFace 논문에 나오는 내용  
- similarity function d를 만들기 위한 방법  

### Example  

![image](https://user-images.githubusercontent.com/32921115/103856891-c4468700-50f8-11eb-9ee9-f4c2572d3c5e.png)

- input Image x(1)ConvNet에 Feed 하고 결과물로 Output Vector를 받는다고 가정. (Softmax Layer가 없음)  
- x(2)도 위와 같은 방법 사용  
- 이후 결과 값에 norm을 적용한다. 값이 작으면 같은 사람, 값이 크면 다른 사람이다.  

![image](https://user-images.githubusercontent.com/32921115/103857070-1e474c80-50f9-11eb-856f-ece86416a68a.png)

**즉 같은 사람일 때는 norm의 결과가 작아야 하고, 다른 사람일 때는 norm의 결과가 크도록 만들어야 한다!**  

## Triplet Loss  
- Siamese Network에 사용되는 Loss Fucntion  

### Example  

![image](https://user-images.githubusercontent.com/32921115/103857968-d45f6600-50fa-11eb-9c4f-65019c61b643.png)

- 우리가 원하는 결과물은 d(A,P) <= d(A,N)  
- 만약 f(x)의 결과가 항상 0이어도, 부등식은 만족하므로 parameter를 이상하게 learning할 수 있음.  
- 그래서 알파라는 **최소한의 gap**를 더하고, 이를 **margin**이라고 한다. (알파는 hyperparameter)  

![image](https://user-images.githubusercontent.com/32921115/103858210-4df75400-50fb-11eb-95c4-8d0bde0b13d2.png)

- 즉 L(A,P,N)은 위와 같은 식이 되고, prediction과 정답 label이 가까울 수록 loss는 0에 가까워짐. (우리가 원하는 건 **D(A,P) - D(A,N) + alpha <= 0**)  
- A, P, N의 데이터 셋을 구성하기 위해서 먼저 A, P를 먼저 짝을 지어야 한다. 평균적으로 1명당 10개의 사진으로 learning해야 함.  
- Learning을 끝내면, 한 장의 사진만 있으면 Face Recognition이 가능함.  
- A,P는 동일한 사람을 쌍으로 만든다 해도, N을 램덤하게 구성하면 Loss가 항상 0이 나와 Learning이 잘 안될수 있다. (A,P : 백인 남자 아이와, N:흑인 여자 할머니)  

![image](https://user-images.githubusercontent.com/32921115/103858946-92372400-50fc-11eb-8c79-c4447ad0b68d.png)

- Training set은 **anchor, positive, negative**를 구성해야 한다.  
- Pre-trained 모델을 구하는 것도 좋은 방법.  

## Face verification and binary classfication  

![image](https://user-images.githubusercontent.com/32921115/103860540-24402c00-50ff-11eb-8d14-06dda0e53cbf.png)

- Triplet Loss + Siamese Network를 이용하여 **같다 =1, 다르다 =0인 binary classfication 문제**로 바꿀 수 있다.  
- 위 2개의 이미지 중 위의 것을 판별해야 할 이미지라 하고 아래 것을 기존의 db에 있는 이미지라고 가정  
- 기존에 있는 이미지는 굳이 ConvNet에 Feed할 필요가 없음. **Computation Cost를 줄여준다.**  

![image](https://user-images.githubusercontent.com/32921115/103860700-72552f80-50ff-11eb-99fe-b0f903e50f1c.png)

- 이런 식으로 Face Verification이 가능하다.  
