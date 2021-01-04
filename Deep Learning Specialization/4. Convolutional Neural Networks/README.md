# Convolutional Neural Networks  

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
