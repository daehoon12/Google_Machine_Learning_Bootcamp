# 5. Sequence Models

## Sequence data  
- 각각의 데이터가 순서가 있는 데이터 (음성, text, 주가 등)  

### Example of sequence data  

![image](https://user-images.githubusercontent.com/32921115/103970988-d97ced80-51ac-11eb-9c2e-7dafda44c239.png)

## Notation  

![image](https://user-images.githubusercontent.com/32921115/103971090-06310500-51ad-11eb-96ad-9ffb210b520c.png)

- Voca에 있는 단어들에 index를 매김 (dictionary 사용)  
- x 문장을 word로 쪼갠 뒤 one-hot encoding을 사용해 해당 index만 1로 체크 한다.  
- word의 벡터는 voca와 똑같은 dimension을 가짐  

## Recurrent Neural Network Model  

### Why not a standard network?  
- 다른 예제에서 input, output이 다른 길이일 수 있음.  
- text의 다른 위치에서 learning된 feature가 공유되지 않음.  

### Recurrent Neural Networks  
- Sequence Data를 처리하기 위해 등장  
- 기존의 Neural Network과 다르게 **Hidden State**를 가지고 있음.  

### RNN Architecture  

![image](https://user-images.githubusercontent.com/32921115/103976190-69289900-51b9-11eb-8c8b-53df153f6f67.png)

a : Hidden State  
y^ : 예측 값  

- y2에서 예측할 때, y1에서 compute된 정보(첫번째 layer의 hidden state)를 일부 사용함.  

### RNN Forward Pass  

![image](https://user-images.githubusercontent.com/32921115/103976660-b48f7700-51ba-11eb-8f14-69f83de19f86.png)

- 일반 RNN에서는 Hidden State를 구할 때, 보통 Tanh 활성화 함수 사용.  
- 예측 값은 어떤 목적인지 따라서 sigmoid, softamx로 나뉨.  

### RNN Backward Pass  
- 출력부터 모든 시간 스텝에 적용되는 gradient를 모두 계산  
- RNN의 구조를 펼쳐보면(unfold), RNN을 시간 스텝 수만큼 히든 레이어를 가진 deep FNN으로 볼수 있음  
- 따라서 동일한 방식으로 역전파를 수행(Back-propagation througn time, BPTT)  

## Vanishing Gradients with RNNs  
- layer가 깊어질 수록 앞의 값 영향을 받지 않음.  
- 각 layer에서는 tanh, sigmoid 함수를 통과한다. 이 때 Backpropagation을 하면 **미분값은 0~1 사이 값**이 나오고, 이 값을 계속 곱하면 **gradient는 0에 수렴해 올바른 Learning이 되지 않는다.**  
- Vanising Gradients Problems을 해결하려고 나온 두 개의 기법! **LSTM, GRU**  

## Long short-term memory (LSTM)  

![image](https://user-images.githubusercontent.com/32921115/103977362-7b580680-51bc-11eb-85bb-ff74c24261c1.png)

- Time Step 사이에 Hidden State와 더불어 **Cell State**도 전달.  
- 기존 RNN보다 훨씬 많은 파라미터를 가짐 -> learning 하는데 더 많은 시간과 데이터 필요  

### Cell State  
- 어떤 정보를 잊고 기억할 지에 관한 판단이 반영된 정보  
- forget, input, output gate로 구성  

### Forget Gate  

![image](https://user-images.githubusercontent.com/32921115/103977886-8d867480-51bd-11eb-80c6-68d465a13238.png)

- 현재 time-step t의 입력값 x_t와 이전 Hidden state h_(t-1)을 고려해, **이전 Cell State를 얼마나 보존할 지 판단한다.**  

### Input Gate  

![image](https://user-images.githubusercontent.com/32921115/103978015-d9391e00-51bd-11eb-8156-17e831c65868.png)

- 현재 time-step t의 input x_t와 이전 Hidden State h_(t-1)를 고려해, **Cell State에 현재 State에 대한 값을 얼마나 더할지 판단한다.**  

### Cell State Updadte  

![image](https://user-images.githubusercontent.com/32921115/103978133-1ef5e680-51be-11eb-812c-e0bc1f1817e8.png)

### Output Gate  

![image](https://user-images.githubusercontent.com/32921115/103978293-87dd5e80-51be-11eb-8706-c67b2ba1e95b.png)

- 업데이트된 Cell State와와 x_t 와 h_(t-1) 을 고려해 Hidden State 업데이트하고, 다음 time-step t+1 로 전달  
- [] 의미는 벡터를 쌓아놓은 걸로 생각하면 된다.  

## Gated recurrent unit (GRU)  

![image](https://user-images.githubusercontent.com/32921115/103979042-3504a680-51c0-11eb-9c17-4214f61dcca8.png)

- LSTM보다 간소화, 별도의 Cell State 없이 2개의 Gate만 사용  

### Reset Gate  

![image](https://user-images.githubusercontent.com/32921115/103979106-52397500-51c0-11eb-97bd-57e02ece83f9.png)

- 이전 히든 스테이트 h_(t-1)과 현재 입력 값 x_t를 고려해, **현재 입력 값을 히든 스테이트 h_t에 얼마나 반영**할 지 판단  

### Update Gate  

![image](https://user-images.githubusercontent.com/32921115/103979118-5bc2dd00-51c0-11eb-9332-fbd37f92bf7a.png)

- 히든 스테이트 h_(t-1)과 현재 입력 값 x_t로부터 z_t값을 생성하고 z_t를 기준으로 **리셋 게이트로부터 반환된 값과 이전 히든 스테이트 중 어디에 얼만큼 가중치를 둘 지 판단**  

## Bidirectional RNNs  

![image](https://user-images.githubusercontent.com/32921115/103982563-11912a00-51c7-11eb-8609-e45292c3d514.png)

- 기존 RNN은 이전 timestep의 정보를 이용해 현재의 label을 예측하는 방식  
- **미래 시점에서 현재 시점으로도** 고려해서 처리할 수 있음.  
- 많은 Text가 있는 경우 LSTM을 이용한 Bidirectional RNN을 사용한다.  
- Parameter수는 기존 RNN 수의 2배  

## Deep RNNs  

![image](https://user-images.githubusercontent.com/32921115/103982595-1f46af80-51c7-11eb-8298-156cbfaaa0bc.png)

- 기존 RNN의 위에 layer를 하나 더 쌓음.  

## 2. Natural Language Processing & Word Embeddings  

## Word Representation  

### One-hot Representation  

![image](https://user-images.githubusercontent.com/32921115/103984790-21127200-51cb-11eb-99f8-4727c542d066.png)

- 표현하고자 하는 단어의 index만 1, 나머지는 0  
- **V의 개수가 늘어나면 vector의 dimension이 늘어난다**는 단점이 있다.  
- **각 단어를 하나의 Object로 여기기 때문에 단어 간의 관계를 추론할 수 없다.** (I want a glass of orange ____ 를 통해서 빈칸에 juice가 들어가도록 학습했다고 하더라고, I want a glass of apple ____ 이라는 입력이 들어왔을 때, apple을 orange와 비교해서 비슷하다고 여겨서 juice를 추론할 수가 없음.
