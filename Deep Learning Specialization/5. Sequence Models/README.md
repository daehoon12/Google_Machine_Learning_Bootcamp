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

### Simplified RNN Notation  

![image](https://user-images.githubusercontent.com/32921115/104986951-ccd38180-5a57-11eb-8c68-be3f87a201ff.png)


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
- **각 단어를 하나의 Object로 여기기 때문에 단어 간의 관계를 추론할 수 없다.** (I want a glass of orange ____ 를 통해서 빈칸에 juice가 들어가도록 학습했다고 하더라고, I want a glass of apple ____ 이라는 입력이 들어왔을 때, apple을 orange와 비교해서 비슷하다고 여겨서 juice를 추론할 수가 없음.)  

### Featurized Representation : Word Embedding  
- Word를 사용자가 지정한 size만큼 숫자로 된 vector로 만듬.  

![image](https://user-images.githubusercontent.com/32921115/103986649-79973e80-51ce-11eb-9436-268cf030d4f7.png)

- row : feature  
- cal : V에 있는 단어들.  
- 위에 Man이라는 단어를 보면 Gender에 해당하는 값이 -1이고, woman은 1이다. **서로 반대되는 개념이기 때문에 두 합이 0에 가까움.**  
- Apple이나 Orange는 Gender에 관련이 없기 때문에 두 값이 0에 가까운 것을 확인할 수 있다.  
- 이렇게 Word Embedding을 통해 Embedding Matrix를 얻을 수 있음.  
- Word Matrix의 값을 통해 "I want a glass of apple ____" 다음에 나올 word는 juice라는 것을 더 쉽게 예측할 수 있다.  

### Visualizing word embeddings  

![image](https://user-images.githubusercontent.com/32921115/103987037-25d92500-51cf-11eb-945d-467d6292ec06.png)

- 300 dimension의 word embedding 행렬을 조금 더 쉽게 이해하기 위해 시각화.  
- **t-SNE 알고리즘** : 임베딩 행렬을 더 낮은 차원으로 (위의 그림에서는 300D -> 2D)로 mapping해서 단어들을 시각화, 유사한 단어들은 서로 가까이에 있는 것을 확일할 수 있다.  

## Using word embeddings  

![image](https://user-images.githubusercontent.com/32921115/103987137-4e611f00-51cf-11eb-9717-e41fcfd0e53d.png)

- 위의 사진은 이름을 인식하는 예제. Sally Johnson이 이름이라는 것을 확실히 알기 위한 방법은 orange farmer가 사람인 것을 알아내는 것이다.  
- one-hot encoding이 아닌 word embedding을 사용해 학습한 뒤, 새로운 example에 대해서 apple이 orange와 유사하다는 것을 알기 때문에 Robert Lin이라는 사람 이름인 것을 더 쉽게 예측 가능.  
- dataset에 없는 durian cultivator라는 단어가 나와도 **durian이 과일, cultivator가 사람을 나타내는 것을 learning**하면 orange farmer처럼 normalization이 될 것이다. (값이 비슷하기 때문)  
- Word Embedding이 이런 Normalization이 가능한 이유는 매우 큰 단어 뭉치들 (10억 ~ 100억개의 단어) Learning하기 때문이다. 

### Transfer Learning and Word Embeddings  

![image](https://user-images.githubusercontent.com/32921115/103988103-ddbb0200-51d0-11eb-8c8a-fe918edb14a8.png)

- 적은 수의 training set을 가지고 있어도, transfer learning을 통해 미리 learning된 word embedding을 가지고 learning 할 수 있다.  
1. text corpus로 word embedding을 학습  
2. 기존 모델에 소규모 tranining set에 learning   
3. 새로운 data로 word embedding을 계속 튜닝  


![image](https://user-images.githubusercontent.com/32921115/103988125-e4e21000-51d0-11eb-90f1-fa79a30ef053.png)

- Face encoding과 유사하지만 word embedding 경우에는 사용자의 **단어가 정해져 있기 때문에 voca에 존재하지 않는 단어는 learning 할 수 없다.**  


## Properities of word embeddings  
- Word Embedding은 **유추(Analogy)하는 문제**에 큰 도움을 줌  

### Analogy  

![image](https://user-images.githubusercontent.com/32921115/103993495-0d6e0800-51d9-11eb-9dd4-9e01f433af6b.png)

- 'Man과 Woman은 King과 ____ 과 같다' 라는 유추 문제가 있을 때, 어떻게 예측할까?  
- Man은 4차원 Vector로 표시, e(5391) = e(man), Woman도 마찬가지로 e(woman)으로 표현된다.  

![image](https://user-images.githubusercontent.com/32921115/103993538-1a8af700-51d9-11eb-9161-95bbb2b2a53a.png)  

- 위의 결과에 의해서 우리는 man과 woman의 관계가 king과 queen의 관계와 유사하다고 추론할 수 있다.  

![image](https://user-images.githubusercontent.com/32921115/103993843-8c634080-51d9-11eb-8e1f-7de9e2f6a2b6.png)

- Word Embedding은 약 300D의 공간에서 표현될 것이고, 그 공간 안에서 각 단어들은 점으로 표현이 된다.  
- 위 그림처럼 값이 비슷한 두 차이 벡터는 매우 유사할 것이다. (300차원에서 그려진 벡터, 2차원 아님)  
- **Sim은 Similarity Function**을 의미하며, 두 단어 사이의 Similarity를 계산한다. (실제 논문에선 30 ~75% Acc를 보여줌)  

![image](https://user-images.githubusercontent.com/32921115/103994109-e3691580-51d9-11eb-8066-e7d1af7ebd70.png)

- t-SNE 알고리즘은 300D를 2D로 매핑해주는데, 매우 복잡하고 non-linear한 매핑이다. **즉 임베딩을 통해 단어간의 analogy를 구할 때, 원레 벡터의 Dimension인 300D를 통해 비교 해야한다.**  

### Cosine Similarity  

![image](https://user-images.githubusercontent.com/32921115/103994337-3773fa00-51da-11eb-9554-1f34795e16b4.png)

- 일반적인 Simiairty Function으로 가장 많이 사용  
- **벡터가 수직이면 두 벡터간의 유사도는 0**  

## Embedding Matrix  
- Word Embedding을 통해 learning 됨.  
- 만약 1만개의 단어를 사용하고 feature로 300개를 사용하면 (300, 10000)의 Matrix를 가짐.  
- Orange를 예로 들면 6257 column에 있으며, orange에 해당하는 embedding vector가 됨.  
- one-hot encoding을 통한 o(6257)과 내적을 수행하면 우리가 원하는 embedding vector를 구할 수 있음.  

#### 공식  

![image](https://user-images.githubusercontent.com/32921115/103994829-e57fa400-51da-11eb-9be3-5e89f05f027a.png)

- **우리가 학습해야되는 것이 Embedding Matrix E라는 것이 가장 중요, E는 초기에 무작위로 초기화 된다.**  
- one-hot vector 곱은 **매우 비효율적**, one-hot vector가 꽤 큰 차원인데다가 대부분 0으로 채워져 있어 메모리 낭비도 심하고 compute 양도 많음. 실제로는 **Embedding vector를 얻는 특화된 함수를 사용함**  

## Learning word embedding  

### Neural language model  

- 가장 자연스러운 단어 시퀀스를 찾아내는 Model  
- 이전 단어들이 주어졌을 때, 다음 단어를 예측 or 주어진 양쪽의 단어들로부터 가운데 비어있는 단어를 예측  
- **Embedding Matrix를 학습**할수 있는 방법

![image](https://user-images.githubusercontent.com/32921115/103997623-b1a67d80-51de-11eb-81bb-10c166175fa6.png)

- 위의 그림에서 단어들의 one-hot vector O가 있고, **Matrix E를 통해 Embedding vector**를 얻을 수 있음.  
- **Embedding Vector는 Network의 Input**으로 들어가고, Output은 **Softmax 함수를 통해 Voca의 확률들로 변환**된다.  

![image](https://user-images.githubusercontent.com/32921115/103998751-9be58800-51df-11eb-934d-f1c112580ad6.png)

- **fixed historical window**를 사용, 예를 들어 다음 단어를 에측하는데 앞의 4개의 단어만 사용하는 것이다. 크기는 hyperparameter로 설정할 수 있음.  
- 

## Word2Vec  
- Embedding의 한 종류 (Word Embedding), **단어 간 유사도를 반영할 수 있도록**하기 위해단어 간 유사도를 반영할 수 있도록 word를 Vector로 바꾸어주는 Algorithm.  

### Skip-gram  
- 중심이 되는 단어를 무작위로 선택하고 주변 단어를 예측.  
- 중심 단어가 Context(input)이 되고 주변 단어를 선택해서 Target(prediction)이 되도록 하는 **superivised learning**  
- 주변 단어는 여러 개를 선택할 수 있다.  

![image](https://user-images.githubusercontent.com/32921115/104000715-013a7880-51e2-11eb-8058-02f2665e0db8.png)

### 과정  
1. Orange라는 Context를 input으로 넣는다.  
2. Orange의 One-hot Vector와 Embedding Matrix E를 dot-Product  
3. Output Embedding Vector를 Softmax Layer에 넣는다.  
4. Output y^을 얻는다.  

### Softmax Fucntion and Loss Fucntion  

![image](https://user-images.githubusercontent.com/32921115/104000939-4a8ac800-51e2-11eb-836e-266b9f7355a3.png)  

![image](https://user-images.githubusercontent.com/32921115/104000965-54143000-51e2-11eb-8241-c38f508d7e49.png)  

- 여기서 세타는 output과 관련된 weight parameter  

### 문제점  
- 계산 속도가 느림. 특히 softmax일 경우 len(v) =10000일 때, 10000개의 단어를 계산해야 한다.  
- **hierarchical softmax**를 사용해 해결, Tree를 사용하는 것인데, 자주 사용하는 단어일 수록 Tree의 top, 그렇지 않다면 bottom에 둔다. -> Search가 log로 줄기 때문에 softmax보다 빠름.  

## Negative Sampling  

### 배경
- Context C를 샘플링하면, Target t를 context의 앞 뒤 10단어 내에서 샘플링할 수 있음.  
- 무작위로 샘플링하는 방법이 가장 간단한데, the, of,a 같은 단어들이 빈번하게 샘플링됨.  
- 따러서 이 것들의 균형을 맞추기 위해 다른 방법을 사용해야 함.  

### Defining a new learning problem

![image](https://user-images.githubusercontent.com/32921115/104003516-d7835080-51e5-11eb-9e13-a4f9e68ca218.png)

- 위에서 orange-juice와 같은 positive training set이 있다면, 무작위로 negative traning set을 K개 샘플링한다.  
- 'of'같이 context에 있는 단어를 선택할 수도 있는데, Positive지만 일단은 negative라고 취급한다.  

### Model  

![image](https://user-images.githubusercontent.com/32921115/104004156-c38c1e80-51e6-11eb-9407-d4d45a9df3c0.png)

- Context와 Target이 input x가 되고, positive or negative는 output y가 된다.  
- logistic regression model로 정의할 수 있다. -> 1만 차원의 softmax가 아닌 1만차원의 Binary classfication 문제로 변함.  
- **skip-gram에 비해 계산량이 줄어든다.**  

### 어떻게 Negative sample을 선택?  
1. Corpus에서의 Empirical frequency(경험적 빈도)에 따라서 샘플링, 얼마나 자주 다른 단어들이 나타나는 지에 따라서 샘플링 할 수 있음.  
2. 1/voca_size 사용해 무작위로 샘플링

## Global vector for word representation (GloVe) word vector  
- **corpus에서 i의 context에서 j가 몇번 나오는지 (Xij)** 구하는 작업.  
- Context와 target의 범위를 어떻게 지정하느냐에 따라 Xij == Xji 일수 있고, 아닐 수도 있다.  

## Sentiment Classfication  
- NLP에서 중요한 구성요소 중 하나.  
- 많은 dataset이 필요 없이 **Word Embedding**을 사용해 좋은 성능의 classfication을 만들 수 있음.  

### Simple sentiment classfication model  

![image](https://user-images.githubusercontent.com/32921115/104005031-e0752180-51e7-11eb-8490-6f8a08020179.png)

- "The dessert is excellent"라는 input이 있을 때, 각 word들 embedding vector(300D)로 변환하고 각 단어의 vector 값들의 평균을 구해 softmax의 output으로 결과를 예측하는 모델  
- 적은 dataset or 자주 사용되지 않는 단어가 input이어도 해당 모델에 적용 가능.  
-'Completely lacking in good taste, good service, and good ambience'라는 부정적인 sentence가 있을 경우, **단어의 순서**를 무시하고 단순히 good이라는 단어가 많이 나와 positive로 예측할 수 있음.
- 위의 문제를 RNN을 이용해서 해결할 수 있다.  

### RNN for sentiment classfication  

![image](https://user-images.githubusercontent.com/32921115/104005568-ab1d0380-51e8-11eb-9541-8a4d82ef2b16.png)  

- RNN을 이용해 **Sequence를 고려하기 때문**에 해당 리뷰가 부정적인 것을 알 수 있다.  
- absent라는 단어가 training set에 포함되지 않아도 word embedding이 되면, normalization되어 제대로 된 결과를 얻을 수 있다.  
