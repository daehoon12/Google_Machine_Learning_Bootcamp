# 개념  
## Tokenizer  
- Sentence를 token화 하는데 사용되는 객체  

### Parameter
- oov_token = <Token> : 정의되지 않은 word들을 <Token>으로 만든다.  
- num_word = n : 가장 빈도가 높은 n개의 단어만 선택한다.   

## Function  
- Tokenizer.word_index : 단어와 숫자의 키-값 쌍을 포함하는 딕셔너리를 반환   
- fit_on_texts(sentences) : 문자 데이터를 입력받아서 리스트의 형태로 변환  
- texts_to_sequences(sentences) : sentences의 word들을 index로 변환해 저장  
- pad_sequence() : data의 전처리를 담당. 제일 긴 데이터의 길이를 기준으로 padding 함. parameter에 따라 앞 or 뒤에 padding 가능.  

## Word Embedding  
- 워드 임베딩(Word Embedding)은 단어를 벡터로 표현하는 방법으로, 단어를 밀집 표현(Dense Representation)으로 변환  
- 케라스에서 제공하는 도구인 Enbedding()는 단어를 랜덤한 값을 가지는 밀집 벡터로 변환한 뒤에 인공 신경의 Weight를 학습하는 것과 같은 방식으로 단어 벡터를 학습하는 방법을 사용  
### 1. Sparse Representation  
- 표현하고자 하는 단어의 인덱스의 값만 1이고, 나머지 인덱스에는 전부 0으로 표현되는 벡터 표현 방법
- 단어의 개수가 늘어나면 벡터의 차원이 한없이 커진다는 문제점  
- One-Hot Encoding이 이 방식  

### 2. Dense Representation  
- 사용자가 설정한 값으로 모든 단어의 벡터 표현의 차원을 맞춤, 이 과정에서 0,1 값만 가지는게 아니라 실수 값을 가지게 된다.  

** Ex) 강아지 = [0.2 1.8 1.1 -2.1 1.1 2.8 ... 중략 ...] dim = 128  
- 이 벡터의 차원은 128  
- 이 경우 벡터의 차원이 조밀해졌다고 하여 밀집 벡터(dense vector)라고 합니다.  
- Word Embedding이 이 표현을 사용  
