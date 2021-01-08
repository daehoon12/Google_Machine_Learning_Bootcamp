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

![image](https://user-images.githubusercontent.com/32921115/103974917-2c0ed780-51b6-11eb-9cfe-d4d4fe439bc2.png)

Ht : Hidden State, 이전 TimeStep의 정보  
