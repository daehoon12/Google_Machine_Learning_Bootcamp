## 3. Shallow Neural Networks

### The tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero, and so it centers the data better for the next layer. True/False? (tanh 함수는 보통 hidden unit에 대한 sigmoid 활성화 함수보다 더 잘 작동하는데, 그 결과의 평균이 0에 가까우므로 다음 layer의 데이터가 더 잘 중심에 배치하기 때문이다. T/F?)

- Yes. Sigmoid outputs a value between 0 and 1 which makes it a very good choice for binary classification. You can classify as 0 if the output is less than 0.5 and classify as 1 if the output is more than 0.5. It can be done with tanh as well but it is less convenient as the output is between -1 and 1

### Suppose you have built a neural network. You decide to initialize the weights and biases to be zero. Which of the following statements is true? (W를 0으로 초기화하면 무슨 일이 일어날까?)
- 0으로 가중치를 초기화 한다면 모든 뉴런들(hidden Layers)이 같은 값을 나타낼 것이고, 역전파 과정에서 각 가중치의 update가 동일하게 이뤄질 것이다. (Symmetric breaking)  

### Logistic regression’s weights w should be initialized randomly rather than to all zeros, because if you initialize to all zeros, then logistic regression will fail to learn a useful decision boundary because it will fail to “break symmetry”  True/False? (Logstic regression's 가중치 w는 random으로 초기화해야한다, 0으로 초기화 할경우 break symmetry 현상이 발생한다. T/F?)  
- Logistic regression은 hidden layer가 없어 모델이 X의 값에 영향을 받는다. 따라서 두 번째 반복에서 가중치 값은 x의 분포를 따르고 x가 일정한 벡터가 아닌 경우 값이 무조건 값이 다르게 나온다.
