# 개념  

## Time Series Patterns  
- Trend + seasonality + autocorrelation + noise  

### Trend  
- 시간이 지남에 따라 변환하는 경향  
- 변화 방향에 따라 증가, 감소, 유지로 나뉨  
- Linear, Non-linear trend 존재  

### Seasonality  
- 일정 기간 동안 나타나는 패턴  
- 4계절이 아니더라도 일간, 연간 등 반복적인 규칙을 Seasonality로 볼 수 있음.  
- 아이스크림, 의류 등 소비재 쪽에서 많이 보임  

### Cycles  
- Seasonality와 비슷하긴 하지만 구간이 정해져 있지 않음  
- 즉, 고정된 빈도가 아닌 형태로 증가나 감소하는 모습을 보일 때 Cycle이 나타난다고 한다.  

### Random, White Noise  
- 주기성을 찾을 수 없는 Data  

**Example**  
![image](https://user-images.githubusercontent.com/32921115/103260896-b602b780-49e2-11eb-8736-6df4a6ba0a66.png)

1. 매년 강한 Seasonality와 Cycle을 보임, 하지만 전체적인 Trend는 보이지 않음  
2. 가격이 내려가는 Trend를 보임  
3. 강한 Seasonality와 함께 증가 Trend를 보임  
4. No Seasonality, No Cycle and No Trendm Random  

### Autocorrelation (자기상관)  
![image](https://user-images.githubusercontent.com/32921115/103262083-31fefe80-49e7-11eb-91f2-a4d0e98bc865.png)
  
  
- 이전의 Pattern이 현재 Pattern에 영향을 끼침  
- 데이터가 spike 때문에 Scale이 다르더라도 이전 패턴과 비슷  

## Non-stationary time series  
![image](https://user-images.githubusercontent.com/32921115/103262302-f153b500-49e7-11eb-99ec-62c8bef02c49.png)
  
  
- trend나 seasonality가 갑자기 깨지는 time series  
- 이 time series를 예측하는 방법은 거의 불가능함. 그래서 갑자기 데이터가 변하는 부분을 training set으로 정함.  

## Fixed Partitioning  
![image](https://user-images.githubusercontent.com/32921115/103262409-32e46000-49e8-11eb-8848-bac3cefee47e.png)
  
## Metrics  
![image](https://user-images.githubusercontent.com/32921115/103262564-a4241300-49e8-11eb-88a9-598422202e51.png)
  
  
- Model을 평가하는데의 지표  
- 보통 mse, mae 많이 사용  

## Naive Forecast  
![image](https://user-images.githubusercontent.com/32921115/103262797-5065f980-49e9-11eb-850f-5ca087edf125.png)
  
  
- 이전 time stamp를 그대로 재사용  

## Moving Average  
![image](https://user-images.githubusercontent.com/32921115/103262823-5eb41580-49e9-11eb-96f6-bb2e09c4caf9.png)
  
  
- 일정한 Time에서 Time + Window size의 평균 값을 사용  
ex) 10시간 동안의 평균 값을 다음 예측 값으로 사용  

## Differencing  
![image](https://user-images.githubusercontent.com/32921115/103262866-7b504d80-49e9-11eb-8947-0f1f49499653.png)
  
  
- Seasonality를 없애기 위해 값을 빼줌  
- 1년 전의 데이터를 다 빼서 Cycle을 없애서 Random한 값을 뽑아냄  

## Differencing + Moving Average  

- 여기서 아까 뺏던 365일의 차이값을 다시 더하면 아래의 그래프가 나온다.  
![image](https://user-images.githubusercontent.com/32921115/103262900-99b64900-49e9-11eb-8ef5-00d34f4f5fd2.png)  

- 여기서 원본의 Data의 Moving Average를 더하면 이런 그래프가 나온다.  

![image](https://user-images.githubusercontent.com/32921115/103262942-bf435280-49e9-11eb-9b53-99bcddb052ad.png)
  
