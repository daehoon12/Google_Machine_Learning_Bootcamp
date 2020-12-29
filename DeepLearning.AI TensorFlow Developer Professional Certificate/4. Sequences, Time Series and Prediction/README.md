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
- 데이터가 spike 때문에 약간 튀더라도 이전 패턴과 비슷  
