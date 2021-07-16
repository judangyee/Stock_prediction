# Stockple
import FinanceDataReader as fdr
from matplotlib import pyplot as plt
import pandas as pd


#종목정보 입력
stock_code = ('950190')
df_krx = fdr.StockListing('KRX')
df = fdr.DataReader(stock_code, '2020')

#시간(ds)와 종가(y)값만 남김
df = df.reset_index()
df['ds'] = df['Date']
df['y'] = df['Close']
data = df[['ds','y']]

#과거~현재값 그래프 그리기
x = (df['ds'])
y = (df['y'])

plt.plot(x, y, label = 'present')

plt.xlabel('Time') #x축 라벨
plt.ylabel('Price') #y축 라벨
plt.title("Stock_prediction") #제목
plt.legend()
plt.show() #보여주기
plt.savefig('photo.png') #저장

#prophet 불러옴
from Prophet import fbprophet
#학습
model = Prophet()
model.fit(data)
#24시간 예측
future = model.make_future_dataframe(periods=24, freq='H')
forecast = model.predict(future)
#그래프1
future = model.plot(forecast) #예측 그래프
#그래프2
fig2 = model.plot_components(forecast) #트렌드, 경향성 그래프 
