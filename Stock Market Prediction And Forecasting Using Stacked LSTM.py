#!/usr/bin/env python
# coding: utf-8

# # Stock Market Prediction And Forecasting Using Stacked LSTM
# 
# By: Ritesh Dhakad
# 

# Import required libraries

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')


# Load and read the data

# In[60]:


df=pd.read_csv('https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv')
df.head()


# Data understanding

# In[61]:


df.shape


# In[62]:


df.columns


# In[63]:


df.info()


# 
# 
# Statistical Information
# 

# In[64]:


df.describe()


# 1.Average stockprice of TATAGlobal is 149 Rupees
# 
# 2.Maximum Stockprice is 325 Rupees and Minimum Stockprice is 80 Rupees.
# 
# 3.Maximum Trade Quantity is 29M and Minimum Trade Quantity is 2.33M.
# 
# 4.Maximum Turnover is 55,755 Lakhs, and minimum Turnover is 37 lakhs.

# 
# 
# 
# Sort the Date by Ascending

# In[65]:


df=df.sort_values(by='Date', ignore_index=True)
df.head()


# 
# Exploratory Data Analaysis
# 

# In[66]:


df.Date.dtype


# 
# 
# 
# Convert Date columns data type fron Object to Datetime format

# In[67]:


df['Date']=pd.to_datetime(df['Date'])
print(df.Date.dtype)


# 
# 
# 
# 
# How many Days Stock Data we have?

# In[68]:


print("Starting Date:", df.Date.min())
print("Last Date:", df.Date.max())
print("Number of Days:", (df.Date.max()-df.Date.min()))


# 
# 
# 
# 
# 
# For 2991 Days of Stock Data we have, Including saturday and sunday.

# 
# 
# TATAGlobal Stock Price Analaysis

# In[69]:


figure = go.Figure(data=[go.Candlestick(x=df["Date"],
                                        open=df["Open"],
                                        high=df["High"],
                                        low=df["Low"],
                                        close=df["Close"])])
figure.update_layout(title = "TATAGlobal Stock Price Analysis",
                     xaxis_rangeslider_visible=False)
figure.show()


# 1.From 2011 to 2017 Stock Price of TATAGlobal is almost constant.
# 
# 2.From start of the 2017 the Stock price increased rapidly.
# 
# 3.TATAGlobal stock price is high in Start of 2018.
# 
# 4.After the Start of 2018 the stock price decreasing.

# 
# 
# Stock Analysis by Perticular period
# 

# In[70]:


figure = px.line(df, x='Date', y='Close', 
                 title='Stock Market Analysis with Time Period Selectors')

figure.update_xaxes(
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
figure.show()


# In[ ]:





# For Further Analysis we have to choose any one column between Open, Close, High, Low, Here i am selecting Close column and Data column.

# In[71]:


data=df[['Date','Close']]
data.head()


# In[ ]:





# Time Series Forescasting
# 

# In[ ]:





# Stock Price contains trend and seasonality, so we are going to use Holt winters method, ARIMA & SARIMA.

# In[ ]:





# To Run Time series method we have to set Date to Index

# In[72]:


data.set_index('Date', inplace=True)
time_series=data[['Close']]
time_series.head()


# 
# 
# As we know that the stock prices depends on recent historical data and current situations so Lets choose only 1 year of historical data.

# In[73]:


ts=time_series[-365:]


# Split data into training and testing set

# In[74]:


len(ts)


# 
# 
# 
# Lets Select 335 Data Points as training and Rest are testing set.

# In[75]:


train=ts[:335]
test=ts[335:]
print("Training Data Points:", len(train), "Testing DataPoints:", len(test))


# 
# 
# 
# 1. Holt's Winters Additive Method

# In[76]:


warnings.filterwarnings('ignore')


# In[77]:


from statsmodels.tsa.holtwinters import ExponentialSmoothing

model=ExponentialSmoothing(train, seasonal_periods=12, trend='add', seasonal='add')
model=model.fit(optimized=True)
forecast=model.forecast(len(test))
forecast.index=test.index
forecast[20:]


# 
# 
# Plot Train Test and Forecasted values
# 

# In[78]:


plt.figure(figsize=(12,4))
plt.plot(train, label='Train data', color='green')
plt.plot(test, label='Test data')
plt.plot(forecast, label='Forecasted data')
plt.legend()
plt.title('Holt Winters Additive Method', color='green');


# 
# 
# Evaluating the Model

# In[79]:


from sklearn.metrics import mean_squared_error as mse,  mean_absolute_percentage_error as mape, r2_score

rmse=np.sqrt(mse(test,forecast)).round(2)
mpe=mape(test, forecast).round(2)*100
result=pd.DataFrame({'Method':'Holt Winters Add Method', 'RMSE':[rmse], 'MAPE':[mpe]})
result


# 
# 
# 
# R2 Score is Negative so not best method to forecast for stock price.

# In[ ]:





# 2. Holts Winters Multiplicative method
# 

# In[81]:


model=ExponentialSmoothing(train, seasonal_periods=12, trend='add', seasonal='mul')
model=model.fit(optimized=True)
forecast=model.forecast(len(test))
forecast.index=test.index
forecast[20:]


# 
# 
# 
# Plot train, test adn forecasted values

# In[82]:


plt.figure(figsize=(12,4))
plt.plot(train, label='Train data', color='green')
plt.plot(test, label='Test data')
plt.plot(forecast, label='Forecasted data')
plt.legend()
plt.title('Holt Winters Multiplicative Method', color='green');


# 
# 
# 
# Evaluation of model

# In[27]:


rmse=np.sqrt(mse(test,forecast)).round(2)
mpe=mape(test, forecast).round(2)*100
mul_result=pd.DataFrame({'Method':'Holt Winters Mul Method', 'RMSE':[rmse], 'MAPE':[mpe]})
result=pd.concat([result,mul_result])
result


# 
# 
# 
# RMSE, MAPE are lower than Additive method.

# In[ ]:





# 3. ARIMA

# Check Trend and Seasonality

# In[28]:


from statsmodels.tsa.seasonal import seasonal_decompose
decompose = seasonal_decompose(ts, 
                            model='multiplicative', period=30)
fig = plt.figure()  
fig = decompose.plot()  
fig.set_size_inches(12, 8)


# Time Series contains Seasonality of 3 Months.

# 
# 
# ACF & PACF to see the p,q,d values

# In[29]:


pd.plotting.autocorrelation_plot(ts)


# In[30]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(ts, lags=150), plot_pacf(ts)


# In[31]:


import statsmodels.api as sm

model=sm.tsa.arima.ARIMA(train, order=(1,1,1))
model=model.fit()
forecast=model.forecast(len(test))
forecast.index=test.index
forecast[20:]


# In[32]:


rmse=np.sqrt(mse(test, forecast)).round(2)
rmse


# 
# 
# 
# RMSE is lower than above two methods, lets select best for value of p,d,q to lower than RMSE.

# In[33]:


warnings.filterwarnings('ignore')

import itertools
p=range(0,8)
q=range(0,8)
d=range(0,2)
pdq=list(itertools.product(p,d,q))
len(pdq)


# In[34]:


rmse=[]
order1=[]
for pdq in pdq:
    try:
        model=sm.tsa.arima.ARIMA(train, order=pdq).fit()
        pred=model.predict(start=len(train), end=len(ts)-1)
        error=np.sqrt(mse(pred,test))
        order1.append(pdq)
        rmse.append(error)
    except:
        continue


# In[36]:


order=pd.DataFrame(data=rmse, index=order1, columns=['RMSE'])
order.head()


# In[39]:


pd.pivot_table(order, values='RMSE', index=order.index, aggfunc=min).sort_values(by='RMSE', ascending=True)[:5].round(2)


# 
# 
# Using p=0, d=0, q=4 train the ARIMA model.
# 
# 

# In[40]:


arima=sm.tsa.arima.ARIMA(train, order=(0,0,4))
arima_fit=arima.fit()
forecast=arima_fit.forecast(len(test))
forecast.index=test.index
forecast


# 
# 
# 
# Plot train, test and forecasted values

# In[41]:


plt.figure(figsize=(12,4))
plt.plot(train, label='Train data', color='green')
plt.plot(test, label='Test data')
plt.plot(forecast, label='Forecasted data')
plt.legend()
plt.title('ARIMA Method', color='green');


# 
# 
# 
# Evaluation of model

# In[42]:


rmse=np.sqrt(mse(test,forecast)).round(2)
mpe=mape(test, forecast).round(2)*100
arima_result=pd.DataFrame({'Method':'ARIMA', 'RMSE':[rmse], 'MAPE':[mpe]})
result=pd.concat([result,arima_result])
result


# 
# 
# 
# RMSE and MAPE are lower than both Holt winters additive and multiplicative methods.
# 

# In[ ]:





# 4. SARIMA
# 

# In[43]:


from statsmodels.tsa.statespace.sarimax import SARIMAX

model=SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
model=model.fit()
forecast=model.forecast(len(test))
forecast.index=test.index
forecast[20:]


# 
# 
# Plot train, test and forecasted values

# In[44]:


plt.figure(figsize=(12,4))
plt.plot(train, label='Train data', color='green')
plt.plot(test, label='Test data')
plt.plot(forecast, label='Forecasted data')
plt.legend()
plt.title('SARIMA Method', color='green');


# 
# 
# 
# Evaluation of model

# In[45]:


rmse=np.sqrt(mse(test,forecast)).round(2)
mpe=mape(test, forecast).round(2)*100
sarima_result=pd.DataFrame({'Method':'SARIMA', 'RMSE':[rmse], 'MAPE':[mpe]})
result=pd.concat([result,sarima_result])
result


# 
# 
# ARIMA is the best algorithm to forecast the TATAGlobal's stocks price
# 

# In[ ]:





# In[ ]:





# Forecast TATAGlobal's stock price using Stacked LSTM.

# In[95]:


from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# Preparing the data
# 
# .The LSTM model will need data input in the form of X Vs y. Where the X will represent the last N day’s(15) prices and y will represent the N+1th day (16th-day) price.
# 
# .Since LSTM is a Neural network-based algorithm, standardizing or normalizing the data is mandatory for a fast and more accurate fit.
# 

# In[49]:


ts.head()


# In[50]:


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
scaler=scaler.fit(ts)
ts=scaler.transform(ts)
ts[:5]


# Preparing Data for LSTM
# 

# 
# 
# Split into Samples X and Y(Input X output Y)
# 

# In[51]:


x=[]
y=[]
no_of_rows=len(ts)

# next day's Price Prediction is based on last how many past day's prices
time_step=15

# iterate through values to create combination
for i in range(time_step, no_of_rows, 1):
    x0=ts[i-time_step:i]
    y0=ts[i]
    x.append(x0)
    y.append(y0)

# reshape the input to a 3D (no_of_sample, time_step, features)
x_data=np.array(x)
x_data=x_data.reshape(x_data.shape[0], x_data.shape[1], 1)
print("x data shape :", x_data.shape)

# reshape the output to 2D as it is supposed to single column
y_data=np.array(y)
y_data=y_data.reshape(y_data.shape[0],1)
print("y data shape: ", y_data.shape)


# 
# 
# Split the data into train and test
# 
# .Keeping last few days of data to test abd the learnings of the model and rest for training the model.
# .Here I am choosing Last 30 days as testin

# In[52]:


# choose number of testing data rows
test_rows=30

# split data into train & test
x_train=x_data[: - test_rows]
x_test=x_data[- test_rows:]
y_train=y_data[: - test_rows]
y_test=y_data[- test_rows:]
print("x train shape: ", x_train.shape)
print("y_train shape: ", y_train.shape)
print("x_test shape: ", x_test.shape)
print("y test shape: ", y_test.shape)


# Creating the Deep Learning LSTM model
# 
# .Use the LSTM function instead of Dense to define the hidden
# layers.
# 
# .The output layer has one neuron as we are predicting the next day price.

# In[53]:


# Defining Input shapes for LSTM
time_step=x_train.shape[1]
tot_feature=x_train.shape[2]
print("Number of TimeSteps:", time_step)
print("Number of Features:", tot_feature)


# In[96]:


# Initialising the RNN
model = Sequential()

# Adding the First input hidden layer and the LSTM layer
# return_sequences = True, means the output of every time step to be shared with hidden next layer.
model.add(LSTM(units = 10, activation = 'relu', input_shape = (time_step, tot_feature), return_sequences=True))
 
# Adding the Second Second hidden layer and the LSTM layer
model.add(LSTM(units = 5, activation = 'relu', input_shape = (time_step, tot_feature), return_sequences=True))
 
# Adding the Second Third hidden layer and the LSTM layer
model.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))
 
 
# Adding the output layer
model.add(Dense(units = 1))
 
# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')


# In[97]:


import time
# Measuring the time taken by the model to train
start_time=time.time()
 
# Fitting the RNN to the Training set
model.fit(x_train, y_train, batch_size = 10, epochs = 100)
 
end_time=time.time()
print("Total Time Taken: ", round((end_time-start_time)/60), 'Minutes')


# In[98]:


# Making predictions on test data
pred = model.predict(x_test)
pred = scaler.inverse_transform(pred)
 
# Getting the original price values for testing data
act=y_test
act=scaler.inverse_transform(y_test)
 
# Accuracy of the predictions
print('Accuracy:', 100 - (100*(abs(act-pred)/act)).mean())
print('RMSE:', np.sqrt(mse(act,pred)))


# In[99]:


rmse=np.sqrt(mse(act,pred)).round(2)
mpe=mape(act, pred).round(2)*100
lstm_result=pd.DataFrame({'Method':'LSTM', 'RMSE':[rmse], 'MAPE':[mpe]})
result=pd.concat([result,lstm_result])
result


# In[100]:


plt.plot(pred, color = 'blue', label = 'Predicted Closing Price')
plt.plot(act, color = 'green', label = 'Actual Closing Price')
 
plt.title('TATA Globels Stock Price Predictions')
plt.xlabel('Trading Date')
plt.xticks(range(test_rows), data.tail(test_rows).index, rotation=45)
plt.ylabel('Stock Price')
 
plt.legend()
fig=plt.gcf()
fig.set_figwidth(20)
fig.set_figheight(6)
plt.show()


# In[ ]:





# LSTM is best model to predict the TATA Global's Stock Price.

# In[ ]:




