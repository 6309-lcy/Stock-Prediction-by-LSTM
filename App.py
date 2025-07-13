import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import keras
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error


st.set_page_config(layout="wide")


model1 = keras.models.load_model("LSTM_3_Layers.keras")
model2 = keras.models.load_model("LSTM_3_Layers_MINMAX.keras")

word = r'''
$\textsf{\Huge Stock Market Predictor}$
'''
st.header(word)
interval_option = ['1d', '5d', '1wk', '1mo', '3mo']
word = r'''
$\textsf{\Huge Enter a Stock Symbol}$
'''
stock = st.text_input(word, "TSLA")
starts = '2015-01-01'
ends = datetime.today()
word = r'''
$\textsf{\Huge Choose the interval of the data}$
'''
intervals = st.selectbox(word, options=interval_option)
data = yf.download(stock, start=starts, end=ends, interval=intervals, multi_level_index=False)

st.markdown("# Stock Data, starting from 2015-01-01")
st.write(data)


scalar = StandardScaler()
scalar2 = MinMaxScaler()
data_train = pd.DataFrame(data.Close[0:int(len(data)*0.85)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.85): len(data)])
past_60_days = data_train.tail(60)
data_test = pd.concat([past_60_days, data_test])
scalar.fit(data_train)
scalar_value_train = scalar.transform(data_train)
scalar_value_test = scalar.transform(data_test)
scalar2.fit(data_train)
scalar2_value_train = scalar2.transform(data_train)
scalar2_value_test = scalar2.transform(data_test)


x, y = [], []
for i in range(60, len(scalar_value_test)):
    x.append(scalar_value_test[i-60:i])
    y.append(scalar_value_test[i])
X, y = np.array(x), np.array(y)

x2, y2 = [], []
for i in range(60, len(scalar2_value_test)):
    x2.append(scalar2_value_test[i-60:i])
    y2.append(scalar2_value_test[i])
X2, y2 = np.array(x2), np.array(y2)


y_predict = model1.predict(X)
y_predict = scalar.inverse_transform(y_predict)
y = scalar.inverse_transform(y.reshape(-1, 1))
y2_predict = model2.predict(X2)
y2_predict = scalar2.inverse_transform(y2_predict)
y2 = scalar2.inverse_transform(y2.reshape(-1, 1))

test_date = data.index[int(len(data)*0.85)-60:]


future_input = scalar_value_test[-60:].reshape(1, 60, 1)
future_input2 = scalar2_value_test[-60:].reshape(1, 60, 1)

future_preds = []
future_preds2 = []
for i in range(7):
    pred = model1.predict(future_input)
    future_preds.append(pred[0, 0])
    new_input = np.append(future_input.ravel(), pred[0, 0])
    new_input = new_input[1:]
    future_input = new_input.reshape(1, 60, 1)

future_preds = np.array(future_preds).reshape(-1, 1)
future_preds = scalar.inverse_transform(future_preds)

for i in range(7):
    pred = model2.predict(future_input2)
    future_preds2.append(pred[0, 0])
    new_input = np.append(future_input2.ravel(), pred[0, 0])
    new_input = new_input[1:]
    future_input2 = new_input.reshape(1, 60, 1)

future_preds2 = np.array(future_preds2).reshape(-1, 1)
future_preds2 = scalar2.inverse_transform(future_preds2)


if len(test_date) != len(y_predict) or len(test_date) != len(y2_predict):
    min_len = min(len(test_date), len(y_predict), len(y2_predict))
    test_date = test_date[:min_len]
    y_predict = y_predict[:min_len]
    y2_predict = y2_predict[:min_len]
    y = y[:min_len]
    y2 = y2[:min_len]


st.markdown("## Using the last 15% of the dataset  to predict")
rmse1 = np.sqrt(mean_squared_error(y, y_predict))
mae1 = mean_absolute_error(y, y_predict)
rmse2 = np.sqrt(mean_squared_error(y2, y2_predict))
mae2 = mean_absolute_error(y2, y2_predict)

st.markdown(f"**StandardScaler** - RMSE: {rmse1:.2f}, MAE: {mae1:.2f}")
st.markdown(f"**MinMaxScaler** - RMSE: {rmse2:.2f}, MAE: {mae2:.2f}")


col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Closing Price")
    fig1 = plt.figure(figsize=(15, 7), dpi=100)
    plt.plot(data.index, data.Close, 'b')
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.tight_layout(pad=2.0)
    st.pyplot(fig1)
    plt.close(fig1)

with col2:
    st.subheader("Closing Price with 100ma, 200ma")
    ma100 = data.Close.rolling(100).mean()
    ma200 = data.Close.rolling(200).mean()
    fig2 = plt.figure(figsize=(15, 7), dpi=100)
    plt.plot(data.index, data.Close, 'b', label="Close Price")
    plt.plot(data.index, ma100, 'orange', label='100ma')
    plt.plot(data.index, ma200, 'green', label='200ma')
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.tight_layout(pad=2.0)
    st.pyplot(fig2)
    plt.close(fig2)


col3, col4 = st.columns([1, 1])
with col3:
    st.subheader("Historical Prediction using StandardScaler")
    fig3 = plt.figure(figsize=(15, 7), dpi=100)
    plt.plot(test_date, y_predict, 'r', label='Predicted Price')
    plt.plot(test_date, y, 'g', label='Actual Price')
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.tight_layout(pad=2.0)
    st.pyplot(fig3)
    plt.close(fig3)

with col4:
    st.subheader("Historical Prediction using MinMaxScaler")
    fig4 = plt.figure(figsize=(15, 7), dpi=100)
    plt.plot(test_date, y2_predict, 'b', label='Predicted Price')
    plt.plot(test_date, y2, 'g', label='Actual Price')
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.tight_layout(pad=2.0)
    st.pyplot(fig4)
    plt.close(fig4)


col5, col6 = st.columns([1, 1])
with col5:
    st.subheader("Future Prediction using StandardScaler (7 Days")
    fig5 = plt.figure(figsize=(15, 7), dpi=100)
    
    plt.plot(test_date, y_predict, 'r', label="Predicted (Historical)")
    plt.plot(test_date, y[:len(y_predict)], 'g', label='Actual Price')
    
    future_dates = [test_date[-1] + timedelta(days=i+1) for i in range(7)]
    plt.plot(future_dates, future_preds.ravel(), 'm--', label='Future Forecast', linewidth=2.0)
    
    plt.fill_between(future_dates, future_preds.ravel().min(), future_preds.ravel().max(), color='m', alpha=0.1)
    
    plt.axvline(x=test_date[-1], color='k', linestyle='--', linewidth=1.0)
    plt.text(test_date[-1], future_preds.ravel().max(), 'Future', fontsize=12, ha='right', va='bottom')
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.tight_layout(pad=2.0)
    st.pyplot(fig5)
    plt.close(fig5)

with col6:
    st.subheader("Future Prediction using MinMaxScaler (7 Days")
    fig6 = plt.figure(figsize=(15, 7), dpi=100)
    
    plt.plot(test_date, y2_predict, 'b', label="Predicted (Historical)")
    plt.plot(test_date, y2[:len(y2_predict)], 'g', label='Actual Price')
    
    future_dates = [test_date[-1] + timedelta(days=i+1) for i in range(7)]
    plt.plot(future_dates, future_preds2.ravel(), 'c--', label='Future Forecast', linewidth=2.0)
    
    plt.fill_between(future_dates, future_preds2.ravel().min(), future_preds2.ravel().max(), color='c', alpha=0.1)
    
    plt.axvline(x=test_date[-1], color='k', linestyle='--', linewidth=1.0)
    plt.text(test_date[-1], future_preds2.ravel().max(), 'Future', fontsize=12, ha='right', va='bottom')
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.tight_layout(pad=2.0)
    st.pyplot(fig6)
    plt.close(fig6)
