import numpy as np
import pandas as pd
import pandas_datareader as web
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,LSTM

from GoldPredictionUsingNN import history
from InsuranceCostPrediction import x_train, y_train, x_test

plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/Users/amarnath/DS_ML_learning/src/MSFT.csv')

print(df.head())

plt.figure(figsize=(16,8))
plt.title('closing price of stock')
plt.plot(df['Close'])
plt.xlabel('Data')
plt.ylabel('Closing price')
plt.show()

data = df.filter(['Close'])
ds = data.values

training_data_len = math.cell(len(ds)*0.8)
print(training_data_len)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(ds)
print('Mean of processed data: ', scaled_data.mean())
print('Standard deviation of processed data: ', scaled_data.std())

train_data = scaled_data[0:training_data_len, :]
x_train=[]
y_train=[]

for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i])
    y_train.append(train_data[i])

x_train, y_train=np.array(x_train),np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))

model = Sequential()
model.add(LSTM(200,return_sequences=True,input_shape=(x_train.shape[1], 1)))
model.add(LSTM(200,return_sequences=False))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

history = model.fit(x_train, y_train, epochs=10)

plt.plot(history.history['loss'])

test_data = scaled_data[training_data_len-60, :]
x_test=[]
y_test=ds[training_data_len: ,:]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0 ])

x_test = np.array(x_test)

x_test= np.reshape(x_test,(x_test.shpe[0],x_test.shape[1],1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
print(predictions)