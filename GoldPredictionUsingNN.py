from pickletools import optimize

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.python.framework.test_ops import test_string_output_eager_fallback
from tensorflow.python.keras import Sequential

from InsuranceCostPrediction import y_train

df = pd.read_csv("/Users/amarnath/DS_ML_learning/src/gld_price_data.csv")

print(df.head())

x = df[['SPX', 'USD', 'SLV', 'EUR/USD']]
y = df['GLD']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)

scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


model = Sequential()

model.add(Dense(10, actiavtion='relu', input_dim=4))
model.add(Dense(10, activation='relu'))
model.add(Dense(1,activation='linear'))

model.compile(loss="mean_squared_error",optimizer = 'Adam')
history = model.fit(x_train_scaled, y_train, epochs=50, validation_split=0.1)

pred = model.predict(x_test_scaled)

print(r2_score(y_test,pred))
