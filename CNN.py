import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10

from GoldPredictionUsingNN import history

(x_train, y_train), (x_test,y_test) = cifar10.load_data()

x_train - x_train/255
x_test = x_test/255

y_train_en  = to_categorical(y_train,10)
y_train_en = to_categorical(y_test,10)

model = Sequential()
model.add(Conv2D(32,(4,4), input_shape=(32,32,3),activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy_score'])

model.summary()

history = model.fit(x_train, y_train_en, epochs=20, verbose=1, validation_data=(x_test, y_train_en))
