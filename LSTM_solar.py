# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:17:45 2023

@author: mrslk
"""
#preprocessing
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt 

# Load the data
df = pd.read_csv("Solar.csv")
df.dropna(inplace=True)

# Extract the training and test sets
training_set = df.iloc[:8712, 1:2].values
test_set = df.iloc[:8712, 1:2].values

# Feature scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
test_set_scaled = sc.transform(test_set)

# Creating the training data
x_train = []
y_train = []
windowsize = 24

for i in range(windowsize, len(training_set_scaled)):
    x_train.append(training_set_scaled[i - windowsize:i, 0:1])
    y_train.append(training_set_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Developing LSTM
model_P = Sequential()
model_P.add(LSTM(units=60, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model_P.add(Dropout(0.2))
model_P.add(LSTM(units=60, return_sequences=True))
model_P.add(Dropout(0.2)) 
model_P.add(LSTM(units=60, return_sequences=True))
model_P.add(Dropout(0.2)) 
model_P.add(LSTM(units=60))
model_P.add(Dropout(0.2)) 
model_P.add(Dense(units=1))

model_P.compile(optimizer='adam', loss='mean_squared_error')

# Save the model
model_P.save('LSTM-univeriat')

# Load the model
from keras.models import load_model
model_P = load_model('LSTM-univeriat')

# Fit the model
history = model_P.fit(x_train, y_train, epochs=30, batch_size=32)

# Plot the training loss
plt.plot(range(len(history.history['loss'])), history.history['loss'])
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.show()





 



