# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 20:17:45 2023

@author: mrslk
"""
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
training_set = df.iloc[:-48, 1:2].values  # Use all data except the last 48 points for training
test_set = df.iloc[-48:, 1:2].values  # Use the last 48 points for testing

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
model_P.save('LSTM-univariate')

# Load the model
from keras.models import load_model
model_P = load_model('LSTM-univariate')

# Fit the model
history = model_P.fit(x_train, y_train, epochs=10, batch_size=32)

# Plot the training loss
plt.plot(range(len(history.history['loss'])), history.history['loss'])
plt.xlabel('Epoch number')
plt.ylabel('Loss')
plt.show()

# Predictions
prediction_test = []
batch_one = training_set_scaled[-windowsize:]
batch_new = batch_one.reshape(1, windowsize, 1)

import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, r2_score

# Assuming you have defined the variables batch_one, windowsize, model_P, test_set, and sc

prediction_test = []

for i in range(720): # one month 
    batch_one_reshaped = batch_one.reshape(1, windowsize, 1)
    first_pred = model_P.predict(batch_one_reshaped)[0, 0]
    prediction_test = np.append(prediction_test, first_pred)
    
    # Update batch_new with the new prediction
    batch_new = np.concatenate((batch_new[:, 1:, :], first_pred.reshape(1, 1, 1)), axis=1)
    
    # Update batch_one for the next iteration
    batch_one = batch_new[:, -windowsize:, :]

# Reshape prediction_test to match the shape of the original data
prediction_test = prediction_test.reshape(-1, 1)

# Inverse transform the scaled predictions
prediction = sc.inverse_transform(prediction_test)

plt.plot(test_set, color='red', label='Actual values')
plt.plot(prediction, color='blue', label='Predicted values')
plt.title('LSTM Univariate forecast')
plt.xlabel('time(h)')
plt.ylabel('solar Irradiance')
plt.legend()
plt.show()

# Reshape prediction to match the shape of the original data
prediction = prediction[:len(test_set)]

# Calculate evaluation metrics
RMSE = math.sqrt(mean_squared_error(test_set, prediction))
Rsquare = r2_score(test_set, prediction)

print(f"Root Mean Squared Error (RMSE): {RMSE}")
print(f"R-squared (R^2): {Rsquare}") 





 



