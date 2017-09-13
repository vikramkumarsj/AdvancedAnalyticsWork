# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
training_set = pd.read_csv('D:/Data/Retail_Soft_Drink_Sale.csv')
training_set = training_set.iloc[:,1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting the inputs and the ouputs
# Output is always as time t+1 , next month stock based on current month stock
X_train = training_set[0:104]
y_train = training_set[1:105]

# Reshaping
# 3 dimentional array includes no of observations, timesteps , no of input features
X_train = np.reshape(X_train, (104, 1, 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size = 1, epochs = 400)

# Part 3 - Making the predictions and visualising the results

# Getting the predicted soft drink sales
inputs = training_set[0:105]
inputs = np.reshape(inputs, (105, 1, 1))
predictions = regressor.predict(inputs)
predicted_Monthly_stock = sc.inverse_transform(predictions)

# Getting the real monthly stock
test_set = pd.read_csv('D:/Data/Retail_Soft_Drink_Sale.csv')
real_Monthly_stock = test_set.iloc[:,1:2].values

# Visualising the results
plt.plot(real_Monthly_stock, color = 'red', label = 'real_Monthly_stock')
plt.plot(predicted_Monthly_stock , color = 'blue', label = 'predicted_Monthly_stock')
plt.title('Monthy Retail Stock Prediction')
plt.xlabel('Time')
plt.ylabel('Monthly Stock')
plt.legend()
plt.show()

# Part 4 - Evaluating the RNN

import math
from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(real_Monthly_stock, predicted_Monthly_stock))






