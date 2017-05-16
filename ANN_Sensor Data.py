# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:49:43 2017

@author: 10639391
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = 'D:\Sensor_Data_TSV.TSV'
dataset = pd.read_table(path , sep = '\t')


X = dataset.drop(['id','cycle','label1','label2','RUL'], axis=1)

y = dataset.ix[:, ['label2']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 34, kernel_initializer = 'uniform', activation = 'relu', input_dim = 66))

classifier.add(Dense(units = 34, kernel_initializer = 'uniform', activation = 'relu'))

classifier.add(Dense(units = 3, kernel_initializer = 'uniform', activation = 'softmax'))

classifier.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 20, epochs = 30)

y_pred = classifier.predict(X_test)

y_pred_before_anomaly = y_pred[:,1]
y_pred_before_anomaly[y_pred_before_anomaly > 0.5] = 1
y_pred_before_anomaly[y_pred_before_anomaly <= 0.5] = 0

y_test[y_test == 2] = 0

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_before_anomaly)










