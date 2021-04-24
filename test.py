# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 18:53:43 2021

@author: laker
"""

import numpy as np
import pandas as pd
# For visualizations
import matplotlib.pyplot as plt
# For regular expressions
import re
# For handling string
import string
# For performing mathematical operations
import math

# Importing dataset
df=pd.read_csv("C:\\Users\\laker\\Desktop\\pc2\\cs_class\\CAP6135\\6135-term proj\\Car_Hacking_Challenge_Dataset\\0_Preliminary\\0_Training\\traindata.csv") 
print("Shape of data=>",df.shape)

df.head()

df.dtypes
#Timestamp          int64
#Arbitration_ID    object
#DLC                int64
#Data              object
#Class             object
#SubClass          object
#dtype: object

type(df['Data'])
type(df['Data'][1])
len(df['Data'][0])
len(df['Data'][1])

x=df['Data'][1].split()
len(x)
x.shape  # list
x[0]
x1=pd.DataFrame(x)
x1.shape

type(df['Timestamp'][1])
type(df['Arbitration_ID'][0])
type(df['DLC'][0])
max(df['Timestamp'])
min(df['Timestamp'])

df1=df
df1['Timestamp']=df['Timestamp']-min(df['Timestamp'])
df1['Timestamp'].head()

int('A',16)
int(df1['Arbitration_ID'][0],16)
len(df1['Arbitration_ID'])

#convert str to int
for i in range(len(df1['Arbitration_ID'])):
    df1['Arbitration_ID'][i]=int(df['Arbitration_ID'][i],16)
    
df1['Arbitration_ID'].head()
df1.head()
#extract rightmost four-bit of data
df1['d1']=0
df1['d2']=0
df1['d3']=0
df1['d4']=0
for i in range(len(df1['Data'])):
    temp=df['Data'][i].split()
    df1['d1'][i]=int(temp[len(temp)-1],16)
    df1['d2'][i]=int(temp[len(temp)-2],16)
    df1['d3'][i]=int(temp[len(temp)-3],16)
    df1['d4'][i]=int(temp[len(temp)-4],16)

from sklearn.model_selection import train_test_split

# Using subclass as y
x=df1[['Timestamp','Arbitration_ID','DLC','d1','d2','d3','d4']]
y=df1[['SubClass']]
x.shape
y.shape
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.33, random_state=4)

#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
model = LogisticRegression(solver='liblinear', random_state=4)
model.fit(x_train, y_train)
model.score(x_train, y_train) # 0.9210686431771669
model.score(x_test, y_test) #0.9233101280035931

confusion_matrix(y_test, model.predict(x_test))
#array([[1390,    0,    0,    0,    0],
#       [   0, 1308,    0,  111,    6],
#       [   0,    0, 4487,    0,    0],
#       [   0,  310,    0,  881,   97],
#       [   0,    0,    0,  159,  157]], dtype=int64)

#visualize the confusion matrix.
cm = confusion_matrix(y_test, model.predict(x_test))

fig, ax = plt.subplots(figsize=(10, 20))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1,2,3,4), ticklabels=('Predicted Flooding', 'Predicted Fuzzing','Predicted Normal','Predicted Replay','Predicted Spoofing'))
ax.yaxis.set(ticks=(0, 1,2,3,4), ticklabels=('actual Flooding', 'actual Fuzzing','actual Normal','actual Replay','actual Spoofing'))
ax.set_ylim(4.5, -0.5)
for i in range(5):
    for j in range(5):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.show()

print(classification_report(y_test, model.predict(x_test)))
#              precision    recall  f1-score   support

#    Flooding       1.00      1.00      1.00      1390
#     Fuzzing       0.81      0.92      0.86      1425
#      Normal       1.00      1.00      1.00      4487
#      Replay       0.77      0.68      0.72      1288
#    Spoofing       0.60      0.50      0.55       316

#    accuracy                           0.92      8906
#   macro avg       0.84      0.82      0.83      8906
#weighted avg       0.92      0.92      0.92      8906

# Using class as y
x=df1[['Timestamp','Arbitration_ID','DLC','d1','d2','d3','d4']]
y=df1[['Class']]
x.shape
y.shape
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.33, random_state=4)

#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
model = LogisticRegression(solver='liblinear', random_state=4)
model.fit(x_train, y_train)
model.score(x_train, y_train) # 1.0
model.score(x_test, y_test) #1.0

confusion_matrix(y_test, model.predict(x_test))
#array([[4419,    0],
#       [   0, 4487]], dtype=int64)

#visualize the confusion matrix.
cm = confusion_matrix(y_test, model.predict(x_test))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Attack', 'Predicted Normal'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Attack', 'Actual Normal'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()

print(classification_report(y_test, model.predict(x_test)))
#              precision    recall  f1-score   support

#      Attack       1.00      1.00      1.00      4419
#     Normal       1.00      1.00      1.00      4487

#    accuracy                           1.00      8906
#   macro avg       1.00      1.00      1.00      8906
#weighted avg       1.00      1.00      1.00      8906

#####################################################################
#random forest
from sklearn.ensemble import RandomForestRegressor
  
 # create regressor object
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
  
# fit the regressor with x and y data
# Using class as y
x=df1[['Timestamp','Arbitration_ID','DLC','d1','d2','d3','d4']]
y=df1[['Class']]
yy=y
yy[yy['Class']=='Normal']=1
yy[yy['Class']=='Attack']=0
yy.shape
yy.head()
x_train, x_test, y_train, y_test = train_test_split(x, yy,test_size=0.33, random_state=4)
regressor.fit(x_train, y_train)  
regressor.score(x_train, y_train) #1.0
regressor.score(x_test, y_test)#1.0
# Using subclass as y
x=df1[['Timestamp','Arbitration_ID','DLC','d1','d2','d3','d4']]
y=df1[['SubClass']]  #'Flooding', 'Fuzzing','Normal','Replay','Spoofing'
yy=y
yy[yy['SubClass']=='Flooding']=0
yy[yy['SubClass']=='Fuzzing']=1
yy[yy['SubClass']=='Normal']=2
yy[yy['SubClass']=='Replay']=3
yy[yy['SubClass']=='Spoofing']=4
yy.shape
yy.head()
x_train, x_test, y_train, y_test = train_test_split(x, yy,test_size=0.33, random_state=4)
regressor.fit(x_train, y_train)  
regressor.score(x_train, y_train) #0.9999830023319188
regressor.score(x_test, y_test)#0.9999643716638226

#The impurity-based feature importances.

#The higher, the more important the feature. 
#The importance of a feature is computed 
#as the (normalized) total reduction of the criterion brought by that feature. 
#It is also known as the Gini importance.
regressor.feature_importances_
#array([4.42505160e-01, 5.56269234e-01, 0.00000000e+00, 3.20685733e-04,
#       2.44540216e-05, 4.45946587e-05, 8.35872300e-04])

#from sklearn.metrics import plot_roc_curve
#svc_disp = plot_roc_curve(regressor, x_test, y_test)
#plt.show()

#######################
# CNN
# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

...
# define the keras model
#The model expects rows of data with 8 variables (the input_dim=8 argument)
#The first hidden layer has 12 nodes and uses the relu activation function.
#The second hidden layer has 8 nodes and uses the relu activation function.
#The output layer has one node and uses the sigmoid activation function.
model = Sequential()
model.add(Dense(12, input_dim=7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
# Using class as y
x=df1[['Timestamp','Arbitration_ID','DLC','d1','d2','d3','d4']]
y=df1[['Class']]
yy=y
yy[yy['Class']=='Normal']=1
yy[yy['Class']=='Attack']=0
yy.shape
yy.head()
x_train, x_test, y_train, y_test = train_test_split(np.asarray(x).astype('float32'), np.asarray(yy).astype('float32'),test_size=0.33, random_state=4)
model.fit(x_train, y_train, epochs=10, batch_size=10)  
#loss: 6.9704e-06 - accuracy: 1.0000

# evaluate the keras model
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100)) 
 #loss: 2.6429e-04 - accuracy: 0.9999  Accuracy: 99.99
 
# Using subclass as y
model = Sequential()
model.add(Dense(12, input_dim=7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
x=df1[['Timestamp','Arbitration_ID','DLC','d1','d2','d3','d4']]
y=df1[['SubClass']]  #'Flooding', 'Fuzzing','Normal','Replay','Spoofing'
yy=y
yy[yy['SubClass']=='Flooding']=0
yy[yy['SubClass']=='Fuzzing']=1
yy[yy['SubClass']=='Normal']=2
yy[yy['SubClass']=='Replay']=3
yy[yy['SubClass']=='Spoofing']=4
yy.shape
yy.head()
x_train, x_test, y_train, y_test = train_test_split(np.asarray(x).astype('float32'), np.asarray(yy).astype('float32'),test_size=0.33, random_state=4)
model.fit(x_train, y_train, epochs=10, batch_size=10)  
#loss: -228671888.0000 - accuracy: 0.1613


#loss: 6.9704e-06 - accuracy: 1.0000

# evaluate the keras model
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100)) #accuracy: 0.16


##########################
#SVM
#Import svm model
from sklearn import svm

# Using class as y
x=df1[['Timestamp','Arbitration_ID','DLC','d1','d2','d3','d4']]
y=df1[['Class']]
y.shape
yy=y
yy[yy['Class']=='Normal']=1
yy[yy['Class']=='Attack']=0
yy.shape
yy.head()
df.values.tolist()
x_train, x_test, y_train, y_test = train_test_split(x.values.tolist(),yy.values.tolist(),test_size=0.33, random_state=4)
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))  #Accuracy: 1.0

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred)) #Precision: 1.0

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred)) #Recall: 1.0

# Using subclass as y
x=df1[['Timestamp','Arbitration_ID','DLC','d1','d2','d3','d4']]
y=df1[['SubClass']]
y.shape
yy=y
yy[yy['SubClass']=='Flooding']=0
yy[yy['SubClass']=='Fuzzing']=1
yy[yy['SubClass']=='Normal']=2
yy[yy['SubClass']=='Replay']=3
yy[yy['SubClass']=='Spoofing']=4
yy.shape
yy.head()

x_train, x_test, y_train, y_test = train_test_split(x.values.tolist(),yy.values.tolist(),test_size=0.33, random_state=4)
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))  #Accuracy:  0.9839434089377948

y_pred_train = clf.predict(x_train)
print("Accuracy:",metrics.accuracy_score(y_train, y_pred_train))  #Accuracy:  0.9845677305160684
