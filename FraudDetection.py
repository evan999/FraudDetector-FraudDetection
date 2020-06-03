# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 19:22:55 2020

@author: emesh
"""


import matplotlib.pyplot as plt
import pandas as pd

from Flask import Flask, request
from Flask_json import FlaskJSON, JsonError, json_response

# PREPARING THE MODEL

#Importing the dataset

dataset = pd.read_csv('transaction_dataset.csv')

from sklearn.preprocessing import StandardScaler
columns = ['amount', 'oldBalanceOrig', 'newBalanceOrig' , 'oldBalanceDest', 'newBalanceDest']

features = dataset[columns]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)

dataset[columns] = features

x = dataset.iloc[:, [1, 2, 4, 5, 7, 8]].values
y = dataset.iloc[:, 9].values

# Encoding categorical data

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

A = make_column_transformer(
    (OneHotEncoder(categories='auto'), [0]),
    remainder = 'passThrough')    
    
x = A.fit_transform(x)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
regressor.fit_transform

# Feature Scaling

"""from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)"""




