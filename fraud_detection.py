# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:37:27 2020

@author: emesh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import f1_score, recall_score


def PrintStats(cmat, y_test, pred):
   # separate out the confusion matrix components
   tpos = cmat[0][0]
   fneg = cmat[1][1]
   fpos = cmat[0][1]
   tneg = cmat[1][0]
   # calculate F!, Recall scores
   f1Score = round(f1_score(y_test, pred), 2)
   recallScore = round(recall_score(y_test, pred), 2)
   # calculate and display metrics
   print(cmat)
   print( 'Accuracy: '+ str(np.round(100*float(tpos+fneg)/float(tpos+fneg + fpos + tneg),2))+'%')
   print( 'Cohen Kappa: '+ str(np.round(cohen_kappa_score(y_test, pred),3)))
   print("Sensitivity/Recall for Model : {recall_score}".format(recall_score = recallScore))
   print("F1 Score for Model : {f1_score}".format(f1_score = f1Score))
   

def RunModel(model, X_train, y_train, X_test, y_test):
   model.fit(X_train, y_train.values.ravel())
   pred = model.predict(X_test)
   matrix = confusion_matrix(y_test, pred)
   return matrix, pred


data_frame = pd.read_csv("transaction_dataset.csv")
is_fraud = {0:'Not Fraud', 1:'Fraud'}
is_flagged_fraud = {0:'Not Fraud', 1:'Fraud'} 
print(data_frame.isFraud.value_counts().rename(index = is_fraud))
print(data_frame.isFlaggedFraud.value_counts().rename(index = is_flagged_fraud))

#columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig' , 'oldbalanceDest', 'newbalanceDest']

#features = df[columns]

#feature_names = data_frame.iloc[:, 1:12].columns
feature_names = ['amount', 'oldbalanceOrg', 'newbalanceOrig' , 'oldbalanceDest', 'newbalanceDest']
target = data_frame.iloc[:1, 9:10].columns

data_features = data_frame[feature_names]
data_target = data_frame[target]

#Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
np.random.seed(123)

X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30, random_state=1)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
cmat, pred = RunModel(lr, X_train, y_train, X_test, y_test)
PrintStats(cmat, y_test, pred)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, n_jobs =4)
cmat, pred = RunModel(rf, X_train, y_train, X_test, y_test)
PrintStats(cmat, y_test, pred)


#Undersampling
fraud_records = len(data_frame[data_frame.isFraud == 1])
fraud_indices = data_frame[data_frame.isFraud == 1].index
normal_indices = data_frame[data_frame.isFraud == 0].index

under_sample_indices = np.random.choice(normal_indices, fraud_records, False)

dataframe_undersampled = data_frame.iloc[np.concatenate([fraud_indices,under_sample_indices]),:]
X_undersampled = dataframe_undersampled.iloc[:, [2, 4, 5, 7, 8]]
Y_undersampled = dataframe_undersampled.isFraud
X_undersampled_train, X_undersampled_test, Y_undersampled_train, Y_undersampled_test = train_test_split(X_undersampled,Y_undersampled,test_size = 0.3)
lr_undersampled = LogisticRegression(C=1)

# run the new model
cmat, pred = RunModel(lr_undersampled, X_undersampled_train, Y_undersampled_train, X_undersampled_test, Y_undersampled_test)
PrintStats(cmat, Y_undersampled_test, pred)



