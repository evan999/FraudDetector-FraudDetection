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


df = pd.read_csv("transaction_dataset.csv")
is_fraud = {0:'Not Fraud', 1:'Fraud'}
is_flagged_fraud = {0:'Not Fraud', 1:'Fraud'} 
is_flagged_fraud
print(df.isFraud.value_counts().rename(index = is_fraud))


columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig' , 'oldbalanceDest', 'newbalanceDest']

features = dataset[columns]
