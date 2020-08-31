# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:46:41 2020

@author: Rajesh Sharma
"""
## Import the required libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## Import Generic Datasplit Package
from Generic_Functions import train_test_datasets as ttd
from Generic_Functions import minority_upsampling as upsample

## Cross validation Packages
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

## Import the Estimators
from sklearn.ensemble import RandomForestClassifier

from yellowbrick.classifier import confusion_matrix as conf_matrix

## Model Metrics evaluation packages
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
from yellowbrick import ROCAUC

## Ignoring the warnings by messages
import warnings
warnings.filterwarnings("ignore")

pre_proc_df = pd.read_csv(os.path.join('Cleaned_Trans_Scaled_Features.csv'))

cv_dataset, unseen_dataset = ttd(pre_proc_df,train_size=0.85,test_size=0.15,random_state=0)

cv_dataset_upsampled = upsample(cv_dataset,minority_label=0,random_state=0)

# rfc = RandomForestClassifier()
# rfc = RandomForestClassifier(n_estimators=25,max_depth=16,min_samples_split=3,min_samples_leaf=1,max_features='sqrt')
# skf = StratifiedKFold(n_splits=10)

# print(cross_val_score(estimator=rfc,X=cv_dataset_upsampled.iloc[:,0:-1],y=cv_dataset_upsampled.iloc[:,-1],scoring='f1',cv=skf).mean())

# print(cross_val_score(estimator=rfc,X=cv_dataset_upsampled.iloc[:,0:-1],y=cv_dataset_upsampled.iloc[:,-1],scoring='precision',cv=skf).mean())
    
# print(cross_val_score(estimator=rfc,X=cv_dataset_upsampled.iloc[:,0:-1],y=cv_dataset_upsampled.iloc[:,-1],scoring='recall',cv=skf).mean())

# estimators = [15,20,25,30,35]
# max_depth = [4,8,16,32]
# min_samples_split = [2,3,4,5]
# min_samples_leaf = [1,2,3,4]
# max_features = ['auto','sqrt','log2']

# param_grid = dict(n_estimators=estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)

# grid = GridSearchCV(rfc,param_grid=param_grid,scoring=['f1','precision','recall'],cv=skf,n_jobs=-1,refit=False)

# grid.fit(cv_dataset_upsampled.iloc[:,0:-1],cv_dataset_upsampled.iloc[:,-1])

# pd.DataFrame(grid.cv_results_).to_csv('HP_results.csv')

rfc_model = RandomForestClassifier(n_estimators=25,max_depth=16,min_samples_split=3,min_samples_leaf=1,max_features='sqrt',random_state=0)

rfc_model.fit(cv_dataset_upsampled.iloc[:,0:-1],cv_dataset_upsampled.iloc[:,-1])

y_pred = rfc_model.predict(unseen_dataset.iloc[:,0:-1])
 
print(f1_score(unseen_dataset.iloc[:,-1],y_pred))

print(precision_score(unseen_dataset.iloc[:,-1],y_pred))

print(recall_score(unseen_dataset.iloc[:,-1],y_pred))

print(confusion_matrix(unseen_dataset.iloc[:,-1],y_pred))

# visualizer = conf_matrix(rfc_model, X_train=dX_train, y_train=dy_train, X_test=dX_test, y_test=dy_test, cmap="Greens")
