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
from Generic_Functions import plot_learning_curve as plc

## Cross validation Packages
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

## Import the Estimators
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from xgboost import XGBRFClassifier, XGBClassifier
from sklearn.linear_model import LogisticRegression

from yellowbrick.classifier import confusion_matrix as conf_matrix

## Model Metrics evaluation packages
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score
from yellowbrick import ROCAUC

## Ignoring the warnings by messages
import warnings
warnings.filterwarnings("ignore")

from sklearn.utils import compute_class_weight
from sklearn.ensemble import BaggingClassifier
from imblearn.over_sampling import SMOTE
import xgboost

pre_proc_df = pd.read_csv(os.path.join('Cleaned_Trans_Scaled_Features.csv'))

# y_0 = np.zeros(230)
# y_1 = np.ones(416)
# classes = [0,1]
# y = np.concatenate([y_0,y_1])
# cw = compute_class_weight('balanced',classes,y)
# print(cw)
# print(cw/2.11)

cv_dataset, unseen_dataset = ttd(pre_proc_df,train_size=0.80,test_size=0.20,random_state=91)

# cv_dataset_upsampled = upsample(cv_dataset,minority_label=0,random_state=19)
# rfc_model = RandomForestClassifier()
rfc_model = RandomForestClassifier(n_estimators=25,random_state=34,
                                    max_depth=16,
                                    min_samples_split=2,
                                    # class_weight={0:0.66,1:0.37},
                                    min_samples_leaf=2,
                                    max_features='auto')
# rfc = RandomForestClassifier(n_estimators=25,max_depth=16,min_samples_split=3,min_samples_leaf=1,max_features='sqrt')
# skf = StratifiedKFold(n_splits=10)
# cvk = StratifiedShuffleSplit(n_splits=10, test_size=0.20, random_state=0)

# print(cross_val_score(estimator=rfc_model,X=cv_dataset.iloc[:,0:-1],y=cv_dataset.iloc[:,-1],scoring='f1',cv=cvk).mean())

# print(cross_val_score(estimator=rfc_model,X=cv_dataset.iloc[:,0:-1],y=cv_dataset.iloc[:,-1],scoring='precision',cv=cvk).mean())
    
# print(cross_val_score(estimator=rfc_model,X=cv_dataset.iloc[:,0:-1],y=cv_dataset.iloc[:,-1],scoring='recall',cv=cvk).mean())

# n_estimators = [20,25,30,35,40,45,50,75]
# max_depth = [4,8,16,32,64]
# min_samples_split = [2,3,4,5]
# min_samples_leaf = [1,2,3,4]
# max_features = ['auto','sqrt','log2']
# max_leaf_nodes = [1,2]

# param_grid = dict(n_estimators=estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features)

# grid = GridSearchCV(rfc,param_grid=param_grid,scoring=['f1','precision','recall'],cv=skf,n_jobs=-1,refit=False)

# grid.fit(cv_dataset_upsampled.iloc[:,0:-1],cv_dataset_upsampled.iloc[:,-1])

# pd.DataFrame(grid.cv_results_).to_csv('HP_results.csv')

# print(cross_val_score(estimator=rfc_model,X=cv_dataset.iloc[:,0:-1],y=cv_dataset.iloc[:,-1],scoring='f1',cv=cv).mean())

# print(cross_val_score(estimator=rfc_model,X=cv_dataset.iloc[:,0:-1],y=cv_dataset.iloc[:,-1],scoring='precision',cv=cv).mean())
    
# print(cross_val_score(estimator=rfc_model,X=cv_dataset.iloc[:,0:-1],y=cv_dataset.iloc[:,-1],scoring='recall',cv=cv).mean())

rfc_model.fit(cv_dataset.iloc[:,0:-1],cv_dataset.iloc[:,-1])

y_pred = rfc_model.predict(unseen_dataset.iloc[:,0:-1])
 
print(f1_score(unseen_dataset.iloc[:,-1],y_pred))

print(precision_score(unseen_dataset.iloc[:,-1],y_pred))

print(recall_score(unseen_dataset.iloc[:,-1],y_pred))

print(confusion_matrix(unseen_dataset.iloc[:,-1],y_pred))

# visualizer = conf_matrix(rfc_model, X_train=cv_dataset.iloc[:,0:-1],
#                          y_train=cv_dataset.iloc[:,-1],
#                          X_test=unseen_dataset.iloc[:,0:-1], y_test=unseen_dataset.iloc[:,-1], cmap="Greens")

# roc_rfc = ROCAUC(rfc_model)
# roc_rfc.fit(cv_dataset.iloc[:,0:-1],cv_dataset.iloc[:,-1])
# roc_rfc.predict()
# roc_rfc.show()

# cv = StratifiedShuffleSplit(n_splits=100, test_size=0.20, random_state=0)
# from sklearn.model_selection import validation_curve
# train_scores, test_scores = validation_curve(estimator=rfc_model,
#                                               X=cv_dataset.iloc[:,0:-1], 
#                                               y=cv_dataset.iloc[:,-1],
#                                               param_name='max_features',
#                                               param_range=max_features,
#                                               cv=cv,scoring='recall')

# # Calculate mean and standard deviation for training set scores
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)

# # Calculate mean and standard deviation for test set scores
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)

# # Plot mean accuracy scores for training and test sets
# plt.plot(max_features, train_mean, label="Training score", color="black")
# plt.plot(max_features, test_mean, label="Cross-validation score", color="dimgrey")

# # Plot accurancy bands for training and test sets
# plt.fill_between(max_features, train_mean - train_std, train_mean + train_std, color="gray")
# plt.fill_between(max_features, test_mean - test_std, test_mean + test_std, color="gainsboro")

# # Create plot
# plt.title("Validation Curve With Random Forest")
# plt.xlabel("max_features")
# plt.ylabel("recall Score")
# plt.tight_layout()
# plt.legend(loc="best")
# plt.show()

# fig, axes = plt.subplots(3, 2, figsize=(10, 15))
# plc(rfc_model,"Learning Curves for RF",X=cv_dataset.iloc[:,0:-1], y=cv_dataset.iloc[:,-1],
#     axes=axes[:, 0], ylim=(0.5, 1.01),cv=cv, n_jobs=-1)
# plt.show()



