# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:34:09 2020

@author: Rajesh Sharma
"""
## Import the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

## Data Sampling Package
from sklearn.utils import resample

def val_iqr_limits(df_name,col_name,w_width=None):
    """
    Description: This function is created for calculating the upper and lower limits using Tuky's IQR method.
    
    Input parameters: It accepts below two input parameters:
        1. df_name: DataFrame
        2. col_name: Feature name
        3. w_width: Whisker width provided by user and by default 1.5 
        
    Return: It returns the median, upper and lower limits of the feature based on Tuky's IQR method.
    """
    if w_width == None:
        w_width = 1.5
    else:
        w_width = w_width
        
    val_median = df_name[col_name].median()
    q1 = df_name[col_name].quantile(0.25)
    q3 = df_name[col_name].quantile(0.75)
    iqr = q3 - q1
    lower_limit = q1 - (w_width*iqr)
    upper_limit = q3 + (w_width*iqr)
#     print(val_median,q1,q3,iqr,lower_limit,upper_limit)     ## Uncomment if you want to see the values of median, q1, q2, iqr, lower and upper limit 
    return val_median, upper_limit, lower_limit

def fix_outliers(df_name,col_name,whis_width=None):
    """
    Description: This function is created for applyng the Tuky's IQR method on variable.
    
    Input parameters: It accepts the below two parameters:
        1. df_name: DataFrame
        2. col_name: Feature name
        3. whis_width: Whisker width provided by user and by default 1.5 
    
    Return: It returns the modified feature with the removed outliers.
    """
    print("######## Applied Tuky IQR Method-I ########")
    v_median, upr_limit , low_limit = val_iqr_limits(df_name,col_name,whis_width)
    df_name[col_name] = df_name[col_name].apply(lambda val: low_limit + (val-upr_limit) if val > upr_limit 
                                                else upr_limit - (low_limit-val) if val < low_limit else val)
    
    print("######## Applied Tuky IQR Method-II ########\n")
    v1_median, upr_limit1, low_limit1 = val_iqr_limits(df_name,col_name,whis_width)
    
    # df_name[col_name] = df_name[col_name].apply(lambda val: upr_limit1 if val > upr_limit1 
    #                                             else low_limit1 if val < low_limit1 else val)
    df_name[col_name] = df_name[col_name].apply(lambda val: low_limit1 + (val-upr_limit1) if val > upr_limit1 
                                                else upr_limit1 - (low_limit1-val) if val < low_limit1 else val)

def plot_data(df_name):
    """
    This function is plotting the box plot of the dataframe.

    Parameters
    ----------
    df_name : DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    with plt.style.context('seaborn'): 
        sns.boxplot(data=df_name.iloc[:,:])
    plt.show()
    
    with plt.style.context('seaborn'):
        sns.heatmap(df_name.corr(),cmap='coolwarm',annot=True,cbar=True,linecolor='k',linewidths=0.9)
        plt.title("Heatmap Post Outliers Removal",fontdict={'size':12,'color':'blue','style':'oblique','family':'calibri'})
        plt.xticks(size=12,rotation=90,style='oblique',color='coral')
    plt.show()

def train_test_datasets(df,train_size=0.80,test_size=0.20,random_state=0):
    """
    This function is perfroming the dataset split into Train/CV/Test/Unseen datasets.
    
    Parameters
    ----------
    df : DataFrame
        DESCRIPTION.
    train_size : float, optional
        Training or Cross-Validation dataset size. The default is 0.80.
        Range --> [0.0-1.0] 
    test_size : float, optional
        Test or first model evaluation or unseen dataset size. The default is 0.20. 
        Range --> [0.0-1.0]
    random_state : int, optional
        Random state for data reproducibility. The default is 0.

    Returns
    -------
    set1_df : DataFrame 
        Training or Cross-Validation dataset.
    set2_df : DataFrame
        Test or first model evaluation or unseen dataset.
    """
    df_X = df.iloc[:,0:-1]
    df_y = df.iloc[:,-1]
    sss = StratifiedShuffleSplit(n_splits=1,train_size=train_size,test_size=test_size,random_state=random_state)
    set1_idx = []
    set2_idx = []
    for set1 , set2 in sss.split(df_X,df_y):
        set1_idx.append(set1)
        set2_idx.append(set2)
    
    set1_idx = np.array(set1_idx).flatten()
    set2_idx = np.array(set2_idx).flatten()
    
    set1_df = df.iloc[set1_idx].reset_index(drop=True)
    set2_df = df.iloc[set2_idx].reset_index(drop=True)
    return set1_df , set2_df

def minority_upsampling(df,minority_label,upsamples=None,replace=True,random_state=0):
    """
    This function is created for perfrming the upsampling of minority class records.

    Parameters
    ----------
    df : DataFrame
        Dataframe with all the records.
    minority_label : TYPE
        Minority class label in the dataframe.
    upsamples : int, optional
        Number of new records to be created in over sampling
    replace : Bool, optional
        Replace the records during upsampling. The default is True.
    random_state : TYPE, optional
        Random state for reproducibility. The default is 0.

    Returns
    -------
    df_upsampled : DataFrame
        Upsampled minority class clubbed with majority class records.

    """
 
    df_minority = df[df.iloc[:,-1] == minority_label]
    df_majority = df[df.iloc[:,-1] != minority_label]
    
    if upsamples == None:
        upsamples = df_minority.shape[0]*2
    else: 
        upsamples = upsamples
        
    df_minority_upsampled = resample(df_minority,replace=replace,n_samples=upsamples,random_state=random_state)
    df_minority_upsampled.reset_index(drop=True,inplace=True)
    df_majority.reset_index(drop=True,inplace=True)
    df_upsampled = pd.concat([df_majority,df_minority_upsampled],axis=0)
    df_upsampled.reset_index(drop=True,inplace=True)
    return df_upsampled