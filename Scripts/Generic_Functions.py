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
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

## Data Sampling Package
from sklearn.utils import resample

def inspect_feature(df_obj,feature_name,scaler):
    """
    This function is created for plotting the histograms and box-plots.

    Parameters
    ----------
    df_obj : DataFrame
        Containing feature that needs to be inspected.
    feature_name : str
        Feature that you want to inspect
    scaler : str, optional
        DESCRIPTION. The default is 'ss'.

    Returns
    -------
    None.
    """
    if scaler == 'ss':
        scaler_obj = StandardScaler()
    elif scaler =='mm':
        scaler_obj = MinMaxScaler()
    elif scaler =='rb':
        scaler_obj = RobustScaler()
    
    with plt.style.context('seaborn'):
        df_obj[feature_name].plot(kind='hist')
        plt.title('Raw Data')
        plt.show()
        df_obj[feature_name].plot(kind='box')
        plt.title('Raw Data')
        plt.show()
        np.log1p(df_obj[feature_name]).plot(kind='hist')
        plt.title('Log1p Data')
        plt.show()
        np.log1p(df_obj[feature_name]).plot(kind='box')
        plt.title('Log1p Data')
        plt.show()  
        pd.DataFrame(scaler_obj.fit_transform(pd.DataFrame(np.log1p(df_obj[feature_name])))).plot(kind='hist')
        plt.title('Scaled Log1p Data')
        plt.show()
        pd.DataFrame(scaler_obj.fit_transform(pd.DataFrame(np.log1p(df_obj[feature_name])))).plot(kind='box')
        plt.title('Scaled Log1p Data')
        plt.show()
        np.sqrt(np.log1p(df_obj[feature_name])).plot(kind='hist')
        plt.title('Sqrt Log1p Data')
        plt.show()
        np.sqrt(np.log1p(df_obj[feature_name])).plot(kind='box')
        plt.title('Sqrt Log1p Data')
        plt.show()
        pd.DataFrame(scaler_obj.fit_transform(pd.DataFrame(np.sqrt(np.log1p(df_obj[feature_name]))))).plot(kind='hist')
        plt.title('Scaled Sqrt Log1p Data')
        plt.show()
        pd.DataFrame(scaler_obj.fit_transform(pd.DataFrame(np.sqrt(np.log1p(df_obj[feature_name]))))).plot(kind='box')
        plt.title('Scaled Sqrt Log1p Data')
        plt.show()

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
    
    df_name[col_name] = df_name[col_name].apply(lambda val: upr_limit1 if val > upr_limit1 
                                                else low_limit1 if val < low_limit1 else val)
    # df_name[col_name] = df_name[col_name].apply(lambda val: low_limit1 + (val-upr_limit1) if val > upr_limit1 
    #                                             else upr_limit1 - (low_limit1-val) if val < low_limit1 else val)

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
        upsamples = df_minority.shape[0]+90
    else: 
        upsamples = upsamples
        
    df_minority_upsampled = resample(df_minority,replace=replace,n_samples=upsamples,random_state=random_state)
    df_minority_upsampled.reset_index(drop=True,inplace=True)
    df_majority.reset_index(drop=True,inplace=True)
    df_upsampled = pd.concat([df_majority,df_minority_upsampled],axis=0)
    df_upsampled.reset_index(drop=True,inplace=True)
    return df_upsampled

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,scoring='recall',
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt