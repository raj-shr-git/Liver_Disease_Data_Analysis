# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:52:41 2020

@author: Rajesh Sharma
"""
## Import Outliers fixing function
import Generic_Functions as Tuky_IQR

## Import the required libraries
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

## Feature Scaling and Normalizing Packages
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, normalize, QuantileTransformer, PowerTransformer
from sklearn.preprocessing import normalize

## Data Sampling Package
from sklearn.utils import resample

## Ignoring the warnings by messages
import warnings
warnings.filterwarnings("ignore")

class feature_standardizing:
    """
    This class is created for performing the scaling and normalization on the model input features.

    Parameters
    ----------
    X : DataFrame
        This is the DataFrame which you want to be scale or normalize.
    
    Feature Transformer : str, optional
        Type of Feature transformation you can apply on the dataset. The default is 'pt'.
            pt: Power Transformer
            qt: Quantile Transformer
            
    scaler : str, optional
        Kind of scaler or normalizer that can be performed on the dataset. The default is 'ss'.
            ss : Standard Scaler
            mm : Min Max Scaler
            rb : Robust Scaler
            nm : Normalizer
            log : Log Transformation
    
    random_state : int, optional
        For reproducilibilty of the results. This is by default 0.
        
    Returns
    -------
    scaled_X : DataFrame
        Transformed or Scaled or normalized dataframe.
    """      
    def __init__(self,X,trans='pt',scaler='ss',random_state=0):
        """
        This function is created for initializing the required variables or objects.
        """
        self.X = X
        self.trans = trans
        self.scaler = scaler
        self.random_state = random_state
        self.label_font_style = {'size':10,'color':'blue','style':'oblique','family':'calibri'}
        self.title_font_style = {'size':12,'color':'blue','style':'oblique','family':'calibri'}

    def feat_trans(self):
        """
        This function is created for performing the transformers on the model input features.
    
        Returns
        -------
        trans_X : DataFrame
            transformed dataframe.
        """
        if self.trans == 'pt':
            trans_obj = PowerTransformer()
        elif self.trans == 'qt':
            trans_obj = QuantileTransformer()
        else:
            trans = None
        
        trans_X = pd.DataFrame(trans_obj.fit_transform(self.X))
        trans_X.columns = self.X.columns
        
        trans_title = {'pt':'Power Transformer',
                       'qt':'Quantile Transformer'}
        
        with plt.style.context('seaborn'):
            sns.boxplot(data=trans_X.iloc[:,:])
            plt.title(trans_title[self.trans],fontdict=self.title_font_style)
            plt.xticks(size=12,rotation=90,style='oblique',color='coral')
        plt.show()
        
        with plt.style.context('seaborn'):
            sns.heatmap(trans_X.corr(),cmap='coolwarm',annot=True,cbar=True,linecolor='k',linewidths=0.9)
            plt.title(str("Heatmap Post")+str(trans_title[self.trans]),fontdict=self.title_font_style)
            plt.xticks(size=12,rotation=90,style='oblique',color='coral')
        plt.show()
        
        return trans_X        

    def scale_norm(self):
        """
        This function is created for performing the scaling and normalization on the model input features.
    
        Returns
        -------
        scaled_X : DataFrame
            Scaled or normalized dataframe.
        """
    
        if self.scaler == 'ss':
            scaler_obj = StandardScaler()
        elif self.scaler =='mm':
            scaler_obj = MinMaxScaler()
        elif self.scaler =='rb':
            scaler_obj = RobustScaler()
        elif self.scaler =='log':
            scaler_obj = np.log
    
        if self.scaler !='log':
            scaled_X = pd.DataFrame(scaler_obj.fit_transform(self.X))
        else: 
            scaled_X = pd.DataFrame(scaler_obj(self.X))
            
        scaled_X.columns = self.X.columns
        
        scaler_title = {'ss':'Standard Scaled',
                        'mm':'Min Max Scaled',
                        'rb':'Robust Scaled',
                        'nm':'Normalized',
                        'log':'Log Transformed'}
        
        with plt.style.context('seaborn'):
            sns.boxplot(data=scaled_X.iloc[:,:])
            plt.title(scaler_title[self.scaler],fontdict=self.title_font_style)
            plt.xticks(size=12,rotation=90,style='oblique',color='coral')
        plt.show()
        
        with plt.style.context('seaborn'):
            sns.heatmap(scaled_X.corr(),cmap='coolwarm',annot=True,cbar=True,linecolor='k',linewidths=0.9)
            plt.title(str("Heatmap Post ")+str(scaler_title[self.scaler]),fontdict=self.title_font_style)
            plt.xticks(size=12,rotation=90,style='oblique',color='coral')
        plt.show()
        
        return scaled_X

if __name__ == "__main__":
    cleaned_df = pd.read_csv(os.path.join('Cleaned_Imp_Features.csv'))
    cols = cleaned_df.columns
 
    cleaned_X = cleaned_df[['Age','Total_Bilirubin','Alkaline_Phosphotase','Total_Protiens','Albumin_and_Globulin_Ratio','AST_ALT_Ratio']]
    cleaned_y = cleaned_df.iloc[:,-1]
       
    scale_feat = feature_standardizing(cleaned_X,trans='pt')
    # cleaned_scaled_X = scale_feat.scale_norm()
    cleaned_trans_X = scale_feat.feat_trans()
    
    cleaned_trans_X = pd.DataFrame(normalize(cleaned_trans_X,norm='l2',axis=1))
    
    # scale_feat2 = feature_standardizing(cleaned_trans_X,scaler='mm')
    # cleaned_trans_scaled_X = scale_feat2.scale_norm()
    
    for col in cleaned_trans_X.columns:
        Tuky_IQR.fix_outliers(df_name = cleaned_trans_X,col_name=col,whis_width=1.5)
    Tuky_IQR.plot_data(cleaned_trans_X)
    
    cleaned_trans_scaled = pd.concat([cleaned_trans_X,cleaned_df[['Gender_0','Gender_1','Label']]],axis=1)
    
    cleaned_trans_scaled.to_csv(os.path.join('Cleaned_Trans_Scaled_Features.csv'),index=False)
