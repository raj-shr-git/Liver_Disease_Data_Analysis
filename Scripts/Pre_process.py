# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:22:06 2020

@author: Rajesh Sharma
"""
import numpy as np
import pandas as pd
import os
import pickle

def data_pre_process(df_obj1,df_obj2,model_file_name="Liver_Patient_Model.pkl"):
    
    #### Merging Feature('Age') in dataframes ####
    df_obj = pd.concat([pd.DataFrame(df_obj1['Age'],columns=['Age']),df_obj2],axis=1)

    #### Creating feature AST/ALT ####
    df_obj['AST_ALT_Ratio'] = df_obj['AST']/df_obj['ALT']
    
    #### Applied Feature Transformation :: Method-I ####")
    trained_data_limits1 = {'Total Bilirubin':[1. , 4.55, -1.45],
                            'ALP':[204. , 465.5,  -2.5],
                            'ALT':[33., 112., -32.],
                            'AST':[40. , 171.5, -64.5],
                            'AST_ALT_Ratio':[1.16,  2.88, -0.38],
                            'Protiens':[6.65, 9.49, 3.59],
                            'Albumin Globulin Ratio':[1. , 1.6 , 0.26]}

    for col in df_obj.columns[1:]:
        median, upper_limit, lower_limit = trained_data_limits1[col]
        df_obj[col] = df_obj[col].apply(lambda val: np.log1p(upper_limit) if val > upper_limit else np.sqrt(np.square(lower_limit)) if val < lower_limit else val)                

    #### Applied Feature Transformation :: Method-II ####")
    trained_data_limits2 = {'Total Bilirubin':[1. , 3.08, -0.57],
                            'ALP':[191., 362.,  34.],
                            'ALT':[28. ,  85.5, -22.5],
                            'AST':[32. , 112.5, -35.5],
                            'AST_ALT_Ratio':[1.16,  2.57, -0.2],
                            'Protiens':[6.6, 9.3, 3.7],
                            'Albumin Globulin Ratio':[1. , 1.6 , 0.26]}

    for col in df_obj.columns[1:]:
        median2, upper_limit2, lower_limit2 = trained_data_limits2[col]
        df_obj[col] = df_obj[col].apply(lambda val: np.log1p(upper_limit2) if val > upper_limit2 else np.sqrt(np.square(lower_limit2)) if val < lower_limit2 else val)

    #### Applying Feature Normalization ####")
    trained_data_norms = {'Age':1201.57,
                          'Total Bilirubin':34.19,
                          'ALP':4790.73,
                          'ALT':842.96,
                          'AST':923.22,
                          'AST_ALT_Ratio':31.94,
                          'Protiens':168.06,
                          'Albumin Globulin Ratio':24.88}

    for col in df_obj.columns:
        df_obj[col] = df_obj[col].apply(lambda val: val/trained_data_norms[col])
        
    #### Merging Feature('Gender') with Scaled and Normalized dataframe ####
    df_obj = pd.concat([df_obj,pd.DataFrame(df_obj1['Gender'],columns=['Gender'])],axis=1)  

    #### Encoding('Gender') variable ####
    gender = {'Female':0,'Male':1}
    df_obj['Gender'] = df_obj['Gender'].apply(lambda val: gender[val])

    # Load the Model from pickle file
    with open(os.path.join(model_file_name), 'rb') as file:  
        Pickled_RF_Model = pickle.load(file)
    
    df_obj = df_obj[['Age','Total Bilirubin','ALP','ALT','AST','AST_ALT_Ratio','Protiens',
                     'Albumin Globulin Ratio','Gender']]
    
    y_pred = pd.DataFrame(Pickled_RF_Model.predict(df_obj),columns=['Result'],index=[''])
    y_pred_prob = pd.DataFrame(Pickled_RF_Model.predict_proba(df_obj),
                               columns=['No Liver Ailment',
                                        'Liver Disease Present'],
                               index=[''])
    y_pred_prob = y_pred_prob.applymap(lambda val : str(np.round((val*100),2))+'%')
    return y_pred, y_pred_prob        
        
        