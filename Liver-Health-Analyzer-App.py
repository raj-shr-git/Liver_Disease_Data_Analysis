# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:40:17 2020

@author: Rajesh Sharma
"""
import streamlit as st
import os
import pandas as pd
from Pre_process import data_pre_process

def sidebar():
    #### Sidebar ####
    st.sidebar.title("Liver Health Analyzer App")
    st.sidebar.image(os.path.join("App_Img_Vid/Liver_img.jpg"),width=110)
    st.sidebar.subheader("User Input Parameters")

def user_inputs():
    #### Entering user details ####
    first_name = st.sidebar.text_input("First name")
    last_name = st.sidebar.text_input("Last name")
    
    gender = st.sidebar.selectbox('Gender',('Male','Female'))
    
    age = st.sidebar.slider("How many years old are you?",min_value=4,max_value=100,step=1,value=25)
    
    total_bilirubin = st.sidebar.slider("Total Bilirubin level (mg/dL)",
                                min_value=0.0,max_value=200.0,step=0.1,value=0.5)
    
    alp = st.sidebar.slider("Alkaline Phosphotase level (iU/L)",
                    min_value=0.0,max_value=1200.0,step=0.1,value=65.0)
    
    alt = st.sidebar.slider("Alanine Aminotransferase level (iU/L)"
                    ,min_value=0.0,max_value=1200.0,step=0.1,value=10.0)
    
    ast = st.sidebar.slider("Aspartate Aminotransferase level (iU/L)",
                    min_value=0.0,max_value=1200.0,step=0.1,value=10.0)
    
    total_protiens = st.sidebar.slider("Total Protiens level (iU/L)",
                    min_value=0.0,max_value=200.0,step=0.1,value=3.0)
    
    a_g_ratio = st.sidebar.slider("Albumin & Globulin Ratio level",
                    min_value=0.0,max_value=100.0,step=0.1,value=0.5) 
    
    user_data = pd.DataFrame({'First Name':first_name,
                 'Last Name':last_name,
                 'Gender':gender,
                 'Age':age,
                 'Total Bilirubin':total_bilirubin,
                 'ALP':alp,
                 'ALT':alt,
                 'AST':ast,
                 'Protiens':total_protiens,
                 'Albumin Globulin Ratio':a_g_ratio},index=[''])
    
    user_personal_info = pd.DataFrame(user_data[['First Name',
                                                  'Last Name',
                                                  'Gender',
                                                  'Age']],index=[''])
    
    user_inp_features = pd.DataFrame(user_data[['Total Bilirubin',
                                                'ALP',
                                                'ALT',
                                                'AST',
                                                'Protiens',
                                                'Albumin Globulin Ratio']],index=[''])
       
    return user_personal_info , user_inp_features

#$$$$$$$$ Main Page $$$$$$$$
def display_main_page_video(vid_name):
    #### Entry Video ####
    video_file = open(os.path.join("App_Img_Vid/"+vid_name),'rb')
    video_bytes = video_file.read()
    st.video(video_bytes,start_time=0)
    st.markdown("##### **Video courtesy** :: ***National Institute for Health and Research***")

def display_user_info():
    #### User entered details ####
    st.text('\n \n')
    st.subheader("User Info")
    st.text("\n")
    st.markdown("##### **Basic Details**")
    inp_usr_info, inp_user_feat = user_inputs()
    st.write(inp_usr_info)
    st.markdown("##### **Liver Enzymes Level**")
    st.write(inp_user_feat)
    return inp_usr_info, inp_user_feat

def display_class_labels_desc():
    #### Dataset :: Class labels description ####
    st.subheader("Result Description")
    class_labels = pd.DataFrame({1:['Liver Ailment present'],
                                 0:['No presence of Liver Ailment']},index=['']).T
    st.write(class_labels)

def predict_user_result():
    #### Predcition ####
    usr_info, usr_feat = display_user_info()
    y_pred, y_pred_prob = data_pre_process(df_obj1=usr_info, df_obj2=usr_feat)
    st.subheader("Level-I: Screening Outcome")
    st.write(y_pred)
    display_class_labels_desc()
    return y_pred_prob

def predict_user_result_prob():
    #### Prediction Probability ####
    y_pred_prob = predict_user_result()
    st.subheader("How much liver is affected?")
    st.write(y_pred_prob)
    
def footer():
    #### Webpage footer #### 
    st.text("\n \n")
    st.text("\n \n")
    st.text("\n \n")
    st.info("##### **Creator : Rajesh Sharma** \n  ###### ***Indian Liver Patient Dataset : Source - UCI ML Repository : https://archive.ics.uci.edu/ml/datasets.php*** \n ###### ***Video Link : https://www.youtube.com/watch?v=oki1vdxYp0o*** \n ###### ***Image Source : https://www.dreamstime.com/***")
    
if __name__ == '__main__':
    sidebar()
    display_main_page_video('MP_video.mp4')
    predict_user_result_prob()
    footer()