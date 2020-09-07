# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 16:40:17 2020

@author: rajsh
"""
import streamlit as st
import os
import numpy as np

st.title("Welcome to Liver Health Analyzer App")

st.image(os.path.join("App_Images/happy-liver.jpeg"),width=600)

st.info("I'm liver doctor and will help you to assess your liver health.\
        You only need to provide me your liver enzymes and protiens levels.")
    
st.subheader("Why are you here?")

status = st.selectbox('',("Just wanted to see liver doctor",
                              "For consultancy and I know my Liver Enzymes level",
                              "For consultancy but I don't know my Liver Enzymes level"))

if status == "For consultancy and I know my Liver Enzymes level":
    first_name = st.text_input("Enter your first name")
    last_name = st.text_input("Enter your last name")
    
    gender = st.selectbox('',('Male','Gender','Other'))
    
    age = st.slider("How old are you?",min_value=4,max_value=100,step=10,value=20)
    
    total_bilirubin = st.slider("Enter your Total Bilirubin level",
                                min_value=0.0,max_value=500.0,step=0.1,value=0.1)
    
    alp = st.slider("Enter your Alkaline Phosphotase(ALP) level",
                    min_value=0.0,max_value=1000.0,step=0.1,value=0.1)
    
    alt = st.slider("Enter your Alamine Aminotransferase(ALT) level"
                    ,min_value=0.0,max_value=1000.0,step=0.1,value=0.1)
    
    ast = st.slider("Enter your Aspartate Aminotransferase(AST) level",
                    min_value=0.0,max_value=1000.0,step=0.1,value=0.1)
    
    tot_protiens = st.slider("Enter your Total Protiens level",
                    min_value=0.0,max_value=500.0,step=0.1,value=0.1)
    
    a_g_ratio = st.slider("Enter your Albumin & Globulin Ratio level",
                    min_value=0.0,max_value=500.0,step=0.1,value=0.1)   

    if st.button("Submit"):
        st.info("Your data has been accepted and results will be mailed to you..")
elif status == "For consultancy but I don't know my Liver Enzymes level":
    st.warning("You can consult us once you have your reports!!")
    st.stop()