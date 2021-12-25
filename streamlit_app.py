# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 18:13:34 2021

@author: 1386317
"""

import joblib
import re
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import ktrain
import pandas as pd
import json


st.write("# Product Categorization")
st.markdown("***")




message_text = st.text_area("Enter Product Description")
st.markdown("***")

if st.button('PREDICT'):
    
    predictor = ktrain.load_predictor('C:/Users/1386317/Documents/noam_fiverr/')
    model = ktrain.get_predictor(predictor.model, predictor.preproc)
    predictions = model.predict(message_text,return_proba=True)
    
    classes=model.get_classes()
    predictions_df=pd.DataFrame(predictions)
    predictions_df.index=classes
    predictions_df.columns=['Confidence_Score']
    predictions_df['Confidence_Score']=predictions_df.Confidence_Score.round(decimals=4)
    predictions_df=predictions_df.sort_values(by='Confidence_Score',ascending=False)
    # pred_dict=predictions_df.to_dict(orient='records')
    # pred_dict1=dict(sorted(pred_dict.items(),key=lambda item: item[1],reverse=True))
    # pred_dict=predictions_df.to_dict(orient='records')[0]
    # pred_dict=dict(sorted(pred_dict.items(),reverse=True))
    st.write(predictions_df)
    

# fc=dict(sorted(pred_dict.items(),reverse=True))

