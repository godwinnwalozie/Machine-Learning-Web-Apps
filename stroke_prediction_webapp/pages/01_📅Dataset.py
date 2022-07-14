import streamlit as st
import pandas as pd
import numpy as np
import os
import random

st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 3rem;
                    padding-bottom: 5rem;
                }
               .css-hxt7ib {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                }
                .css-zx8yrj {
                    border: double;
                    border-color: black;
                    background-color: white;
                }
                .css-1frylpx {
                color: #ffffff;
                }
                .css-15490cf{
                   color: #ffffff; 
                }
                    
        </style>
        """, unsafe_allow_html=True)

# button styling
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: white;
    color:blue;
}
div.stButton > button:hover {
    background-color: #white;
    color:#fffff;
    }
</style>""", unsafe_allow_html=True)



dir_name = os.path.abspath(os.path.dirname(__file__))

@st.cache(suppress_st_warning=True, allow_output_mutation=True, persist= True)
def dataset_load ():
    dataset = pd.read_csv(os.path.join(dir_name, 'stroke_train.csv'))
    return dataset
data = dataset_load()




with st.container():
    row_count = len(data ) 
    col_count = len(data.columns)
    col_count = len(data.columns)
    
    
    st.title("Details of trained dataset")
    st.caption(" ### Data Source : https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset")

    col1, col2, col3 = st.columns(3)
    col1.metric("Number of rows", row_count, "")
    col2.metric("Number of columns",col_count, "")


if st.button('👇 click to randomize 5 rows '): 
    random.random()
            
st.header("Dataset")
st.write(data.sample(5))