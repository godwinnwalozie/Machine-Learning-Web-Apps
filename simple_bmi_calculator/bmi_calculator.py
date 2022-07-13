import streamlit as st
from PIL import Image
from datetime import datetime
from pathlib import Path
import pathlib
import os
import time
import random


st.set_page_config(layout="wide")

# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 4rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-wjbhl0 {
                    padding-top: 3rem;
                    padding-right: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)


with st.container():
    st.title("▌│█║▌║▌║ BMI Calculator")
    st.info(" ##### by Godwin Nwalozie")


with st.container():
    col1, col2 = st.columns(2)

    with col1: 
        st.markdown("")
        st.write(" ### Provide your name to view bmi calculator")

        name = st.text_input("enter name").capitalize()
        
        
        if name != "":
            st.button("Click to change messages")
            love  = ["May you blossom 💐🌹🌵🌷", "You are loved 💕💕💕💕", "Na odogwu you be💰💰💰💰" ,"You are blessed 🙌🙌🙌",  \
                "Your stars will continue to shine ✨🤩✡️🌟", "Ensure you eat enough fruits 🍒🍑🍍🍎", "Ensure you stay hydrated 🥂🍹🫖🥤"]
            wish = random.choice(love)
             
            st.success(f' ###### {wish}')
            gender =st.selectbox("Gender", ("Male", "Female"))
                
            
            weight = st.number_input('Input weight in Kg' ,min_value= 1.00)
            height = st.number_input('Input height in Meters', min_value= 1.00)
            
            if weight == 1 and  height == 1:
                st.error("###### Enter parameters to calculate !")
            else:
                def check ():
                    bmi = (weight//(height)**2)
                    if bmi >= 0 and bmi < 18.5:
                        st.success(f" ##### {name},  your bmi is {bmi},  and class is 𝐔𝐧𝐝𝐞𝐫𝐰𝐞𝐢𝐠𝐡𝐭 " )
                    if bmi >= 18.5 and bmi <= 24.9:
                        st.success(f"##### {name} your bmi is {bmi}, and classs is 𝐍𝐨𝐫𝐦𝐚𝐥 🥰" )
                    elif bmi >= 25 and bmi <= 29.9:
                        st.success(f" ##### {name} your bmi is {bmi}, and class is 𝐎𝐯𝐞𝐫𝐰𝐞𝐢𝐠𝐡𝐭 😟" )
                    elif bmi >= 30  and bmi <=39.9:
                        st.success(f" ##### {name} your bmi is {bmi}, and class is 𝐌𝐨𝐫𝐛𝐢𝐝 𝐎𝐛𝐞𝐝𝐢𝐭𝐲 😨" )
                    elif  bmi > 40:
                        st.success(f"##### {name} your bmi is {bmi}, and class is 𝐒𝐮𝐩𝐞𝐫 𝐎𝐛𝐞𝐝𝐢𝐭𝐲 😨 " )
                        return bmi
                # with st.spinner('Processing .... '):
                #         time.sleep(0)
                bmi = (check())



with col2: 
    st.markdown("")  
    st.subheader("The importance of bmi")
    st.write(""" Body mass index is a tool for calculating a person's risk of obesity and underweight.
    The emergence of conditions categorized as civilization diseases, such as diabetes, stroke, atherosclerosis,\
    hypertension, and nutrition disorders, may be linked to the phenomenon of obesity and underweight.

The detection of diabetes is a further crucial aspect of using BMI.
According to research, having a high BMI level has a significant impact on the disease's development, and maintaining a healthy BMI could slow or even stop the disease's progression. """)
             
    image_header = Image.open("/home/mazi/Data_Science_ML/Machine-Learning-Web-Apps/simple_bmi_calculator/diagram.jpg")
    st.image(image_header) 

kaggle=' 🔍Find me on Linkedin [link](https://www.linkedin.com/in/godwinnwalozie/)'
st.markdown(kaggle,unsafe_allow_html=True)
git=' 🔍 Find me on Git [link](https://github.com/godwinnwalozie)'
st.markdown(git,unsafe_allow_html=True)
kaggle=' 🔍Find me on Kaggle [link](https://www.kaggle.com/godwinnwalozie/code)'
st.markdown(kaggle,unsafe_allow_html=True)
