# core libraries
from pyexpat import features
from turtle import width
import joblib
import plotly
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pickle
from PIL import Image
import os
import time
import plotly.express as px




# css styling
st.set_page_config(layout="wide")


# Remove whitespace from the top of the page and sidebar
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
    background-color:dimgray;
    color:white;
}
div.stButton > button:hover {
    background-color: white;
    color:blue;
    }
</style>""", unsafe_allow_html=True)


# get dir name os path
dir_name = os.path.abspath(os.path.dirname(__file__))


# path = os.path.dirname(__file__)
# my_file = path+'\stroke_train.csv'
# st.write(pd.read_csv(my_file))

# Loading and caching Our dataset



file = Image.open(os.path.join(dir_name,"stroke_app_title_image-bg.png"))
st.image(file )
st.caption("### by Godwin Nwalozie")



@st.cache(suppress_st_warning=True, allow_output_mutation=True, persist= True)
def load_files():
    dataset = pd.read_csv(os.path.join(dir_name, 'stroke_train.csv'))
    cf = pd.read_csv(os.path.join(dir_name, 'confusion_matrix_dataframe.csv'))
    model = joblib.load(os.path.join(dir_name,"model_stroke.joblib"))
    return dataset, cf, model
dataset, cfmax, model = load_files()

# unpack 
master_df, conf_max_df, stroke_model = dataset, cfmax, model



        
#with st.expander(" Expand to view detaild About this Model"):


# Title message about stroke

st.info(""" ##### This machine learning model predicts the likelihood of a stroke disease based on historical data from patients, which is very critical in the early detection of causative factors, for treatment and prevention of fatal conditions.""")



# Model statistics
with st.container():
    row_count = len(master_df ) 
    col_count_ini = len(master_df .columns)
    col_count = len(master_df .columns)-3
    st.markdown("<h4 style='text-align: left; color: brown;'> Model estimator and score 📊 </h4>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Number of rows", row_count, "")
    col2.metric("Estimator","RandomForest", "")
    col3.metric("Prediction Accuracy", "92%", "")


# collect user input
st.sidebar.title('Select Features')

gender = st.sidebar.selectbox('Whats your gender ?',("","Female","Male"))
age = st.sidebar.number_input('Input Age (Minimun 18 years) ', key = 'int',max_value  =100,min_value = 18)
hypertension = st.sidebar.selectbox('Are you hpertensive? ',{"","Yes", "No"})
heart_disease = st.sidebar.selectbox('Any heart related disease ? ',("","Yes", "No"))
ever_married= st.sidebar.selectbox('Ever married ?', ("","Yes" ,"No"))
work_type = st.sidebar.selectbox('Work type ?', ("","Private","Self-employed","children","Govt_job","Never_worked"))
avg_glucose_level= st.sidebar.number_input('Avg gloucose level in blood stream(mg/dL)', min_value= 0.0 , max_value=350.0)
bmi = st.sidebar.number_input('Enter your current BMI', min_value= 0.0, max_value= 100.0)
smoking_status = st.sidebar.selectbox('Smoking status',("","never smoked" , "formerly smoked", "smokes"))
st.sidebar.markdown("") 


# input dictionary to build datafarame  
input = {"gender":gender,"age":age,"hypertension": hypertension, "heart_disease":heart_disease,\
    "ever_married":ever_married,"work_type":work_type,
    "avg_glucose_level" : avg_glucose_level, "bmi" : bmi,"smoking_status": smoking_status }  


st.markdown("<h4 style='text-align: left; color:brown;'> Features selected 📝 </h4>", unsafe_allow_html=True)



frame =  pd.DataFrame(input, index = [0])
st.table(frame.style.format({'age': '{:.0f}', 'avg_glucose_level': '{:.2f}', 'bmi': '{:.2f}'}))


with st.container():



    if st.button('👇 click to make prediction'):
        if gender == "" or hypertension =="" or heart_disease == "" or ever_married == "" or work_type == "" or smoking_status == "":
            st.error(" ##### ⚠️ Mazi says you still have some missing input")
        else:
                        
            
            prediction = stroke_model.predict(frame)[0]
            probability = stroke_model.predict_proba(frame).squeeze()
            probability_low = round(probability[0] *100)
            probability_high = round(probability[1] *100)
            if prediction  == 0:
                prediction =  "Low Risk 😊" 
            else:
                prediction = "High Risk 😞"
                with st.spinner('Processing.......'):
                    time.sleep(2)
                            
            st.header(f"**{prediction}**")
            st.markdown(f" ##### Low chance @ {probability_low}% : High chance @ {probability_high}% ")
    else:   
        st.write("Prediction will be displayed here")




with st.container():
# st.markdown("***")
    st.markdown(""" ##### Remember prediction is based on the patterns found on the dataset (See charts below) """)


    st.markdown("***")
    st.subheader("Exploratory Data Analysis")
    st.markdown (f""" 
        * Stroke disease is predominant amongst age category 45 years and above
        * High bmi might lead to stroke especially age 45+
        * More females with stroke (dependends on the data and distribution)
        * Positive coreelation between heart disease and hypertension""")

    plt.style.use('seaborn-ticks')
    # chart for confusion metrix   
    
    

col1, col2 = st.columns(2)
with col1:
    
    @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
    def conf_max ():
        cf =conf_max_df.rename({0:"No",1: "Yes"}, axis = 1)
        cf2 = cf.rename({0:"No",1: "Yes"}, axis = 0)  
        fig = px.imshow(cf2, text_auto= True, title= "Confusion matrix",aspect="auto", width= 520 )
        return fig
    show = conf_max ()
    st.write(show)
    
    st.cache()
    def gender_stroke ():                
        gender_stroke = master_df.loc[:,["gender","stroke"]].groupby("gender").count()
        fig= px.bar(gender_stroke, width=530, title =" stroke disease by gender")
        return fig
    gen_stroke = gender_stroke()
    st.write(gen_stroke)      

    st.cache()
    def bmi_glu ():
        fig =px.scatter(master_df, x= "bmi", y= "avg_glucose_level", color = 'stroke', title = "correlation between avg glucose level and bmi")
        fig.update_layout(autosize=False,width=540, height=500 )
        return fig
    bmi_glu = bmi_glu()
    st.write(bmi_glu)
    
    

with col2:

    #bmi age correlation         
    st.cache()
    def bmi_age ():
        fig = px.scatter( master_df, x= "bmi", y= "age", color = 'stroke')
        fig.update_layout(width=550,height =450, title = "correlation between bmi and stroke" )
        return fig
    bmi_age = bmi_age()
    st.write(bmi_age)
    



        # Age category Plot
    st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
    def age_category():
        age_hyper = master_df.loc[:,["age","heart_disease"]]
        age_hyper.heart_disease = age_hyper.heart_disease.apply(lambda x: "Yes" if x == 1 else "No" )
        age_hyper['age_cat'] = age_hyper.age.apply(lambda x :  "0-2" if 0 <= x<2 else
                                            "2-5" if 2<= x<= 5 else
                                            "6-13" if 5< x< 13 else
                                            "13-18" if 13<= x< 18 else
                                            "18-30" if 18<= x< 30 else
                                            "30-40" if 30<= x< 40 else
                                            "40-50" if 40<= x< 50 else
                                            "50-65" if 50<= x< 65 else
                                            "65+" if x>= 65 else "not known")
        pivot_age = age_hyper.pivot_table(index = 'age_cat', columns='heart_disease', values="age", aggfunc= 'count')
        fig = px. bar(pivot_age, barmode='group', title = "age distribution with ehart disease " )
        fig.update_layout(width = 600)
        return fig
    age_cat = age_category()
    st.write(age_cat)

    
    # chart for heart diseaase
    st.cache()
    def heart_gender ():
        disease_check = pd.crosstab(master_df.gender, master_df.heart_disease).rename({0: "No", 1:"Yes"}, axis = 1)
        fig = px.imshow(disease_check,text_auto= True)
        fig.update_layout(width = 550, title = " heart disease by gender")
        return fig
    heart_gender = heart_gender()
    st.write(heart_gender)
    






    
st.markdown("***")
with st.container():

    st.text("""𝑾𝒊𝒕𝒉𝒐𝒖𝒕 𝒅𝒂𝒕𝒂 𝒚𝒐𝒖’𝒓𝒆 𝒋𝒖𝒔𝒕 𝒂𝒏𝒐𝒕𝒉𝒆𝒓 𝒑𝒆𝒓𝒔𝒐𝒏 𝒘𝒊𝒕𝒉 𝒂𝒏 𝒐𝒑𝒊𝒏𝒊𝒐𝒏.” 𝑬𝒅𝒘𝒂𝒓𝒅𝒔 𝑫𝒆𝒎𝒊𝒏𝒈, 𝑺𝒕𝒂𝒕𝒊𝒔𝒕𝒊𝒄𝒊𝒂𝒏 """)    

    kaggle=' 🔍Find me on Linkedin [link](https://www.linkedin.com/in/godwinnwalozie/)'
    st.markdown(kaggle,unsafe_allow_html=True)
    git=' 🔍 Find me on Git [link](https://github.com/godwinnwalozie)'
    st.markdown(git,unsafe_allow_html=True)
    kaggle=' 🔍Find me on Kaggle [link](https://www.kaggle.com/godwinnwalozie/code)'
    st.markdown(kaggle,unsafe_allow_html=True)