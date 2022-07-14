# core libraries
from pyexpat import features
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
import base64



# css styling
st.set_page_config(layout="wide")


# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 1rem;
                    padding-bottom: 5rem;
                }
               .css-wjbhl0 {
                    padding-top: 3rem;
                    padding-bottom: 1rem;
                }
                .css-zx8yrj {
                    border: double;
                    border-color: black;
                    background-color: black;
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
    background-color: teal;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #0066FF;
    color:#fffff;
    }
</style>""", unsafe_allow_html=True)


# Loading and caching Our dataset
@st.cache(suppress_st_warning=True, allow_output_mutation=True, persist= True)
def load_files():
    dataset= pd.read_csv(r"C:\Users\godwi\Data_Science_ML\Machine-Learning-Web-Apps\stroke_prediction_webapp\stroke_train.csv")
    cf = pd.read_csv(r"C:\Users\godwi\Data_Science_ML\Machine-Learning-Web-Apps\stroke_prediction_webapp\confusion_matrix_dataframe.csv")
    cmax= cf.rename({"no":0, "yes" : 1}, axis =1 )
    model = pickle.load(open(r"C:\Users\godwi\Data_Science_ML\Machine-Learning-Web-Apps\stroke_prediction_webapp\model_stroke.pkl","rb"))
    return dataset, cmax, model
dataset, cmax,model = load_files()

# unpack 
master_df, conf_max_df, stroke_model = dataset, cmax, model




def main():
    # Title header "
    st.title(" 𝐒𝐭𝐫𝐨𝐤𝐞 𝐃𝐢𝐬𝐞𝐚𝐬𝐞 𝐏𝐫𝐞𝐝𝐢𝐜𝐭𝐢𝐨𝐧 - 𝐌𝐚𝐜𝐡𝐢𝐧𝐞 𝐋𝐞𝐚𝐫𝐧𝐢𝐧𝐠") 
  
 
    st.info(" ##### by Godwin Nwalozie")
    st.markdown("")
     
    #with st.expander(" Expand to view detaild About this Model"):
    
    
        # Title message about stroke
    
    st.markdown(""" This machine learning model predicts the likelihood of a stroke disease based on historical data from patients.
    This is very critical in the early detection of causative factors, the treatment and avoidance of fatal conditions.""")
    
    

   # Model statistics
    with st.container():
        row_count = len(master_df ) 
        col_count_ini = len(master_df .columns)
        col_count = len(master_df .columns)-3
        st.markdown("<h4 style='text-align: left; color: brown;'> Details of the trained dataset 📊 </h4>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Number of rows", row_count, "")
        col2.metric("Estimator","RFClassifier", "")
        col3.metric("Prediction Accuracy", "92%", "")


    # collect user input
    st.sidebar.markdown("<h2 style='text-align:left; color: brown;'> Select Features </h2>",unsafe_allow_html=True)

    gender = st.sidebar.selectbox('Whats your gender ?',("","Female","Male"))
    age = st.sidebar.number_input('Input Age (Minimun 18 years) ', key = 'int',max_value  =100,min_value = 18)
    hypertension = st.sidebar.selectbox('Are you hpertensive? ',{"","Yes", "No"})
    heart_disease = st.sidebar.selectbox('Any heart related disease ? ',("","Yes", "No"))
    ever_married= st.sidebar.selectbox('Ever married ?', ("","Yes" ,"No"))
    work_type = st.sidebar.selectbox('Work type ?', ("","Private","Self-employed","children","Govt_job","Never_worked"))
    avg_glucose_level= st.sidebar.number_input('Average Gloucose Level', min_value= 0.0 , max_value=350.0)
    bmi = st.sidebar.number_input('Enter your current BMI', min_value= 0.0, max_value= 100.0)
    smoking_status = st.sidebar.selectbox('Smoking status',("","never smoked" , "formerly smoked", "smokes"))
    st.sidebar.markdown("") 


    # input dictionary to build datafarame  
    input = {"gender":gender,"age":age,"hypertension": hypertension, "heart_disease":heart_disease,\
            "ever_married":ever_married,"work_type":work_type,
            "avg_glucose_level" : avg_glucose_level, "bmi" : bmi,"smoking_status": smoking_status }  
    

    st.markdown("<h4 style='text-align: left; color:brown;'> Your selected features (important you enter all values) 📝 </h4>", unsafe_allow_html=True)

   
    
    frame =  pd.DataFrame(input, index = [0])
    st.write(frame.style.format({'age': '{:.0f}', 'avg_glucose_level': '{:.2f}', 'bmi': '{:.2f}'}))
    
    
    with st.container():
        
        
        
        if st.button('click here to predict love 👈'):
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
            * You observe from the charts that stroke is predominant amongst age category 45 years and above
            * More females with stroke """)


        # chart for confusion metrix   
        sns.set_theme(font_scale=0.84)
        col1, col2 = st.columns(2)
        with col1:
            
            @st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
            def conf_max ():
                fig,ax = plt.subplots(figsize = (9.5,4))
                sns.heatmap(conf_max_df ,xticklabels = True, annot =True, ax = ax,linewidths=0.2, \
                linecolor='grey',fmt = "2d",annot_kws={'size': 10})
                ax.set_title ("Performance of the trained model - Confusion Matrix (Truth Table)")
                ax.set_xlabel("Predicted Label",fontsize = 10)
                ax.set_ylabel("Actual Label",fontsize = 10)
                ax.tick_params(labelsize=10)
                return fig              
            conf_max = conf_max()   
            st.write(conf_max)             
                            
            age_stroke = master_df.loc[:,["gender","stroke"]].groupby("gender").count()
            fig,(ax) = plt.subplots(figsize = (7,3))
            sns.barplot(data =age_stroke, y = "stroke", x = age_stroke.index, ax= ax)
            ax.set(title ="Correlation between glucose level and bmi")
            ax.set_title ("Marrital Status to having stroke")
            st.write(fig)
           
        
            fig,ax = plt.subplots(figsize = (6,4))
            sns.regplot(data = master_df, x= "bmi", y= "avg_glucose_level", marker = "*")
            ax.set(title ="Correlation between glucose level and bmi")
            st.write(fig)
            
            
        


            # chart for heart diseaase
            disease_check = pd.crosstab(master_df.gender, master_df.heart_disease).rename({0: "No", 1:"Yes"}, axis = 1)
            fig, ax =plt.subplots(figsize = (9,4))
            #disease_check.plot( kind = 'bar', color = ('teal',"blueviolet"), ax=ax)
            sns.heatmap(data = disease_check, annot = True, fmt ="2d", cmap="Reds",linewidths=0.4, linecolor='grey' )
            ax.set(title ="Heart disease by gender")
            st.write(fig)     
                
            
     

    with col2:
        
        #bmi age correlation         
        st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
        def age_max ():
            fig,ax = plt.subplots(figsize = (6,4))
            sns.scatterplot(data = master_df, x= "bmi", y= "age", hue  = 'stroke')
            ax.set(title ="Correlation between age and bmi to a stroke")
            return fig
        age_bmi = age_max()
        st.write(age_bmi)
        
        
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
            fig,ax = plt.subplots(figsize = (6,3))
            pivot_age.plot(kind = 'bar', ax = ax, fontsize = 8, width=0.4)
            ax.set_title("Stroke disease by age category (Stroke is observed from 40+ years)", fontsize = 8)
            plt.legend(fontsize = 7, loc = "upper left")
            return fig
        age_category = age_category()
        st.write(age_category)
        
        
        
        
            
        #Feature correlation
        st.cache(hash_funcs={matplotlib.figure.Figure: lambda _: None})
        def corr_features ():
            fig,ax =plt.subplots(figsize = (10,5))
            feature_check =sns.heatmap(master_df.drop(['id'],axis =1).corr(), cmap="Blues", annot = True, linewidths=0.3,\
            linecolor='grey', ax = ax, annot_kws={'size': 12})
            ax.set_title ("Feature Correlation")
            return fig
        features = corr_features()
        st.write(features)

            
    st.markdown("***")
    with st.container():

        st.text("""𝑾𝒊𝒕𝒉𝒐𝒖𝒕 𝒅𝒂𝒕𝒂 𝒚𝒐𝒖’𝒓𝒆 𝒋𝒖𝒔𝒕 𝒂𝒏𝒐𝒕𝒉𝒆𝒓 𝒑𝒆𝒓𝒔𝒐𝒏 𝒘𝒊𝒕𝒉 𝒂𝒏 𝒐𝒑𝒊𝒏𝒊𝒐𝒏.” 𝑬𝒅𝒘𝒂𝒓𝒅𝒔 𝑫𝒆𝒎𝒊𝒏𝒈, 𝑺𝒕𝒂𝒕𝒊𝒔𝒕𝒊𝒄𝒊𝒂𝒏 """)    

        kaggle=' 🔍Find me on Linkedin [link](https://www.linkedin.com/in/godwinnwalozie/)'
        st.markdown(kaggle,unsafe_allow_html=True)
        git=' 🔍 Find me on Git [link](https://github.com/godwinnwalozie)'
        st.markdown(git,unsafe_allow_html=True)
        kaggle=' 🔍Find me on Kaggle [link](https://www.kaggle.com/godwinnwalozie/code)'
        st.markdown(kaggle,unsafe_allow_html=True)

if __name__ == '__main__':
    main()