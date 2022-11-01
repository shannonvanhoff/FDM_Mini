# Core Pkgs
from unicodedata import category
from pandas.core.arrays import categorical
from sqlalchemy import true
import streamlit as st
# EDA Pkgs
import pandas as pd
import numpy as np

# Utils
import os
import joblib

# Data Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib

import cv2
import base64
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.model_selection import train_test_split
import pickle

#df=pd.read_csv(r'C:\Users\ashvi\Desktop\FDM\cleaned_dataset.csv')
#loading the saved model
loaded_model = pickle.load(open('models/trained_model.sav','rb'))



col1, col2= st.columns([2,1])

st.markdown(f'''
    <style>
    section[data-testid="stSidebar"] .css-ng1t4o {{width: 200px;}}
    </style>
''',unsafe_allow_html=True)

#with open('style.css') as f:
 #  st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#Adding background image/"""
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url(data:image/{"avif"};base64,{encoded_string.decode()});
#         background-size: cover
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
#     )
# add_bg_from_local('hos2.avif')  
# x = df.drop(['status'],axis=1)
# y = df.iloc[:,-1]

# y=df["status"]
# x= df.loc[:, df.columns != 'status']
# X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=42)

def display():
    col1.title('Startup Success Prediction Application')
    


 

# def load_PreTrainedModelDetails():
#     PreTrainedModelDetails = joblib.load("classification/models/leo_ml_classification.joblib")
#     return PreTrainedModelDetails
#
#
# def prediction(input_df):_df

#     PreTrainedModelDetails = load_PreTrainedModelDetails()
#
#     # Random Forest Classifier
#     LogisticRegressionClassifier = PreTrainedModelDetails.get('model')
#
#     # PreFitted Encoder
#     PreFittedEncoder = PreTrainedModelDetails.get('encoder')
#
#     # PreFitted Scaler
#     PreFittedScaler = PreTrainedModelDetails.get('scaler')
#
#     numeric_cols = PreTrainedModelDetails.get('numeric_cols')
#
#     categorical_cols = PreTrainedModelDetails.get('categorical_cols')
#
#     encoded_cols = PreTrainedModelDetails.get('encoded_cols')
#
#     train_score = PreTrainedModelDetails.get('train_score')
#
#     val_score = PreTrainedModelDetails.get('val_score')
#
#     input_df[encoded_cols] = PreFittedEncoder.transform(input_df[categorical_cols])
#     input_df[numeric_cols] = PreFittedScaler.transform(input_df[numeric_cols])
#
#     inputs_for_prediction = input_df[numeric_cols + encoded_cols]
#
#     prediction = RandomForestClassifier.predict(inputs_for_prediction)
#
#     accuracy = ((train_score + val_score) / 2) * 100
#
#     if prediction == 0:
#         st.success("Lending the loan for the customer is not risky")
#     else:
#         st.warning("Lending the loan for the customer is risky")
#
#     st.write("Accuracy of the prediction : {}".format(accuracy))

# def get_user_input(user_report):
    
    
#     relationships = st.number_input("Relationships", 1, 120)
    
#     funding_rounds = st.number_input("Funding Rounds", 1, 120)
#     funding_total_usd = st.number_input("Total Funding")
#     milestones = st.number_input("milestones", 1, 120)
#     category_code = st.selectbox("category", ('music', 'enterprise', 'web', 'software', 'games_video',
#        'network_hosting', 'finance', 'mobile', 'education',
#        'public_relations', 'security', 'other', 'photo_video', 'hardware',
#        'ecommerce', 'advertising', 'travel', 'fashion', 'analytics',
#        'consulting', 'cleantech', 'search', 'semiconductor', 'social',
#        'biotech', 'medical', 'automotive', 'messaging', 'manufacturing',
#        'hospitality', 'news', 'transportation', 'sports', 'real_estate',
#        'health'))
    
#     category = {'music':19, 'enterprise':8, 'web':34, 'software':30, 'games_video':11,
#        'network_hosting':20, 'finance':10, 'mobile':18, 'education':7,
#        'public_relations':24, 'security':27, 'other':22, 'photo_video':23, 'hardware':12,
#        'ecommerce':6, 'advertising':0, 'travel':33, 'fashion':9, 'analytics':1,
#        'consulting':5, 'cleantech':4, 'search':26, 'semiconductor':28, 'social':29,
#        'biotech':3, 'medical':16, 'automotive':2, 'messaging':17, 'manufacturing':15,
#        'hospitality':14, 'news':21, 'transportation':32, 'sports':31, 'real_estate':25,
#        'health':13}
#     category1=category[category_code]
#     state_code=st.selectbox("state", ('CA', 'MA', 'KY', 'NY', 'CO', 'VA', 'TX', 'WA', 'IL', 'PA', 'GA',
#        'NH', 'MO', 'FL', 'NJ', 'WV', 'MI', 'DC', 'CT', 'NC', 'MD', 'OH',
#        'TN', 'MN', 'RI', 'ME', 'NV', 'OR', 'UT', 'NM', 'IN', 'AZ', 'ID',
#        'AR', 'WI'))

#     state={'CA':2, 'MA':12, 'KY':11, 'NY':23, 'CO':3, 'VA':31, 'TX':29, 'WA':32, 'IL':9, 'PA':26, 'GA':7,
#        'NH':19, 'MO':17, 'FL':6, 'NJ':20, 'WV':34, 'MI':15, 'DC':5, 'CT':4, 'NC':18, 'MD':13, 'OH':24,
#        'TN':28, 'MN':16, 'RI':27, 'ME':14, 'NV':22, 'OR':25, 'UT':30, 'NM':21, 'IN':10, 'AZ':1, 'ID':8,
#        'AR':0, 'WI':33
#     }
#     state1 = state[state_code]
#     is_top500 = st.selectbox(
#         "if Company is part of the top 500", ('Yes', 'No'))
    
#     top={'Yes':1,'No':0}

#     top500=top[is_top500]

    
    
    
    


    
    
#     submitButton = st.button(label='Predict ')

#     if submitButton:
#         user_report={
#         'state_code':state1,
#         'relationships':relationships,
#         'funding_rounds':funding_rounds,
#         'funding_total_usd':funding_total_usd,
#         'milestones':milestones,
#         'category_code':category1,
#         'is_top500':top500
#     }
#     report_data = pd.DataFrame(user_report,index=[0])
#     return report_data
def sprediction (input_data):

    #changing to numpy array
    id_as_numpy=np.asarray(input_data)
    #reshape array
    reshape=id_as_numpy.reshape(1,-1)
    
    prediction = loaded_model.predict(reshape)
    print(prediction)
    
    if (prediction[0]==0):
        return 'unsuccessful'
    else:
        return'successful'



def main ():
    relationships = st.number_input("Relationships", 0, 120)
    
    funding_rounds = st.number_input("Funding Rounds", 0, 120)
    funding_total_usd = st.number_input("Total Funding")
    milestones = st.number_input("milestones", 0, 120)
    category_code = st.selectbox("category", ('music', 'enterprise', 'web', 'software', 'games_video',
       'network_hosting', 'finance', 'mobile', 'education',
       'public_relations', 'security', 'other', 'photo_video', 'hardware',
       'ecommerce', 'advertising', 'travel', 'fashion', 'analytics',
       'consulting', 'cleantech', 'search', 'semiconductor', 'social',
       'biotech', 'medical', 'automotive', 'messaging', 'manufacturing',
       'hospitality', 'news', 'transportation', 'sports', 'real_estate',
       'health'))
    
    category = {'music':19, 'enterprise':8, 'web':34, 'software':30, 'games_video':11,
       'network_hosting':20, 'finance':10, 'mobile':18, 'education':7,
       'public_relations':24, 'security':27, 'other':22, 'photo_video':23, 'hardware':12,
       'ecommerce':6, 'advertising':0, 'travel':33, 'fashion':9, 'analytics':1,
       'consulting':5, 'cleantech':4, 'search':26, 'semiconductor':28, 'social':29,
       'biotech':3, 'medical':16, 'automotive':2, 'messaging':17, 'manufacturing':15,
       'hospitality':14, 'news':21, 'transportation':32, 'sports':31, 'real_estate':25,
       'health':13}
    category1=category[category_code]
    state_code=st.selectbox("state", ('CA', 'MA', 'KY', 'NY', 'CO', 'VA', 'TX', 'WA', 'IL', 'PA', 'GA',
       'NH', 'MO', 'FL', 'NJ', 'WV', 'MI', 'DC', 'CT', 'NC', 'MD', 'OH',
       'TN', 'MN', 'RI', 'ME', 'NV', 'OR', 'UT', 'NM', 'IN', 'AZ', 'ID',
       'AR', 'WI'))

    state={'CA':2, 'MA':12, 'KY':11, 'NY':23, 'CO':3, 'VA':31, 'TX':29, 'WA':32, 'IL':9, 'PA':26, 'GA':7,
       'NH':19, 'MO':17, 'FL':6, 'NJ':20, 'WV':34, 'MI':15, 'DC':5, 'CT':4, 'NC':18, 'MD':13, 'OH':24,
       'TN':28, 'MN':16, 'RI':27, 'ME':14, 'NV':22, 'OR':25, 'UT':30, 'NM':21, 'IN':10, 'AZ':1, 'ID':8,
       'AR':0, 'WI':33
    }
    state1 = state[state_code]
    is_top500 = st.selectbox(
        "if Company is part of the top 500", ('Yes', 'No'))
    
    top={'Yes':1,'No':0}

    top500=top[is_top500]
    output=''
    if st.button(label='Predict '):
        output=sprediction([relationships,funding_rounds,funding_total_usd,milestones,category1,state1,top500])
    
    st.success(output)


        
# user_data = get_user_input()

# SVMC = SVC(probability=True)
# svc_param_grid = {'kernel': ['rbf'], 
#                   'gamma': [ 0.001, 0.01, 0.1, 1],
#                   'C': [1, 10, 50, 100,200,300, 1000]}

# kfold = StratifiedKFold(n_splits=10)

# gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= -1, verbose = 1)

# gsSVMC.fit(X_train,y_train)

# user_result=gsSVMC.predict(user_data)

# output=''
# if user_result[0]==0:
#     output='unsuccesful'
# else:
#     outpput='succesful'

# st.write(output)

# def main():
#     display()
#     input_details_df = get_user_input()

   


st.sidebar.header("Introduction")

if __name__ == '__main__':
    main()

def space():
    st.markdown(
    f"""
    <style>
        <br></br>
    </style>
    """,
    unsafe_allow_html=True
    )
space()  


# col2.image("stroke1.jpg")
# col2.image("stroke3.jpg")
# col2.image("stroke2.jpg")
# #col1.image("images.jfif")
# #col1.image("images.jfif")


# video_file = open('video.mov','rb')
# video_bytes = video_file.read()