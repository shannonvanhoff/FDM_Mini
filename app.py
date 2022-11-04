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


import base64
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.model_selection import train_test_split
import pickle


#loading the saved model
loaded_model = pickle.load(open('models/trained_model.sav','rb'))



col1, col2= st.columns([2,1])

st.markdown(f'''
    <style>
    section[data-testid="stSidebar"] .css-ng1t4o {{width: 200px;}}
    </style>
''',unsafe_allow_html=True)


def display():
    col1.title('Startup Success Prediction Application')
    


 


def sprediction (input_data):

    #changing to numpy array
    id_as_numpy=np.asarray(input_data)
    #reshape array
    reshape=id_as_numpy.reshape(1,-1)
    
    prediction = loaded_model.predict(reshape)
    print(prediction)
    
    if (prediction[0]==0):
        return 'startup will be unsuccessful'
    else:
        return'startup will be  successful'



def main ():
    col1.title('Startup Success Prediction Application')
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

