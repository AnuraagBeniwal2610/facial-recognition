import pandas as pd
import numpy as np
import streamlit as st
import Training 
import Testing
import LiveCapture
from os import listdir
from os.path import isfile, join
from PIL import Image
import pickle
from datetime import datetime
import joblib 

st.sidebar.title("About")

st.sidebar.info("Face recognition is achieved using Deep Learning's sub-field that is Convolutional Neural Network (CNN). It is a multi-layer network trained to perform a specific task using classification. Transfer learning of a trained CNN model that is AlexNet is done for face recognition.")

st.title('FACE RECOGNITION POC')


st.sidebar.title("Train Neural Network")
if st.sidebar.button('Train Model'): 
     Training.train()
     st.sidebar.write("MODEL TRAINED")

st.sidebar.title("Pick An Image")
onlyfiles = [f for f in listdir("C:/Users/Anuraag/Desktop/OLD OFFICE/FACE_REC_SVC/") if isfile(join("C:/Users/Anuraag/Desktop/OLD OFFICE/FACE_REC_SVC/", f))]
imageselect = st.sidebar.selectbox("", onlyfiles)

st.write("")
image = Image.open("C:/Users/Anuraag/Desktop/OLD OFFICE/FACE_REC_SVC/" + imageselect)
st.image(image,caption="", use_column_width=True)


model = joblib.load('finalized_model.sav')  
st.sidebar.title("Use Neural Network to Predict Employee")
if st.sidebar.button('Predict Employee'):
    prediction = Testing.predict(imageselect)
    st.sidebar.title(prediction)

st.sidebar.title("Capture Live Image")
if st.sidebar.button('Click to Capture'): 
     LiveCapture.capture()
     st.sidebar.write("Image saved as NewPicture.jpg")

