import pickle
import streamlit as st
import pandas as pd
from PIL import Image
import joblib
import time

# Load pre-saved model files
model_file = 'model_C=1.0.bin'
dv_file = model_file.replace('.bin', '_dv.joblib')
rf_model_file = model_file.replace('.bin', '_rf.joblib')

with open(model_file, 'rb') as f_in:
    dv, rf_classifier = pickle.load(f_in)

# Streamlit setup
st.set_page_config(
    layout='wide',
    page_title='Telco Customer Churn Prediction'
)

# Title
st.markdown("<h1 style='text-align: center;'>Telco Customer Churn Prediction</h1>", unsafe_allow_html=True)
image = Image.open('tel.jpg')
st.image(image, caption='PHOTO FROM UNSPLASH-FREE STOCK PHOTO')

# Sidebar for user input
st.sidebar.title("Customer Churn Predictor")
st.sidebar.info('This app is created to Predict Customer Churn')

gender = st.sidebar.selectbox('Gender:', ['male', 'female'])
seniorcitizen = st.sidebar.selectbox('Customer is a senior citizen 0 = no, 1 = yes:', [0, 1])
partner = st.sidebar.selectbox('Customer has a partner:', ['yes', 'no'])
dependents = st.sidebar.selectbox('Customer has dependents:', ['yes', 'no'])
phoneservice = st.sidebar.selectbox('Customer has phoneservice:', ['yes', 'no'])
multiplelines = st.sidebar.selectbox('Customer has multiplelines:', ['yes', 'no', 'no_phone_service'])
internetservice = st.sidebar.selectbox('Customer has internetservice:', ['dsl', 'no', 'fiber_optic'])
onlinesecurity = st.sidebar.selectbox('Customer has onlinesecurity:', ['yes', 'no', 'no_internet_service'])
onlinebackup = st.sidebar.selectbox('Customer has onlinebackup:', ['yes', 'no', 'no_internet_service'])
deviceprotection = st.sidebar.selectbox('Customer has deviceprotection:', ['yes', 'no', 'no_internet_service'])
techsupport = st.sidebar.selectbox('Customer has techsupport:', ['yes', 'no', 'no_internet_service'])
streamingtv = st.sidebar.selectbox('Customer has streamingtv:', ['yes', 'no', 'no_internet_service'])
streamingmovies = st.sidebar.selectbox('Customer has streamingmovies:', ['yes', 'no', 'no_internet_service'])
contract = st.sidebar.selectbox('Customer has a contract:', ['month-to-month', 'one_year', 'two_year'])
paperlessbilling = st.sidebar.selectbox('Customer has a paperlessbilling:', ['yes', 'no'])
paymentmethod = st.sidebar.selectbox('Payment Option:', ['bank_transfer_(automatic)', 'credit_card_(automatic)', 'electronic_check', 'mailed_check'])
tenure = st.sidebar.number_input('Number of months the customer has been with the current telco provider :', min_value=0, max_value=240, value=0)
monthlycharges = st.sidebar.number_input('Monthly charges :', min_value=0, max_value=240, value=0)
totalcharges = st.sidebar.number_input('Total charges:', min_value=0, max_value=10000, value=0)

input_dict = {
    "gender": gender,
    "seniorcitizen": seniorcitizen,
    "partner": partner,
    "dependents": dependents,
    "phoneservice": phoneservice,
    "multiplelines": multiplelines,
    "internetservice": internetservice,
    "onlinesecurity": onlinesecurity,
    "onlinebackup": onlinebackup,
    "deviceprotection": deviceprotection,
    "techsupport": techsupport,
    "streamingtv": streamingtv,
    "streamingmovies": streamingmovies,
    "contract": contract,
    "paperlessbilling": paperlessbilling,
    "paymentmethod": paymentmethod,
    "tenure": tenure,
    "monthlycharges": monthlycharges,
    "totalcharges": totalcharges
}

# Prediction button
predict_button = st.sidebar.button("Predict Churn")

if predict_button:
    with st.spinner('Predicting...'):
        time.sleep(2)

        # Prediction logic
        try:
            X = dv.transform([input_dict])  # Transform input data
            y_pred = rf_classifier.predict_proba(X)[0, 1]  # Get churn probability

            churn = y_pred >= 0.5  # Determine churn status
            output_prob = float(y_pred)  # Store probability for display
            output = "will churn" if churn else "will not churn"  # Set output message

            st.sidebar.success('Prediction:')
            st.sidebar.write(f"The model predicts that the customer {output}.")
            st.sidebar.write(f"Churn probability: {output_prob:.4f}")
            
        except Exception as e:
            st.sidebar.error("An error occurred during prediction:")
            st.sidebar.write(e)
