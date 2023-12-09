# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:23:20 2023
@author: GreyDoer
"""

import streamlit as st
import joblib
import pandas as pd
import time
from PIL import Image

# Load the saved model
lr_model_top_features = joblib.load('logistic_regression_model_top_features.joblib')

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

# Telco features input
selected_features = ['PhoneService_Yes', 'PhoneService_No', 'MultipleLines_No phone service',
                     'LongTermContract', 'MonthlyCharges', 'MonthlyAverageSpending', 'SeniorCitizen',
                     'InternetService_Fiber optic', 'Contract_Month-to-month', 'InternetService_DSL']

# Create sliders or input fields for the selected features
feature_values = {}
for feature in selected_features:
    if feature == 'SeniorCitizen':
        feature_values[feature] = st.sidebar.checkbox(f'{feature}')
    elif feature == 'PhoneService_Yes' or feature == 'PhoneService_No':
        feature_values[feature] = st.sidebar.checkbox(f'{feature}')
    elif feature == 'InternetService_Fiber optic' or feature == 'InternetService_DSL':
        feature_values[feature] = st.sidebar.checkbox(f'{feature}')
    elif feature == 'MultipleLines_No phone service':
        feature_values[feature] = st.sidebar.checkbox(f'{feature}')
    elif feature == 'LongTermContract':
        feature_values[feature] = st.sidebar.selectbox(f'{feature}', ['Yes', 'No'])
    elif feature == 'Contract_Month-to-month':
        feature_values[feature] = st.sidebar.selectbox(f'{feature}', ['Yes', 'No'])
    else:
        feature_values[feature] = st.sidebar.number_input(f'{feature} ($)')

# Predict churn on button click
predict_button = st.sidebar.button('Predict Churn')

# Create a spinner in the main st context
spinner = st.spinner('Predicting...')

# Display the rolling predicting sign under the "Predict Churn" button
if predict_button:
    with spinner:
        time.sleep(2)  # Simulating prediction time

        # Prediction function
        def predict_churn(features):
            # Convert input features to the correct data types
            features['SeniorCitizen'] = 1 if feature_values['SeniorCitizen'] else 0
            features['PhoneService_Yes'] = 1 if feature_values['PhoneService_Yes'] else 0
            features['PhoneService_No'] = 1 if feature_values['PhoneService_No'] else 0
            features['MultipleLines_No phone service'] = 1 if feature_values['MultipleLines_No phone service'] else 0
            features['LongTermContract'] = 1 if feature_values['LongTermContract'] == 'Yes' else 0
            features['MonthlyCharges'] = feature_values['MonthlyCharges']
            features['MonthlyAverageSpending'] = feature_values['MonthlyAverageSpending']
            features['InternetService_Fiber optic'] = 1 if feature_values['InternetService_Fiber optic'] else 0
            features['Contract_Month-to-month'] = 1 if feature_values['Contract_Month-to-month'] == 'Yes' else 0
            features['InternetService_DSL'] = 1 if feature_values['InternetService_DSL'] else 0

            # Ensure the order of features matches the order during training
            prediction = lr_model_top_features.predict(pd.DataFrame([features], columns=selected_features))
            return prediction[0]  # 0 for not churn, 1 for churn

        # Get prediction
        churn_prediction = predict_churn(feature_values)

        # Display the result
        if churn_prediction == 0:
            st.sidebar.write("The model predicts that the customer will not churn.")
        else:
            st.sidebar.write("The model predicts that the customer will churn.")
