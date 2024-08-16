import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the preprocessing pipeline and the trained model
pipeline = joblib.load('preprocessing_pipeline.pkl')
model = joblib.load('CatBoost_model.pkl')

# Title of the Streamlit app
st.title("Predictive Model App")

# Function to get user input
def get_user_input():
    # Create an empty dictionary to store user inputs
    user_input = {}
    
    # List of features
    numeric_features = ['Annual_income', 'Employed_days', 'Family_Members', 'CHILDREN', 'Birthday_count']
    categorical_features = ['Housing_type', 'Type_Occupation', 'Marital_status', 'EDUCATION', 'GENDER']
    
    # Get user inputs for numeric features
    for feature in numeric_features:
        user_input[feature] = st.number_input(f'Enter {feature}', min_value=0)
    
    # Get user inputs for categorical features
    options_dict = {
        'Housing_type': ['Rented apartment', 'Municipal apartment', 'House / apartment', 'With parents', 'Office apartment', 'Co-op apartment'],
        'Type_Occupation': ['Security staff', 'Pensioner', 'Laborers', 'Core staff', 'Cleaning staff', 'Commercial associate', 'Sales staff', 'Medicine staff', 'Managers', 'Accountants', 'Drivers', 'High skill tech staff', 'Low-skill Laborers', 'Secretaries', 'Waiters/barmen staff', 'Private service staff', 'Realty agents', 'HR staff', 'Cooking staff'],
        'Marital_status': ['Single / not married', 'Married', 'Civil marriage', 'Separated', 'Widow'],
        'EDUCATION': ['Higher education', 'Secondary / secondary special', 'Incomplete higher', 'Lower secondary', 'Academic degree'],
        'GENDER': ['F', 'M']
    }
    
    for feature, options in options_dict.items():
        user_input[feature] = st.selectbox(f'Select {feature}', options)
    
    return pd.DataFrame(user_input, index=[0])

# Main function to run the app
def main():
    # Get user input
    input_df = get_user_input()

    # Preprocess the input using the loaded pipeline
    input_processed = pipeline.transform(input_df)

    # Make predictions
    prediction = model.predict(input_processed)
    prediction_prob = model.predict_proba(input_processed)
    
    # Display the prediction and probabilities
    st.write(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
    st.write(f"Prediction Probability: {prediction_prob[0]}")

if __name__ == '__main__':
    main()
