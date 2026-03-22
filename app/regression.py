import pandas as pd
import streamlit as st
import joblib
import numpy as np

# --- 1. Load the Model and Feature List ---
linear_model = joblib.load('linear_regression_model.joblib')

# Assuming X_final was the DataFrame used for training the model
# We need the exact order of columns for prediction
model_features = X_final.columns.tolist()

# Original categorical columns for consistent one-hot encoding
categorical_cols = joblib.load('categorical_columns_for_encoding.joblib')

# --- 2. Streamlit UI ---
st.title('Flight Price Prediction App')
st.write('Predict the price of a flight based on its characteristics.')

# Input fields
st.sidebar.header('Flight Details')

# Numerical Inputs
distance = st.sidebar.slider('Distance (km)', min_value=100.0, max_value=1000.0, value=500.0, step=10.0)
month = st.sidebar.slider('Month (1-12)', min_value=1, max_value=12, value=6, step=1)
day_of_week = st.sidebar.slider('Day of Week (0=Monday, 6=Sunday)', min_value=0, max_value=6, value=3, step=1)

# Get unique values for dropdowns from the original DataFrame (df)
unique_from = df['from'].unique().tolist()
unique_to = df['to'].unique().tolist()
unique_flightType = df['flightType'].unique().tolist()
unique_agency = df['agency'].unique().tolist()

# Categorical Inputs
flight_from = st.sidebar.selectbox('Origin City', sorted(unique_from))
flight_to = st.sidebar.selectbox('Destination City', sorted(unique_to))
flight_type = st.sidebar.selectbox('Flight Type', sorted(unique_flightType))
agency = st.sidebar.selectbox('Agency', sorted(unique_agency))

# --- 3. Prepare Input for Prediction ---
input_data = {
    'distance': distance,
    'month': month,
    'day_of_week': day_of_week,
    'from': flight_from,
    'to': flight_to,
    'flightType': flight_type,
    'agency': agency
}

input_df = pd.DataFrame([input_data])

# Apply one-hot encoding to the categorical features in the input DataFrame
# Ensure all possible one-hot encoded columns from training are present
input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

# Add missing columns (those present in model_features but not in input_encoded)
# and fill with 0 to match the training data's structure
missing_cols = set(model_features) - set(input_encoded.columns)
for c in missing_cols:
    input_encoded[c] = 0

# Ensure the order of columns is the same as during training
input_final = input_encoded[model_features]

# --- 4. Make Prediction ---
if st.sidebar.button('Predict Price'):
    prediction = linear_model.predict(input_final)[0]
    st.success(f'Predicted Flight Price: ${prediction:,.2f}')
