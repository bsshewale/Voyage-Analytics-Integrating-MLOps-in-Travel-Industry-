import pandas as pd
import streamlit as st
import joblib
import numpy as np

# Load model
linear_model = joblib.load(
'H:\\Voyage-Analytics-Integrating-MLOps-in-Travel-Industry\\model\\flight_prediction\\linear_regression_model.joblib'
)

# Load feature order
model_features = joblib.load(
'H:\\Voyage-Analytics-Integrating-MLOps-in-Travel-Industry\\model\\flight_prediction\\model_features.joblib'
)

# Load categorical columns
categorical_cols = joblib.load(
'H:\\Voyage-Analytics-Integrating-MLOps-in-Travel-Industry\\model\\flight_prediction\\categorical_columns_for_encoding.joblib'
)

# Load dataset for dropdown values
df = pd.read_csv(
'H:\\Voyage-Analytics-Integrating-MLOps-in-Travel-Industry\\data\\flight_prediction\\processed_flights.csv'
)

# UI
st.title("Flight Price Prediction App")

st.sidebar.header("Flight Details")

distance = st.sidebar.slider("Distance (km)",100.0,1000.0,500.0)
month = st.sidebar.slider("Month",1,12,6)
day_of_week = st.sidebar.slider("Day of Week",0,6,3)

unique_from = df['from'].unique().tolist()
unique_to = df['to'].unique().tolist()
unique_flightType = df['flightType'].unique().tolist()
unique_agency = df['agency'].unique().tolist()

flight_from = st.sidebar.selectbox("Origin City",sorted(unique_from))
flight_to = st.sidebar.selectbox("Destination City",sorted(unique_to))
flight_type = st.sidebar.selectbox("Flight Type",sorted(unique_flightType))
agency = st.sidebar.selectbox("Agency",sorted(unique_agency))

input_data = {
'distance':distance,
'month':month,
'day_of_week':day_of_week,
'from':flight_from,
'to':flight_to,
'flightType':flight_type,
'agency':agency
}

input_df = pd.DataFrame([input_data])

input_encoded = pd.get_dummies(input_df,columns=categorical_cols)

missing_cols = set(model_features) - set(input_encoded.columns)

for c in missing_cols:
    input_encoded[c] = 0

input_final = input_encoded[model_features]

if st.sidebar.button("Predict Price"):
    prediction = linear_model.predict(input_final)[0]
    st.success(f"Predicted Flight Price: ${prediction:,.2f}")