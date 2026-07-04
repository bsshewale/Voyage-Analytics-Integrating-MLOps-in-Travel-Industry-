import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("H:\\Voyage-Analytics-Integrating-MLOps-in-Travel-Industry-\\model\\flight_prediction\\linear_regression_model.joblib")

# Load categorical columns
categorical_cols = joblib.load("H:\\Voyage-Analytics-Integrating-MLOps-in-Travel-Industry-\\model\\flight_prediction\\categorical_columns_for_encoding.joblib")

st.title("✈️ Flight Price Prediction App")

st.write("Enter flight details to predict the price")

# User Inputs   
distance = st.number_input("Distance (km)", min_value=0.0)

month = st.selectbox( 
    "Month",
    list(range(1, 13))
)

day_of_week = st.selectbox(
    "Day of Week (0 = Monday, 6 = Sunday)",
    list(range(7))
)

from_city = st.selectbox(
    "From City",
    [
        "Aracaju (SE)",
        "Brasilia (DF)",
        "Campo Grande (MS)",
        "Florianopolis (SC)",
        "Natal (RN)",
        "Recife (PE)",
        "Rio de Janeiro (RJ)",
        "Salvador (BH)",
        "Sao Paulo (SP)"
    ]
)

to_city = st.selectbox(
    "To City",
    [
        "Aracaju (SE)",
        "Brasilia (DF)",
        "Campo Grande (MS)",
        "Florianopolis (SC)",
        "Natal (RN)",
        "Recife (PE)",
        "Rio de Janeiro (RJ)",
        "Salvador (BH)",
        "Sao Paulo (SP)"
    ]
)

flight_type = st.selectbox(
    "Flight Type",
    ["economic", "firstClass", "premium"]
)

agency = st.selectbox(
    "Agency",
    ["CloudFy", "FlyingDrops", "Rainbow"]
)

# Prediction Button
if st.button("Predict Price"):

    input_data = {
        "distance": distance,
        "month": month,
        "day_of_week": day_of_week,
        "from": from_city,
        "to": to_city,
        "flightType": flight_type,
        "agency": agency
    }

    df = pd.DataFrame([input_data])

    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    # Match training features
    model_features = model.feature_names_in_

    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[model_features]

    # Predict
    prediction = model.predict(df_encoded)

    st.success(f"💰 Predicted Flight Price: {prediction[0]:.2f}")