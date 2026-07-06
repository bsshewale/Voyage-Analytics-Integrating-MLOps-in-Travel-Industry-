import streamlit as st
import requests

st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Flight Price Prediction")

st.write("Enter flight details below.")

# Cities
cities = [
    "Recife (PE)",
    "Florianopolis (SC)",
    "Brasilia (DF)",
    "Aracaju (SE)",
    "Salvador (BH)",
    "Campo Grande (MS)",
    "Sao Paulo (SP)",
    "Natal (RN)",
    "Rio de Janeiro (RJ)"
]

# Flight Type
flight_types = [
    "economy",
    "business",
    "firstClass"
]

# Agencies
agencies = [
    "FlyingDrops",
    "CloudFy",
    "Rainbow"
]

source = st.selectbox("From", cities)

destination = st.selectbox("To", cities)

flight_type = st.selectbox(
    "Flight Type",
    flight_types
)

agency = st.selectbox(
    "Agency",
    agencies
)

time = st.number_input(
    "Flight Time (hours)",
    min_value=0.1,
    value=1.5
)

distance = st.number_input(
    "Distance (km)",
    min_value=100.0,
    value=500.0
)

date = st.date_input("Travel Date")

if st.button("Predict Price"):

    payload = {
        "from": source,
        "to": destination,
        "flightType": flight_type,
        "agency": agency,
        "time": time,
        "distance": distance,
        "date": str(date)
    }

    try:

        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json=payload
        )

        result = response.json()

        if result["success"]:
            st.success(
                f"Predicted Flight Price: ₹ {result['predicted_price']:.2f}"
            )
        else:
            st.error(result["error"])

    except Exception as e:
        st.error(str(e))