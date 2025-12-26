# import necessary libraries and modules
import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.gender_classifier.inference import predict_gender

# Page config
st.set_page_config(
    page_title="Gender Classification",
    page_icon="🧠",
    layout="centered"
)
# Title and description
st.title("🧠 Gender Classification Model")
st.write("Predict gender using name-based ML model")

# Input form
with st.form("prediction_form"):
    name = st.text_input("Enter Name", placeholder="e.g. Amit Sharma")
    age = st.number_input("Enter Age", min_value=0, max_value=120, step=1)
    submit = st.form_submit_button("Predict")

# On submit
if submit:
    if not name.strip():
        st.error("Name cannot be empty")
    else:
        result = predict_gender(name, age)
        
        # Display results
        st.success(f"**Predicted Gender:** {result['gender'].capitalize()}")
        st.info(f"**Confidence:** {result['confidence']}%")
