import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from recommander.preprocessor import build_user_hotel_matrix
from recommander.similarity import compute_user_similarity
from recommander.recommander import recommend_hotels

st.set_page_config("Hotel Recommendation System", layout="wide")

st.title("🏨 Travel Hotel Recommendation System")

DATA_PATH = "H:\\Voyage-Analytics-Integrating-MLOps-in-Travel-Industry-\\data\\hotel_recommander\\hotels.csv"

user_hotel_matrix = build_user_hotel_matrix(DATA_PATH)
similarity_df = compute_user_similarity(user_hotel_matrix)

user_id = st.sidebar.selectbox(
    "Select User ID",
    user_hotel_matrix.index.tolist()
)

st.subheader("Recommended Hotels")

recommendations = recommend_hotels(
    user_id,
    user_hotel_matrix,
    similarity_df
)

if not recommendations:
    st.warning("No recommendations available")
else:
    for hotel, score in recommendations:
        st.write(f"**{hotel}** — score: `{round(score, 3)}`")
