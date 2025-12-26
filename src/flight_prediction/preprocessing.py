import os
import pandas as pd
import joblib
from sklearn.preprocessing import TargetEncoder

# =====================
# CONFIG
# =====================
DATA_PATH = r"H:\\Voyage-Analytics-Integrating-MLOps-in-Travel-Industry-\\data\\flight_prediction"
MODEL_DIR = r"H:\\Voyage-Analytics-Integrating-MLOps-in-Travel-Industry-\\model\\flight_prediction"

os.makedirs(MODEL_DIR, exist_ok=True)

FLIGHT_TYPE_MAP = {
    "economy": 0,
    "business": 1,
    "firstClass": 2
}

print("Starting preprocessing...")

# =====================
# LOAD DATA
# =====================
df = pd.read_csv(DATA_PATH+'\\flights.csv')

# Type casting
df['price']=df['price'].astype(int)
df['date']=df['date'].astype('datetime64[ns]')

# Droping the columns
df.drop(columns=['travelCode', 'userCode'], inplace=True)
     

# Extract the day, month, and year from the date column
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Drop original
df.drop('date', axis=1, inplace=True)

# =====================
# ENCODERS
# =====================
from_encoder = TargetEncoder()
to_encoder = TargetEncoder()
agency_encoder = TargetEncoder()

df["from"] = from_encoder.fit_transform(df[["from"]], df["price"])
df["to"] = to_encoder.fit_transform(df[["to"]], df["price"])
df["agency"] = agency_encoder.fit_transform(df[["agency"]], df["price"])

df["flightType"] = df["flightType"].map(FLIGHT_TYPE_MAP)

# =====================
# SAVE ENCODERS
# =====================
joblib.dump(from_encoder, f"{MODEL_DIR}/from_encoder.pkl")
joblib.dump(to_encoder, f"{MODEL_DIR}/to_encoder.pkl")
joblib.dump(agency_encoder, f"{MODEL_DIR}/agency_encoder.pkl")
joblib.dump(FLIGHT_TYPE_MAP, f"{MODEL_DIR}/flight_type_map.pkl")

# =====================
# SAVE PROCESSED DATA (OPTIONAL)
# =====================
df.to_csv(f"{DATA_PATH}/processed_flights.csv", index=False)

print("Preprocessing completed successfully.")
