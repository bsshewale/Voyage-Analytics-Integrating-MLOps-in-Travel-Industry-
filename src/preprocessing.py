import pandas as pd
import joblib
from sklearn.preprocessing import TargetEncoder
from .config import MODEL_DIR

FLIGHT_TYPE_MAP = {
    "economy": 0,
    "business": 1,
    "firstClass": 2
}

def fit_encoders(df):
    from_encoder = TargetEncoder()
    to_encoder = TargetEncoder()
    agency_encoder = TargetEncoder()

    df["from"] = from_encoder.fit_transform(df[["from"]], df["price"])
    df["to"] = to_encoder.fit_transform(df[["to"]], df["price"])
    df["agency"] = agency_encoder.fit_transform(df[["agency"]], df["price"])
    df["flightType"] = df["flightType"].map(FLIGHT_TYPE_MAP)

    joblib.dump(from_encoder, f"{MODEL_DIR}/from_encoder.pkl")
    joblib.dump(to_encoder, f"{MODEL_DIR}/to_encoder.pkl")
    joblib.dump(agency_encoder, f"{MODEL_DIR}/agency_encoder.pkl")
    joblib.dump(FLIGHT_TYPE_MAP, f"{MODEL_DIR}/flight_type_map.pkl")

    joblib.dump(df["from"].unique().tolist(), f"{MODEL_DIR}/from_options.pkl")
    joblib.dump(df["to"].unique().tolist(), f"{MODEL_DIR}/to_options.pkl")
    joblib.dump(df["agency"].unique().tolist(), f"{MODEL_DIR}/agency_options.pkl")

    return df


def transform_input(
    from_city, to_city, agency, flight_type,
    time, distance, date,
    from_encoder, to_encoder, agency_encoder, flight_type_map
):
    date = pd.to_datetime(date)

    return pd.DataFrame([{
        "from": from_encoder.transform([[from_city]])[0][0],
        "to": to_encoder.transform([[to_city]])[0][0],
        "flightType": flight_type_map[flight_type],
        "time": time,
        "distance": distance,
        "agency": agency_encoder.transform([[agency]])[0][0],
        "day": date.day,
        "month": date.month,
        "year": date.year
    }])
