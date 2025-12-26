import pandas as pd

csv_path = r"H:\\Voyage-Analytics-Integrating-MLOps-in-Travel-Industry-\\data\\recommander\\hotel_bookings.csv"
def build_user_hotel_matrix(csv_path: str):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    interactions = (
        df.groupby(["userCode", "name"])
        .agg(
            trips=("travelCode", "count"),
            spend=("total", "sum")
        )
        .reset_index()
    )

    interactions["score"] = (
        interactions["trips"] * 0.7 +
        interactions["spend"] * 0.0005
    )

    user_hotel_matrix = interactions.pivot_table(
        index="userCode",
        columns="name",
        values="score",
        fill_value=0
    )

    return user_hotel_matrix
