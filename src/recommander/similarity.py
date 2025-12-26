import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def compute_user_similarity(user_hotel_matrix: pd.DataFrame):
    similarity = cosine_similarity(user_hotel_matrix)
    return pd.DataFrame(
        similarity,
        index=user_hotel_matrix.index,
        columns=user_hotel_matrix.index
    )
