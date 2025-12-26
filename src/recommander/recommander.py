import numpy as np

def recommend_hotels(
    user_id,
    user_hotel_matrix,
    similarity_df,
    top_n=5
):
    if user_id not in user_hotel_matrix.index:
        return []

    similar_users = (
        similarity_df[user_id]
        .sort_values(ascending=False)
        .iloc[1:6]
    )

    weighted_scores = np.dot(
        similar_users.values,
        user_hotel_matrix.loc[similar_users.index]
    )

    scores = dict(zip(
        user_hotel_matrix.columns,
        weighted_scores
    ))

    visited = user_hotel_matrix.loc[user_id]
    scores = {k: v for k, v in scores.items() if visited[k] == 0}

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
