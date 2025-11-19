import os
import pandas as pd
from configs.config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def preprocess_movielens_1m():
    """
    Preprocess the MovieLens 1M dataset into a single CSV.
    Output: data/processed/movielens_1m_processed.csv
    """
    print("🚀 Starting preprocessing for MovieLens 1M...")

    # File paths
    movies_path = os.path.join(RAW_DATA_PATH, "ml-1m", "movies.dat")
    ratings_path = os.path.join(RAW_DATA_PATH, "ml-1m", "ratings.dat")
    users_path = os.path.join(RAW_DATA_PATH, "ml-1m", "users.dat")

    # Check files exist
    for f in [movies_path, ratings_path, users_path]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"❌ Missing required file: {f}")

    # Load raw data
    print("📥 Loading raw files...")
    movies = pd.read_csv(movies_path, sep="::", engine="python", names=["movie_id", "title", "genres"])
    ratings = pd.read_csv(ratings_path, sep="::", engine="python", names=["user_id", "movie_id", "rating", "timestamp"])
    users = pd.read_csv(users_path, sep="::", engine="python",
                        names=["user_id", "gender", "age", "occupation", "zip"])

    print(f"✅ Loaded {len(ratings):,} ratings, {len(users):,} users, {len(movies):,} movies")

    # Merge datasets
    df = ratings.merge(users, on="user_id").merge(movies, on="movie_id")

    # Encode IDs for neural network compatibility
    df["user_idx"] = df["user_id"].astype("category").cat.codes
    df["movie_idx"] = df["movie_id"].astype("category").cat.codes

    # Sort by time for sequence models
    df = df.sort_values(["user_idx", "timestamp"]).reset_index(drop=True)

    # Output folder
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_PATH, "movielens_1m_processed.csv")

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Preprocessing complete! Saved → {output_path}")
    print(f"Users: {df['user_idx'].nunique()}, Movies: {df['movie_idx'].nunique()}, Ratings: {len(df)}")


if __name__ == "__main__":
    preprocess_movielens_1m()
