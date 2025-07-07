from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import ndcg_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -------------------------- Preprocessing --------------------------
def preprocess(text):
    if pd.isna(text): return ''
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalnum()]
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in tokens])

# -------------------------- Data Loading --------------------------
users = pd.read_csv("u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"])
ratings = pd.read_csv("u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])
genres_raw = pd.read_csv("u.genre", sep="|", names=["genre", "genre_id"], engine='python')
movies = pd.read_csv("u.item", sep="|", encoding="latin-1", header=None)

genre_cols = genres_raw["genre"].tolist()
movie_cols = ["movie_id", "title", "release_date", "video_release_date", "IMDb_URL"] + genre_cols
movies.columns = movie_cols

movies["genres_str"] = movies[genre_cols].apply(lambda row: ' '.join([g for g in genre_cols if row[g] == 1]), axis=1)
movies["genres_str"] = movies["genres_str"].apply(preprocess)
movies["year"] = movies["release_date"].str.extract(r'(\d{4})').fillna(0).astype(int)
movies["full_text"] = (movies["title"].fillna("") + " " + movies["genres_str"].fillna("")).apply(preprocess)

bm25_corpus = movies["full_text"].apply(str.split).tolist()
bm25 = BM25Okapi(bm25_corpus)

# -------------------------- Recommendation Setup --------------------------
recommendations = []
for user_id in users["user_id"].unique():
    user_rated = ratings[ratings["user_id"] == user_id]
    liked_movies = user_rated[user_rated["rating"] >= 4]["movie_id"].tolist()
    query = " ".join(movies[movies["movie_id"].isin(liked_movies)]["genres_str"].tolist())
    query_tokens = preprocess(query).split()
    scores = bm25.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:10]

    for idx in top_indices:
        movie_id = movies.iloc[idx]["movie_id"]
        label = 0
        if not ratings[(ratings["user_id"] == user_id) & (ratings["movie_id"] == movie_id)].empty:
            label = 1 if ratings[(ratings["user_id"] == user_id) & (ratings["movie_id"] == movie_id)]["rating"].values[0] >= 4 else 0
        recommendations.append({
            "user_id": user_id,
            "movie_id": movie_id,
            "BM25_score": scores[idx],
            "genres_str": movies.iloc[idx]["genres_str"],
            "label": label
        })

df = pd.DataFrame(recommendations)

def compute_genre_overlap(row):
    user_movies = ratings[ratings["user_id"] == row["user_id"]]
    liked_movies = movies[movies["movie_id"].isin(user_movies[user_movies["rating"] >= 4]["movie_id"])]
    liked_genres = set(" ".join(liked_movies["genres_str"]).split())
    target_genres = set(row["genres_str"].split())
    return len(liked_genres & target_genres)

def compute_cosine(row):
    user_movies = ratings[ratings["user_id"] == row["user_id"]]
    liked_genres = " ".join(movies[movies["movie_id"].isin(user_movies[user_movies["rating"] >= 4]["movie_id"])]["genres_str"])
    tfidf = TfidfVectorizer()
    try:
        tfidf_matrix = tfidf.fit_transform([liked_genres, row["genres_str"]])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        return 0.0

df["genre_match"] = df.apply(compute_genre_overlap, axis=1)
df["cosine_sim"] = df.apply(compute_cosine, axis=1)

# -------------------------- Train Models --------------------------
features = ["genre_match", "cosine_sim", "BM25_score"]
X = df[features].fillna(0)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LogReg": LogisticRegression(),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}
for model in models.values():
    model.fit(X_train, y_train)

# -------------------------- API Endpoint --------------------------
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    user_query = data.get("query", "")
    N = int(data.get("top_n", 10))
    filtered = data.get("filter", "none").lower()

    query_tokens = preprocess(user_query).split()
    query_scores = bm25.get_scores(query_tokens)
    movies["BM25_query_score"] = query_scores

    avg_rating = ratings.groupby("movie_id")["rating"].mean().reset_index()
    merged = movies.merge(avg_rating, on="movie_id", how="left")
    top_query_movies = merged.sort_values("BM25_query_score", ascending=False).head(N).copy()

    def compute_query_features(row):
        genre_match = len(set(query_tokens) & set(row["genres_str"].split()))
        try:
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform([user_query, row["full_text"]])
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            cosine_sim = 0.0
        return pd.Series([genre_match, cosine_sim])

    top_query_movies[["genre_match", "cosine_sim"]] = top_query_movies.apply(compute_query_features, axis=1)
    top_query_movies["BM25_score"] = top_query_movies["BM25_query_score"]

    results = {}
    for name, model in models.items():
        X_query = top_query_movies[features].fillna(0)
        scores = model.predict_proba(X_query)[:, 1]
        top_query_movies[f"{name}_score"] = scores

        relevance_labels = top_query_movies["genres_str"].apply(lambda g: int(len(set(g.split()) & set(query_tokens)) > 0)).values.reshape(1, -1)
        ndcg_val = ndcg_score(relevance_labels, scores.reshape(1, -1))

        ranked = top_query_movies.sort_values(by=f"{name}_score", ascending=False).reset_index(drop=True)
        results[name] = {
            "NDCG": round(ndcg_val, 4),
            "recommendations": [
                {
                    "rank": i + 1,
                    "title": row["title"],
                    "genres": row["genres_str"],
                    "rating": round(row["rating"], 2) if not pd.isna(row["rating"]) else None,
                    "year": int(row["year"]),
                    "score": round(row[f"{name}_score"], 4)
                } for i, row in ranked.iterrows()
            ]
        }

    if filtered in ["rating", "both"]:
        sorted_by_rating = top_query_movies.sort_values("rating", ascending=False).head(N).reset_index(drop=True)
        sorted_by_rating["rel_score"] = 1 - (np.abs(sorted_by_rating["rating"] - 5) / 4)
        results["Top_by_Rating"] = [
            {
                "rank": i + 1,
                "title": row["title"],
                "rating": round(row["rating"], 2),
                "Score": round(row["rel_score"], 4)
            } for i, row in sorted_by_rating.iterrows()
        ]

    if filtered in ["year", "both"]:
        sorted_by_year = top_query_movies.sort_values("year", ascending=False).head(N).reset_index(drop=True)
        year_range = 2001 - movies["year"].min()
        sorted_by_year["rel_score"] = 1 - (np.abs(sorted_by_year["year"] - 2001) / year_range)
        results["Top_by_Year"] = [
            {
                "rank": i + 1,
                "title": row["title"],
                "year": int(row["year"]),
                "Score": round(row["rel_score"], 4)
            } for i, row in sorted_by_year.iterrows()
        ]

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
