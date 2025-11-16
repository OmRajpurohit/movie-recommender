import csv
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from rapidfuzz import process, fuzz

print("Loading data...")
url = "https://drive.google.com/uc?export=download&id=1OGzQMEu-Akvv4i_8nYcjR9KTO7KlmNCz-Wd3"


df = pd.read_csv(url,
                engine="python",
                on_bad_lines="skip",
                quoting=csv.QUOTE_MINIMAL)
df['genres'] =df['genres'].fillna('').astype(str)
df['tagline'] =df['genres'].fillna('').astype(str)
df['keywords'] =df['keywords'].fillna('').astype(str)

df['tags'] = df['genres'] +df['tagline'] +df['keywords']
df =df.drop(columns =['genres','tagline','keywords'])



# Clean basic columns
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce')
df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
df['tags'] = df['tags'].fillna("").astype(str)

df = df.dropna(subset=['vote_average', 'vote_count']).reset_index(drop=True)

print("Calculating IMDb weighted rating...")

C = df['vote_average'].mean()           # global mean
m = df['vote_count'].quantile(0.90)  # 80th percentile threshold

def imdb_score(row):
    v = row['vote_count']
    R = row['vote_average']
    return (v/(v+m))*R + (m/(v+m))*C

df['imdb_score'] = df.apply(imdb_score, axis=1)

popular = df.sort_values('imdb_score', ascending=False).reset_index(drop=True)

# compacting id which maybe string to 32 int
from sklearn.preprocessing import LabelEncoder
item_le = LabelEncoder()
df['item_idx'] = item_le.fit_transform(df['id'].astype(str)).astype(np.int32)

# check
n_items = df['item_idx'].nunique()
print("Unique item ", n_items)
print("Original id for item_idx 0->",item_le.inverse_transform([0])[0])

print("Creating TF-IDF matrix...")

tfidf = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2)
)
df['combined'] = df['tags'] + df['original_language']
tfidf_matrix = tfidf.fit_transform(df['combined'])
print("TF-IDF shape:", tfidf_matrix.shape)


df['title_lower'] = df['title'].str.lower()
title_to_index = pd.Series(df.index, index=df['title_lower']).to_dict()
titles_list = list(title_to_index.keys())   # for rapidfuzz


def find_best_title(query, limit=5):
    query = query.lower()
    matches = process.extract(
        query,
        titles_list,
        scorer=fuzz.WRatio,
        limit=limit
    )
    # returns list of (title, score, index)
    return matches


def get_similar_movies(idx, top_n=10):
    cosine_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    
    similar_idxs = cosine_scores.argsort()[-top_n-1:-1][::-1]
    
    return df.iloc[similar_idxs][['title', 'imdb_score']]

def get_popular_movies(n=10):
    return popular[['title', 'imdb_score']].head(n)


def hybrid_recommend(title, top_n=10):
    title = title.lower().strip()

    # Case 1 — Exact title match
    if title in title_to_index:
        idx = title_to_index[title]
    
    else:
        # Case 2 — Fuzzy match using rapidfuzz
        matches = find_best_title(title)
        if len(matches) == 0:
            return None, "No similar titles found."
        
        best_match = matches[0][0]
        idx = title_to_index[best_match]
        msg = f"Did you mean: {best_match}?"
        print(msg)
    
    # Case 3 — No tags → use popularity
    if df.loc[idx, 'tags'].strip() == "":
        print("No tags found → Using popularity fallback.")
        return get_popular_movies(top_n), None
    
    # Case 4 — Use content similarity
    similar = get_similar_movies(idx, top_n)
    return similar, None


# quick test
movies, msg = hybrid_recommend("hum sath sath hai", top_n=5)
print(movies)
print("Message:", msg)








