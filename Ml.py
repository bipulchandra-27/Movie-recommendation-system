import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

print("Starting movie data preprocessing...")

# 1. Load the raw datasets
try:
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
except FileNotFoundError:
    print("Error: Please make sure 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv' are in this folder!")
    exit()

# 2. Merge and select relevant features
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# Helper functions to parse metadata
def convert_json_to_list(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def get_top_cast(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

# Apply parsing
movies['genres'] = movies['genres'].apply(convert_json_to_list)
movies['keywords'] = movies['keywords'].apply(convert_json_to_list)
movies['cast'] = movies['cast'].apply(get_top_cast)
movies['crew'] = movies['crew'].apply(fetch_director)

# Remove spaces from names (e.g., "Johnny Depp" -> "JohnnyDepp") to avoid tag confusion
def collapse_spaces(L):
    return [i.replace(" ", "") for i in L]

movies['cast'] = movies['cast'].apply(collapse_spaces)
movies['crew'] = movies['crew'].apply(collapse_spaces)
movies['genres'] = movies['genres'].apply(collapse_spaces)
movies['keywords'] = movies['keywords'].apply(collapse_spaces)

# Combine all elements into a single tags column
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create clean simplified dataframe
new_df = movies[['movie_id', 'title', 'tags']].copy()
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

print("Data cleaning complete. Calculating similarity matrix...")

# 3. Vectorization & Similarity (Optimized to be under 15MB!)
# Limiting features to 1000 keeps the mathematical space compact and lightning fast
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
vector = tfidf.fit_transform(new_df['tags']).toarray()

# Convert matrix to float32 to cut file storage size exactly in half 
similarity = cosine_similarity(vector).astype(np.float32)

# 4. Save the pre-calculated binary files
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("---")
print("Success! Your files are now tiny and ready for GitHub upload.")
print("Generated: 'movie_dict.pkl' and 'similarity.pkl'")