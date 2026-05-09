import streamlit as st
import pickle
import pandas as pd
import requests

st.set_page_config(page_title="PopcornAI", layout="wide")

# Fetch movie poster using TMDB API
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
    try:
        response = requests.get(url, timeout=3)
        data = response.json()
        poster_path = data.get('poster_path')
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    except Exception:
        pass
    return "https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500"

# Load pre-calculated files and cache them in memory
@st.cache_resource
def load_models():
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return movies, similarity

try:
    movies, similarity = load_models()
except FileNotFoundError:
    st.error("Please run Ml.py locally first, and ensure 'movie_dict.pkl' and 'similarity.pkl' are uploaded to your GitHub repository.")
    st.stop()

# Recommendation logic (Now lightning fast!)
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]
    
    recommended_movies = []
    recommended_posters = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_posters.append(fetch_poster(movie_id))
        
    return recommended_movies, recommended_posters

# UI Header
st.markdown("<h1 style='text-align: center; color: #E50914;'>PopcornAI: Movie Recommender</h1>", unsafe_allow_html=True)
st.write("---")

selected_movie_name = st.selectbox(
    "Search or select a movie you loved:",
    movies['title'].values
)

 # 3. Vectorization & Similarity (Optimized to stay under GitHub's 100MB limit!)
tfidf = TfidfVectorizer(max_features=1500, stop_words='english')
vector = tfidf.fit_transform(new_df['tags']).toarray()

# Calculate similarity and immediately convert it to float32 to save massive space
import numpy as np
similarity = cosine_similarity(vector).astype(np.float32)

# 4. Save the pre-calculated objects
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("Success! Generated ultra-compressed models under GitHub's limit.")