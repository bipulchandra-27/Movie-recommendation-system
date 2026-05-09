import streamlit as st
import pickle
import pandas as pd
import requests

st.set_page_config(page_title="PopcornAI Pro", layout="wide")

API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

# Advanced function to fetch poster, rating, synopsis, and OTT platform
def fetch_movie_details(movie_id):
    details = {
        "poster": "https://images.unsplash.com/photo-1536440136628-849c177e76a1?w=500",
        "rating": "N/A",
        "synopsis": "No synopsis available.",
        "ott": "Not streaming online"
    }
    
    # 1. Fetch Basic Details (Poster, Rating, Synopsis)
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    try:
        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            
            # Extract poster
            poster_path = data.get('poster_path')
            if poster_path:
                details["poster"] = f"https://image.tmdb.org/t/p/w500/{poster_path}"
            
            # Extract rating & synopsis
            if data.get('vote_average'):
                details["rating"] = f"⭐ {round(data['vote_average'], 1)}/10"
            if data.get('overview'):
                overview = data['overview']
                details["synopsis"] = overview[:120] + "..." if len(overview) > 120 else overview
                
    except Exception:
        pass

    # 2. Fetch OTT Availability (Filtering for India 'IN')
    watch_url = f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers?api_key={API_KEY}"
    try:
        watch_response = requests.get(watch_url, timeout=3)
        if watch_response.status_code == 200:
            watch_data = watch_response.json()
            results = watch_data.get('results', {})
            
            india_providers = results.get('IN', {})
            flatrate = india_providers.get('flatrate', [])
            
            if flatrate:
                # Extract up to the top 2 streaming platforms (e.g., Netflix, Hotstar)
                providers_list = [provider['provider_name'] for provider in flatrate[:2]]
                details["ott"] = "🎬 Stream on: " + ", ".join(providers_list)
            else:
                details["ott"] = "🎟️ Rent/Buy Only"
    except Exception:
        pass

    return details

# Load pre-calculated models (Cached)
@st.cache_resource
def load_models():
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return movies, similarity

try:
    movies, similarity = load_models()
except FileNotFoundError:
    st.error("Please ensure your pickle files are uploaded successfully.")
    st.stop()

# Recommendation logic
def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), key=lambda x: x[1], reverse=True)[1:6]
    
    recommended_details = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id
        title = movies.iloc[i[0]].title
        
        # Get all dynamic API details for this specific recommendation
        api_data = fetch_movie_details(movie_id)
        api_data['title'] = title
        
        recommended_details.append(api_data)
        
    return recommended_details

# UI Header
st.markdown("<h1 style='text-align: center; color: #E50914;'>🍿 PopcornAI Pro 🍿</h1>", unsafe_allow_html=True)
st.write("---")

selected_movie_name = st.selectbox(
    "Search or select a movie you loved:",
    movies['title'].values
)

if st.button('Get Premium Recommendations'):
    with st.spinner('Fetching streaming info and ratings...'):
        recommendations = recommend(selected_movie_name)
        cols = st.columns(5)
        
        for i in range(5):
            movie = recommendations[i]
            with cols[i]:
                # 1. Poster Image
                st.image(movie['poster'], use_container_width=True)
                
                # 2. Title (Bold)
                st.markdown(f"**{movie['title']}**")
                
                # 3. Rating & OTT Platform
                st.caption(f"{movie['rating']}")
                st.caption(f"_{movie['ott']}_")
                
                # 4. Interactive Dropdown Expander for Synopsis
                with st.expander("Read Synopsis"):
                    st.write(movie['synopsis'])