import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.cluster import KMeans

# Load and clean the dataset
df = pd.read_csv("movies_metadata.csv", low_memory=False)
df = df[['title', 'genres', 'vote_average', 'popularity']].dropna()

# Parse genres from string to list
def parse_genres(genres_str):
    try:
        genres = ast.literal_eval(genres_str)
        return [g['name'] for g in genres]
    except:
        return []

df['genres'] = df['genres'].apply(parse_genres)

# Remove rows with empty genre list
df = df[df['genres'].map(len) > 0]

# Vectorize genres using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(df['genres'])

# Combine genres with vote_average and popularity
features = pd.concat([
    pd.DataFrame(genre_matrix, columns=mlb.classes_),
    df[['vote_average', 'popularity']].reset_index(drop=True)
], axis=1)

# Scale features
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(features_scaled)

# Streamlit App
st.title("Movie Recommender (by Genre & Ratings)")
st.write("Get movie suggestions using unsupervised learning based on genres, ratings, and popularity.")

# User Input
movie = st.selectbox("Select a movie:", df['title'].values)

if st.button("Recommend Similar Movies"):
    selected_cluster = df[df['title'] == movie]['cluster'].values[0]
    recommended = df[(df['cluster'] == selected_cluster) & (df['title'] != movie)].head(10)

    st.subheader("You may also like:")
    for title in recommended['title'].values:
        st.write("•", title)
