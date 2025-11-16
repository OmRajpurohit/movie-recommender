import streamlit as st
import pandas as pd
from recommender import hybrid_recommend, get_popular_movies

# Page config
st.set_page_config(
    page_title="Movi e Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")

# Tabs
tab1, tab2 = st.tabs(["🔍 Search Movies", "🔥 Popular Movies"])

# ---------------------- SEARCH TAB ----------------------
with tab1:
    st.subheader("Search Similar Movies")

    query = st.text_input("Enter a movie title")

    if query:
        with st.spinner("Searching..."):
            movies, msg = hybrid_recommend(query, top_n=10)

        if msg:
            st.warning(msg)

        if movies is None:
            st.error("No similar movies found.")
        else:
            st.write("### Recommended Movies:")
            for i, row in movies.iterrows():
                st.write(f"**{row['title']}** — ⭐ {row['imdb_score']}")

# ---------------------- POPULAR MOVIES TAB ----------------------
with tab2:
    st.subheader("🔥 Top 20 Popular Movies")

    popular = get_popular_movies(20)

    cols = st.columns(2)
for idx, (i, row) in enumerate(movies.iterrows()):
    with cols[idx % 2]:
        st.markdown(f"**{row['title']}**")
        st.write(f"⭐ {row['imdb_score']}")
        st.markdown("---")

