from __future__ import annotations

from functools import lru_cache

from flask import Flask, render_template, request

from recommender import MovieRecommender


app = Flask(__name__)


@lru_cache(maxsize=1)
def get_recommender() -> MovieRecommender:
    return MovieRecommender()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.route("/", methods=["GET", "POST"])
def home():
    engine = get_recommender()

    search_query = (
        request.values.get("title", "") if request.method == "GET" else request.form.get("title", "")
    ).strip()
    selected_movie_id = request.values.get("movie_id", type=int)
    selected_genre = request.values.get("genre", "all").strip() or "all"
    selected_language = request.values.get("language", "all").strip() or "all"
    user_message = None

    featured_movies = engine.get_popular_movies(
        limit=18,
        genre=selected_genre,
        language=selected_language,
    )
    hero_movie = featured_movies[0] if featured_movies else engine.get_featured_movie()

    recommendation_result = None
    if selected_movie_id:
        recommendation_result = engine.recommend_by_id(
            selected_movie_id,
            top_n=18,
            genre=selected_genre,
            language=selected_language,
        )
    elif search_query:
        recommendation_result, user_message = engine.recommend_by_title(
            search_query,
            top_n=18,
            genre=selected_genre,
            language=selected_language,
        )

    return render_template(
        "index.html",
        hero_movie=hero_movie,
        featured_movies=featured_movies,
        recommendation_result=recommendation_result,
        user_message=user_message,
        search_query=search_query,
        selected_genre=selected_genre,
        selected_language=selected_language,
        genre_options=engine.genre_options,
        language_options=engine.language_options,
    )


if __name__ == "__main__":
    app.run(debug=True)
