from __future__ import annotations

from functools import lru_cache
import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from recommender import MovieRecommender


app = Flask(__name__)

frontend_origin = os.getenv("FRONTEND_ORIGIN", "*").strip() or "*"
CORS(
    app,
    resources={r"/api/*": {"origins": frontend_origin}},
)


@lru_cache(maxsize=1)
def get_recommender() -> MovieRecommender:
    return MovieRecommender()


def parse_common_filters() -> tuple[str, str, int]:
    genre = request.args.get("genre", "all").strip() or "all"
    language = request.args.get("language", "all").strip() or "all"
    limit = request.args.get("limit", default=18, type=int)
    return genre, language, max(1, min(limit, 50))


@app.get("/")
def index():
    return jsonify(
        {
            "service": "Cinema Atlas API",
            "status": "ok",
            "message": "Frontend is intended to be hosted on Vercel. Use the /api endpoints from the static frontend.",
            "endpoints": [
                "/health",
                "/api/filters",
                "/api/featured",
                "/api/popular",
                "/api/suggestions?q=inter",
                "/api/recommendations?title=inception",
                "/api/recommendations?movie_id=27205",
            ],
        }
    )


@app.get("/health")
def health():
    engine = get_recommender()
    return {
        "status": "ok",
        "movies_loaded": len(engine.movies),
        "data_path": str(engine.csv_path),
        "cache_path": str(engine.cache_path),
    }


@app.get("/api/filters")
def filters():
    engine = get_recommender()
    return jsonify(
        {
            "genres": engine.genre_options,
            "languages": [{"code": code, "label": label} for code, label in engine.language_options],
        }
    )


@app.get("/api/featured")
def featured():
    engine = get_recommender()
    return jsonify({"movie": engine.get_featured_movie()})


@app.get("/api/popular")
def popular():
    engine = get_recommender()
    genre, language, limit = parse_common_filters()
    movies = engine.get_popular_movies(limit=limit, genre=genre, language=language)
    return jsonify({"movies": movies})


@app.get("/api/suggestions")
def suggestions():
    engine = get_recommender()
    query = request.args.get("q", "").strip()
    return jsonify({"suggestions": engine.get_search_suggestions(query)})


@app.get("/api/recommendations")
def recommendations():
    engine = get_recommender()
    genre, language, limit = parse_common_filters()
    movie_id = request.args.get("movie_id", type=int)
    title = request.args.get("title", "").strip()

    if movie_id:
        result = engine.recommend_by_id(movie_id, top_n=limit, genre=genre, language=language)
        if result is None:
            return jsonify({"error": "Movie not found."}), 404
        return jsonify({"result": engine.serialize_recommendation_result(result)})

    if title:
        result, user_message = engine.recommend_by_title(title, top_n=limit, genre=genre, language=language)
        if result is None:
            return jsonify({"error": user_message or "No close title match found."}), 404
        return jsonify(
            {
                "result": engine.serialize_recommendation_result(result),
                "user_message": user_message,
            }
        )

    return jsonify({"error": "Provide either title or movie_id."}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=True)
