from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
BACKDROP_BASE_URL = "https://image.tmdb.org/t/p/w780"
RAW_DATA_PATH = Path(__file__).with_name("movies1M.csv")
WORKING_SAMPLE_SIZE = 10_000

LANGUAGE_LABELS = {
    "ar": "Arabic",
    "cn": "Chinese",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ml": "Malayalam",
    "pt": "Portuguese",
    "ru": "Russian",
    "ta": "Tamil",
    "te": "Telugu",
    "tr": "Turkish",
    "zh": "Chinese",
}


@dataclass
class RecommendationResult:
    query: str
    resolved_title: str
    resolved_id: int
    message: str | None
    seed_movie: dict[str, Any]
    movies: list[dict[str, Any]]


class MovieRecommender:
    def __init__(self, csv_path: Path | str = RAW_DATA_PATH, sample_size: int = WORKING_SAMPLE_SIZE) -> None:
        self.csv_path = Path(csv_path)
        self.sample_size = sample_size
        self.movies = self._load_working_set()
        self.tfidf_matrix = self._build_tfidf_matrix()
        self.title_choices = self.movies["normalized_title"].tolist()
        self.search_choices = self.movies["search_title"].tolist()
        self.genre_options = self._build_genre_options()
        self.language_options = self._build_language_options()

    def _load_working_set(self) -> pd.DataFrame:
        use_columns = [
            "id",
            "title",
            "vote_average",
            "vote_count",
            "status",
            "release_date",
            "runtime",
            "backdrop_path",
            "original_language",
            "overview",
            "popularity",
            "poster_path",
            "tagline",
            "genres",
            "production_companies",
            "production_countries",
            "spoken_languages",
            "keywords",
        ]
        df = pd.read_csv(
            self.csv_path,
            usecols=use_columns,
            nrows=self.sample_size,
            low_memory=False,
        )

        text_columns = [
            "title",
            "overview",
            "tagline",
            "genres",
            "production_companies",
            "production_countries",
            "spoken_languages",
            "keywords",
            "original_language",
        ]
        for column in text_columns:
            df[column] = df[column].fillna("").astype(str)

        numeric_columns = ["vote_average", "vote_count", "popularity", "runtime"]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

        df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
        df["release_year"] = df["release_date"].dt.year.fillna(0).astype(int)
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
        df = df.dropna(subset=["id", "title"]).copy()
        df["id"] = df["id"].astype(int)

        df["genre_list"] = df["genres"].map(self._split_people_text)
        df["keyword_list"] = df["keywords"].map(self._split_people_text)
        df["spoken_language_list"] = df["spoken_languages"].map(self._split_people_text)
        df["genre_set"] = df["genre_list"].map(set)
        df["primary_language"] = df["original_language"].str.lower().str.strip().replace("", "unknown")
        df["language_label"] = df["primary_language"].map(self._language_label)
        df["normalized_title"] = df["title"].str.casefold().str.strip()
        df["search_title"] = (
            df["title"]
            .str.lower()
            .str.replace(r"[^a-z0-9\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        df["poster_url"] = df["poster_path"].map(self._image_url)
        df["backdrop_url"] = df["backdrop_path"].map(self._backdrop_url)
        df["genre_display"] = df["genre_list"].map(lambda items: ", ".join(items[:3]) if items else "Mixed")
        df["runtime_label"] = df["runtime"].map(lambda value: f"{int(value)} min" if value > 0 else "Runtime n/a")

        vote_average_mean = df["vote_average"].mean()
        vote_count_threshold = df["vote_count"].quantile(0.75)
        df["imdb_rating"] = df.apply(
            lambda row: self._weighted_rating(
                row["vote_average"],
                row["vote_count"],
                vote_average_mean,
                vote_count_threshold,
            ),
            axis=1,
        )

        df["popularity_score"] = self._minmax_scale(df["popularity"])
        df["imdb_score_norm"] = self._minmax_scale(df["imdb_rating"])
        df["vote_score_norm"] = self._minmax_scale(df["vote_count"])
        df["discovery_score"] = (
            0.55 * df["imdb_score_norm"]
            + 0.25 * df["popularity_score"]
            + 0.20 * df["vote_score_norm"]
        )

        df["content_soup"] = df.apply(self._build_content_soup, axis=1)
        df = df.drop_duplicates(subset=["normalized_title"], keep="first").reset_index(drop=True)
        return df

    def _build_tfidf_matrix(self):
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=14_000,
        )
        self.vectorizer = vectorizer
        return vectorizer.fit_transform(self.movies["content_soup"])

    def _build_genre_options(self) -> list[str]:
        genres = sorted({genre for values in self.movies["genre_list"] for genre in values})
        return ["all", *genres]

    def _build_language_options(self) -> list[tuple[str, str]]:
        languages = (
            self.movies[["primary_language", "language_label"]]
            .drop_duplicates()
            .sort_values("language_label")
            .itertuples(index=False, name=None)
        )
        return [("all", "All languages"), *languages]

    @staticmethod
    def _split_people_text(value: str) -> list[str]:
        return [item.strip() for item in value.split(",") if item and item.strip()]

    @staticmethod
    def _weighted_rating(rating: float, votes: float, baseline: float, threshold: float) -> float:
        if votes <= 0:
            return float(baseline)
        return float((votes / (votes + threshold)) * rating + (threshold / (votes + threshold)) * baseline)

    @staticmethod
    def _minmax_scale(series: pd.Series) -> pd.Series:
        minimum = series.min()
        maximum = series.max()
        if pd.isna(minimum) or pd.isna(maximum) or maximum == minimum:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - minimum) / (maximum - minimum)

    @staticmethod
    def _image_url(poster_path: str) -> str | None:
        if not poster_path or poster_path == "nan":
            return None
        return f"{IMAGE_BASE_URL}{poster_path}"

    @staticmethod
    def _backdrop_url(backdrop_path: str) -> str | None:
        if not backdrop_path or backdrop_path == "nan":
            return None
        return f"{BACKDROP_BASE_URL}{backdrop_path}"

    @staticmethod
    def _language_label(language_code: str) -> str:
        if not language_code or language_code == "unknown":
            return "Unknown"
        return LANGUAGE_LABELS.get(language_code, language_code.upper())

    def _build_content_soup(self, row: pd.Series) -> str:
        weighted_bits = []
        weighted_bits.extend(row["genre_list"] * 3)
        weighted_bits.extend(row["keyword_list"] * 2)
        weighted_bits.extend(self._split_people_text(row["production_companies"]))
        weighted_bits.extend(self._split_people_text(row["production_countries"]))
        weighted_bits.extend(self._split_people_text(row["spoken_languages"]))
        weighted_bits.extend([row["primary_language"]] * 2)
        weighted_bits.append(row["overview"])
        weighted_bits.append(row["tagline"])
        return " ".join(bit for bit in weighted_bits if bit)

    def _apply_filters(self, frame: pd.DataFrame, genre: str = "all", language: str = "all") -> pd.DataFrame:
        filtered = frame
        if genre != "all":
            filtered = filtered[filtered["genre_list"].map(lambda items: genre in items)]
        if language != "all":
            filtered = filtered[filtered["primary_language"] == language]
        return filtered

    def _serialize_movie(self, row: pd.Series) -> dict[str, Any]:
        return {
            "id": int(row["id"]),
            "title": row["title"],
            "year": int(row["release_year"]) if row["release_year"] else None,
            "poster_url": row["poster_url"],
            "backdrop_url": row["backdrop_url"],
            "genres": row["genre_display"],
            "language": row["language_label"],
            "imdb_rating": round(float(row["imdb_rating"]), 1),
            "vote_average": round(float(row["vote_average"]), 1),
            "overview": row["overview"] or "No overview available yet.",
            "runtime": row["runtime_label"],
        }

    def get_featured_movie(self) -> dict[str, Any]:
        row = self.movies.sort_values("discovery_score", ascending=False).iloc[0]
        return self._serialize_movie(row)

    def get_popular_movies(
        self,
        limit: int = 18,
        genre: str = "all",
        language: str = "all",
        exclude_id: int | None = None,
    ) -> list[dict[str, Any]]:
        filtered = self._apply_filters(self.movies, genre=genre, language=language)
        if exclude_id is not None:
            filtered = filtered[filtered["id"] != exclude_id]
        ranked = filtered.sort_values(["discovery_score", "imdb_rating"], ascending=False).head(limit)
        return [self._serialize_movie(row) for _, row in ranked.iterrows()]

    def resolve_title(self, query: str) -> tuple[pd.Series | None, str | None]:
        normalized_query = query.casefold().strip()
        if not normalized_query:
            return None, "Type a movie title to search."

        exact_match = self.movies[self.movies["normalized_title"] == normalized_query]
        if not exact_match.empty:
            return exact_match.iloc[0], None

        search_query = re.sub(r"[^a-z0-9\s]", " ", normalized_query)
        search_query = re.sub(r"\s+", " ", search_query).strip()
        fuzzy_match = process.extractOne(
            search_query,
            self.search_choices,
            scorer=fuzz.WRatio,
            score_cutoff=80,
        )
        if not fuzzy_match:
            return None, f'No close match found for "{query}". Try another title.'

        matched_movie = self.movies.iloc[fuzzy_match[2]]
        message = f'Showing results for "{matched_movie["title"]}" based on your search.'
        return matched_movie, message

    def recommend_by_title(
        self,
        query: str,
        top_n: int = 18,
        genre: str = "all",
        language: str = "all",
    ) -> tuple[RecommendationResult | None, str | None]:
        seed_movie, message = self.resolve_title(query)
        if seed_movie is None:
            return None, message

        return (
            self._build_recommendations(
                seed_movie=seed_movie,
                query=query,
                message=message,
                top_n=top_n,
                genre=genre,
                language=language,
            ),
            None,
        )

    def recommend_by_id(
        self,
        movie_id: int,
        top_n: int = 18,
        genre: str = "all",
        language: str = "all",
    ) -> RecommendationResult | None:
        match = self.movies[self.movies["id"] == movie_id]
        if match.empty:
            return None
        seed_movie = match.iloc[0]
        return self._build_recommendations(
            seed_movie=seed_movie,
            query=seed_movie["title"],
            message=None,
            top_n=top_n,
            genre=genre,
            language=language,
        )

    def _build_recommendations(
        self,
        seed_movie: pd.Series,
        query: str,
        message: str | None,
        top_n: int,
        genre: str,
        language: str,
    ) -> RecommendationResult:
        seed_index = self.movies.index[self.movies["id"] == int(seed_movie["id"])][0]
        cosine_scores = linear_kernel(self.tfidf_matrix[seed_index], self.tfidf_matrix).flatten()

        candidate_frame = self.movies.copy()
        candidate_frame["content_similarity"] = cosine_scores
        candidate_frame = candidate_frame[candidate_frame["id"] != seed_movie["id"]]
        candidate_frame = self._apply_filters(candidate_frame, genre=genre, language=language)

        if candidate_frame.empty:
            fallback_movies = self.get_popular_movies(limit=top_n, genre=genre, language=language, exclude_id=int(seed_movie["id"]))
            return RecommendationResult(
                query=query,
                resolved_title=seed_movie["title"],
                resolved_id=int(seed_movie["id"]),
                message="No close matches inside the selected filters, so these are the best popular alternatives.",
                seed_movie=self._serialize_movie(seed_movie),
                movies=fallback_movies,
            )

        seed_genres = seed_movie["genre_set"]
        candidate_frame["genre_overlap"] = candidate_frame["genre_set"].map(
            lambda genres: self._jaccard_similarity(seed_genres, genres)
        )
        candidate_frame["language_bonus"] = (
            candidate_frame["primary_language"] == seed_movie["primary_language"]
        ).astype(float)
        candidate_frame["hybrid_score"] = (
            0.62 * candidate_frame["content_similarity"]
            + 0.18 * candidate_frame["genre_overlap"]
            + 0.08 * candidate_frame["language_bonus"]
            + 0.07 * candidate_frame["imdb_score_norm"]
            + 0.05 * candidate_frame["popularity_score"]
        )

        ranked = candidate_frame.sort_values(
            ["hybrid_score", "imdb_rating", "vote_count"],
            ascending=False,
        ).head(top_n)

        if ranked.empty:
            ranked_movies = self.get_popular_movies(limit=top_n, genre=genre, language=language, exclude_id=int(seed_movie["id"]))
        else:
            ranked_movies = [self._serialize_movie(row) for _, row in ranked.iterrows()]

        return RecommendationResult(
            query=query,
            resolved_title=seed_movie["title"],
            resolved_id=int(seed_movie["id"]),
            message=message,
            seed_movie=self._serialize_movie(seed_movie),
            movies=ranked_movies,
        )

    @staticmethod
    def _jaccard_similarity(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)
