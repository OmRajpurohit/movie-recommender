from __future__ import annotations

import time

from recommender import CACHE_PATH, DEFAULT_DATA_PATH, META_PATH, MovieRecommender


def main() -> None:
    started_at = time.time()
    engine = MovieRecommender(force_rebuild=True)
    elapsed = time.time() - started_at
    print(f"Dataset used: {engine.csv_path if engine.csv_path.exists() else DEFAULT_DATA_PATH}")
    print(f"Movies loaded into cache: {len(engine.movies)}")
    print(f"Cache file: {CACHE_PATH}")
    print(f"Metadata file: {META_PATH}")
    print(f"Cache build complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
