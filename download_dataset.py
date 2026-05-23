from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
from urllib.parse import parse_qs, urlparse

import gdown


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TARGET = PROJECT_ROOT / "data" / "movies1M.csv"
LOCAL_FALLBACK = PROJECT_ROOT / "movies1M.csv"


def extract_google_drive_file_id(url: str) -> str | None:
    parsed = urlparse(url)
    query_file_id = parse_qs(parsed.query).get("id")
    if query_file_id:
        return query_file_id[0]

    match = re.search(r"/file/d/([^/]+)", parsed.path)
    if match:
        return match.group(1)

    return None


def main() -> None:
    target_path = Path(os.getenv("MOVIES_DATA_PATH", DEFAULT_TARGET))
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        print(f"Dataset already present at {target_path}")
        return

    if LOCAL_FALLBACK.exists() and LOCAL_FALLBACK.resolve() != target_path.resolve():
        shutil.copy2(LOCAL_FALLBACK, target_path)
        print(f"Copied local dataset to {target_path}")
        return

    google_drive_file_id = os.getenv("GOOGLE_DRIVE_FILE_ID", "").strip()
    google_drive_url = os.getenv("GOOGLE_DRIVE_URL", "").strip()

    if google_drive_file_id:
        gdown.download(id=google_drive_file_id, output=str(target_path), quiet=False)
        print(f"Downloaded dataset from Google Drive file id to {target_path}")
        return

    if google_drive_url:
        extracted_file_id = extract_google_drive_file_id(google_drive_url)
        if extracted_file_id:
            gdown.download(id=extracted_file_id, output=str(target_path), quiet=False)
            print(f"Downloaded dataset from Google Drive URL via extracted file id to {target_path}")
            return

        gdown.download(url=google_drive_url, output=str(target_path), quiet=False)
        print(f"Downloaded dataset from Google Drive URL to {target_path}")
        return

    raise RuntimeError(
        "Dataset not found locally and no Google Drive source configured. "
        "Set GOOGLE_DRIVE_FILE_ID or GOOGLE_DRIVE_URL."
    )


if __name__ == "__main__":
    main()
