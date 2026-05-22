# Cinema Atlas

Cinema Atlas is a split deployment movie recommender:

- Backend API: Flask on Render
- Frontend: static site on Vercel
- Dataset hosting: Google Drive

## Deployment Architecture

- Render runs the Flask API in `app.py`
- Vercel serves the static frontend from `frontend/`
- The dataset is downloaded from Google Drive by `download_dataset.py`
- `build_index.py` builds a warm recommender cache so Render startup stays fast

## Backend API

Available endpoints:

- `GET /health`
- `GET /api/filters`
- `GET /api/featured`
- `GET /api/popular?genre=all&language=all&limit=18`
- `GET /api/suggestions?q=inter`
- `GET /api/recommendations?title=inception`
- `GET /api/recommendations?movie_id=27205`

## Google Drive Dataset Hosting

The app supports downloading the dataset from Google Drive during the Render build step.

Use either:

- `GOOGLE_DRIVE_FILE_ID`
- `GOOGLE_DRIVE_URL`

Google Drive setup:

1. Upload `movies1M.csv` to Google Drive.
2. Open the file share settings and allow link access for the Render build to download it.
3. Copy either the file id or the share URL.
4. Add that value to Render as `GOOGLE_DRIVE_FILE_ID` or `GOOGLE_DRIVE_URL`.

Render also uses:

- `MOVIES_DATA_PATH`
- `WORKING_SAMPLE_SIZE`
- `FRONTEND_ORIGIN`

`WORKING_SAMPLE_SIZE=0` means "use the full dataset". For local development or small Render instances, keep this set to a smaller value such as `10000`.

## Render Deployment

According to Render’s Flask deployment docs, a Python web service typically uses:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`

This repo includes `render.yaml` with the backend deployment setup. The build command downloads the dataset from Google Drive and builds the recommender cache before startup.

Recommended Render environment variables:

- `PYTHON_VERSION=3.12.10`
- `MOVIES_DATA_PATH=data/movies1M.csv`
- `WORKING_SAMPLE_SIZE=10000`
- `GOOGLE_DRIVE_FILE_ID=<your-file-id>` or `GOOGLE_DRIVE_URL=<your-share-link>`
- `FRONTEND_ORIGIN=https://<your-vercel-domain>`

Recommended deployment notes:

1. Use Render Blueprint import so `render.yaml` is applied automatically.
2. Start with `WORKING_SAMPLE_SIZE=10000` on Render Free for a fast and stable build.
3. Move to `WORKING_SAMPLE_SIZE=0` only when you are ready to run the full dataset on a larger Render instance.
4. Upload the CSV to Google Drive as a direct-share file and set either `GOOGLE_DRIVE_FILE_ID` or `GOOGLE_DRIVE_URL`.

## Vercel Deployment

The frontend lives in `frontend/`. In Vercel:

1. Import the repository
2. Set the project Root Directory to `frontend`
3. Set the Build Command to `npm run build`
4. Set the environment variable `CINEMA_ATLAS_API_BASE_URL=https://<your-render-backend>.onrender.com`
5. Use the included `frontend/vercel.json`

The Vercel build step rewrites `frontend/config.js` automatically, so you do not need to manually edit the deployed frontend for each environment.

## Local Run

```bash
pip install -r requirements.txt
python download_dataset.py
python build_index.py
python app.py
```

Then open:

- Backend: `http://127.0.0.1:5000`
- Frontend: run `cd frontend && npm run build`, then open `frontend/index.html` or serve the `frontend` folder locally

## Official Docs Used

- Render Flask deployment: [render.com/docs/deploy-flask](https://render.com/docs/deploy-flask)
- Render Python version config: [render.com/docs/python-version](https://render.com/docs/python-version)
- Render Blueprint spec: [render.com/docs/blueprint-spec](https://render.com/docs/blueprint-spec)
- Vercel build config: [vercel.com/docs/builds/configure-a-build](https://vercel.com/docs/builds/configure-a-build)
- Vercel rewrites / reverse proxy guide: [examples.vercel.com/guides/vercel-reverse-proxy-rewrites-external](https://examples.vercel.com/guides/vercel-reverse-proxy-rewrites-external)
