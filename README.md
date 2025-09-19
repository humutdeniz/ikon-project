# IKON AI Concierge

Conversational concierge platform for offices. It combines a FastAPI backend, a React dashboard, and on-device speech/TTS pipelines to handle visitors, deliveries, and security workflows in Turkish.

## Features
- Visitor assistant backed by an LLM with tool-calling to verify employees, track deliveries, manage meetings, and escalate incidents
- FastAPI REST API with SQLite persistence and matching admin UI for managing users, deliveries, and meetings
- Real-time chat UI with optional streaming transcription and synthesized voice replies (Piper + FFmpeg)
- Optional computer-vision module for violence detection demos and CLI helpers for Whisper transcription/voice synthesis

## Project Layout
```
backend/           FastAPI app, routers, SQLite helpers, LLM agent
frontend/          React app (chat interface + admin panel)
src/               Stand-alone audio utilities (Whisper demo, Piper sample)
requirements.txt   Audio/TTS dependencies shared by demos and backend
README.md          This file
```

## Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- FFmpeg available on PATH (`ffmpeg` command)
- [Piper](https://github.com/rhasspy/piper) CLI installed (`piper` command). Turkish voices are already checked in under `backend/voices/`.
- (Optional) GPU-enabled PyTorch for Whisper and vision demos

## Backend Setup
1. Create a virtual environment and install dependencies:
   ```bash
   cd ikon-project
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r backend/reqs.txt
   pip install -r requirements.txt
   ```
   - `backend/reqs.txt` covers FastAPI + database + LLM agent dependencies.
   - Root `requirements.txt` adds Piper/Whisper audio extras used by the TTS and streaming endpoints.
2. Configure environment variables (copy `.env` or create one):
   - `OPENAI_BASE_URL` – base URL for the LLM API (defaults to `http://localhost:11434/v1`)
   - `OPENAI_API_KEY` – API key for the model server (defaults to `not-needed` for local deployments)
   - `MODEL_ID` – model name, e.g. `gpt-3.5-turbo` or an Ollama model like `gpt-oss:20b`
   - `WHISPER_MODEL` / `WHISPER_DEVICE` – override for speech recognizer model & device
   - `DB_PATH` – optional path to the SQLite file (defaults to `backend/ai-concierge.db`)
3. Initialize the database and start the API:
   ```bash
   uvicorn backend.main:app --reload --port 5000
   ```
   The API is now reachable at `http://localhost:5000/api`.

### Useful Backend Notes
- `backend/ai-concierge.db` is created automatically with tables for users, deliveries, and meetings.
- Sample seed commands (with the server running):
  ```bash
  curl -s http://localhost:5000/api/users -H 'Content-Type: application/json' \
    -d '{"name":"Umut Deniz","status":"employee","password":"1234"}'
  curl -s http://localhost:5000/api/deliveries -H 'Content-Type: application/json' \
    -d '{"recipient":"Umut Deniz","company":"Aras Kargo","status":"pending"}'
  curl -s http://localhost:5000/api/meetings -H 'Content-Type: application/json' \
    -d '{"host":"Arda Alper","guest":"Mustafa Alkan","date":"2024-09-01T16:00:00"}'
  ```
- TTS streaming (`GET /api/tts?text=...`) requires the `piper` and `ffmpeg` executables to be available.
- Real-time speech recognition (`POST /api/speech/stream`) buffers microphone audio identified by the `X-Session-Id` header and transcribes with Whisper.

## Frontend Setup
1. Install dependencies and start the dev server:
   ```bash
   cd frontend
   npm install
   npm start
   ```
   The app runs at `http://localhost:3000` and proxies API calls to `http://localhost:5000/api`.
2. Configure the API base URL if you change ports or deploy remotely:
   - Set `REACT_APP_API_BASE_URL` in a `.env` file under `frontend/`.
   - Otherwise the app uses `http://localhost:5000/api` in development and `window.location.origin/api` in production.
3. Build static assets for production:
   ```bash
   npm run build
   ```
   Serve the `frontend/build` output with your preferred static host. To let FastAPI serve the bundle directly, copy the build artifacts into `frontend/` or adjust `backend/main.py` to mount the build directory.

### UI Overview
- `/` – Concierge chat interface with message history, microphone capture, and voice playback.
- `/admin` – Admin dashboard shell with quick links.
- `/admin/users`, `/admin/meetings`, `/admin/deliveries` – CRUD views backed by the FastAPI endpoints.

## Audio Utilities (`src/`)
- `src/transcribe_demo.py` – Stand-alone Whisper streaming demo; run with `python src/transcribe_demo.py --model small`.
- `src/piperTest.py` – Generates `sample.wav` using Piper; ensure `piper` CLI and models under `src/voices/` are available.

## Computer Vision Demo
- `backend/violenceDetection/` hosts training and inference utilities for a violence detection model.
- Quick temporal inference demo:
  ```bash
  cd backend/violenceDetection
  python infer.py --model models/violence_efficientnet_b3_best.pt
  ```
  Pass `--video` to analyze a file or keep defaults for webcam input.
- The repository root `test.py` offers a richer HUD overlay; invoke it with explicit paths, for example:
  ```bash
  python test.py --meta backend/violenceDetection/models/meta.json \
    --model backend/violenceDetection/models/violence_efficientnet_b3_best.pt
  ```
  Requires OpenCV, TorchVision, and a webcam or video source.

## Troubleshooting
- **Missing Piper/FFmpeg**: Install via your package manager (`sudo apt install ffmpeg`, follow Piper docs) and ensure both commands are on PATH.
- **Whisper GPU errors**: Set `WHISPER_DEVICE=cpu` or install a CUDA-enabled PyTorch build matching your drivers.
- **Database locked**: SQLite WAL mode is enabled; if you see locks, ensure only one backend instance writes to the file.
- **CORS/404 from frontend**: Confirm backend runs on `:5000` and `API_BASE_URL` matches.
