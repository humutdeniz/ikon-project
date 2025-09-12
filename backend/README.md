FastAPI clone of the existing Express backend.

Run locally

- Create and activate a virtualenv, then install deps:
  - Linux/macOS
    - `python3 -m venv .venv`  
    - `source .venv/bin/activate`
  - Windows (PowerShell)
    - `python -m venv .venv`
    - `.venv\\Scripts\\Activate.ps1`
  - Install requirements
    - `pip install -r backend/requirements.txt`
  - ML extras (for violence detection/train scripts)
    - `pip install -r backend/requirements-ml.txt`
- Start the API on port 5000 (from repo root):
  - `uvicorn backend.main:app --reload --port 5000`

The API exposes the same endpoints used by the frontend:

- `GET /api/users` — optional query: `name`, `status`
- `POST /api/users` — JSON: `{ name, status, password }`
- `GET /api/deliveries` — optional query: `recipient`, `company`, `status`
- `POST /api/deliveries` — JSON: `{ recipient, company, status? }`
- `POST /api/editDeliveries` — JSON: `{ id, recipient?, company?, status? }`
- `GET /api/meetings` — optional query: `host`, `guest`, `date`
- `POST /api/meetings` — JSON: `{ host, guest, date }`
- `POST /api/chat` — JSON: legacy `{ message, history }` or new structured `{ message, context }`.

Chat endpoint (new structured flow)

- Request: `{ "message": string, "context": object }`
- Response: `{ "reply": string, "history": Turn[], "context": object }`
- The `context` is a structured state object updated each turn. It contains
  role, fields (e.g., `employee_name`, `password`, `courier_company`, etc.),
  and verification flags. The backend performs DB checks and decides actions
  deterministically. The LLM is used to fill the context via function calling
  (no separate NLP heuristics).

Notes

- Uses SQLite file at `backend/ai-concierge.db` and initializes schema on startup.
- Passwords are hashed via `bcrypt` (provided by `passlib[bcrypt]`).

Quick seed (optional)

Run these in a separate terminal while the server is running to create a few rows:

```
curl -s http://localhost:5000/api/users -H 'Content-Type: application/json' \
  -d '{"name":"Umut Deniz","status":"employee","password":"1234"}'

curl -s http://localhost:5000/api/deliveries -H 'Content-Type: application/json' \
  -d '{"recipient":"Umut Deniz","company":"Aras Kargo","status":"pending"}'

curl -s http://localhost:5000/api/meetings -H 'Content-Type: application/json' \
  -d '{"host":"Arda Alper","guest":"Mustafa Alkan","date":"2024-09-01T16:00:00"}'
```

Frontend config

- The agent expects the backend at `http://localhost:5000/api`.
- Ensure `ikon-project/src/utility.py` base URL (via env `IKON_API_BASE_URL`) points there if you change the port.
