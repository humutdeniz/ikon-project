from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .db import init_db
from .routers import users, deliveries, meetings, chat, speech, tts


def create_app() -> FastAPI:
    init_db()
    app = FastAPI(title="AI Concierge API (FastAPI)")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers with the same base path as Express
    app.include_router(users.router, prefix="/api", tags=["users"])
    app.include_router(deliveries.router, prefix="/api", tags=["deliveries"])
    app.include_router(meetings.router, prefix="/api", tags=["meetings"])
    app.include_router(chat.router, prefix="/api", tags=["chat"])
    app.include_router(speech.router, prefix="/api", tags=["speech"])
    app.include_router(tts.router, prefix="/api", tags=["tts"])

    # Serve static frontend (chat page as main entry)
    static_dir = Path(__file__).resolve().parents[1] / "frontend"
    if static_dir.exists():
        # Mount at root; /api/* remains available due to more specific matching
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


app = create_app()
