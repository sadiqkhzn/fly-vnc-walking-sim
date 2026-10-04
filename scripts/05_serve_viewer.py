"""Dev helper: run the FastAPI server and open the Three.js viewer.

Serve `viewer/` via any static file server (python -m http.server works) in
a separate terminal, or extend FastAPI to mount it. Kept separate so the
sim backend stays decoupled from the UI.
"""
from __future__ import annotations

from src.server.main import serve

if __name__ == "__main__":
    serve()
