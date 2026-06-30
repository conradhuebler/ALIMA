"""FastAPI APIRouter modules for the ALIMA webapp (F-6 god-file split).

Each module exposes a ``router = APIRouter()`` that ``app.py`` mounts via
``app.include_router(...)``. Routers depend on ``session_state`` / ``render_bridge``
and never import ``app`` (keeps the import DAG acyclic). - Claude Generated
"""
