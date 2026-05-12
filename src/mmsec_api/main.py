from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mmsec_api.routes.bootstrap import router as bootstrap_router
from mmsec_api.routes.datasets import router as datasets_router
from mmsec_api.routes.docs import router as docs_router
from mmsec_api.routes.health import router as health_router
from mmsec_api.routes.jobs import router as jobs_router
from mmsec_api.routes.models import router as models_router
from mmsec_api.routes.reports import router as reports_router
from mmsec_api.routes.runs import router as runs_router
from mmsec_api.routes.samples import router as samples_router
from mmsec_api.routes.system import router as system_router
from mmsec_api.runtime import ensure_app_runtime


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the in-process queue/bootstrap once for the whole API app.
    ensure_app_runtime(app, start_queue=True, start_bootstrap=True)
    yield
    q = getattr(app.state, "job_queue", None)
    if q is not None:
        q.stop()


app = FastAPI(title="ATT-project API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health_router)
app.include_router(bootstrap_router)
app.include_router(datasets_router)
app.include_router(jobs_router)
app.include_router(models_router)
app.include_router(runs_router)
app.include_router(samples_router)
app.include_router(reports_router)
app.include_router(docs_router)
app.include_router(system_router)

_dist_dir = Path(__file__).resolve().parents[2] / "frontend" / "dist"
_assets_dir = _dist_dir / "assets"
_index_file = _dist_dir / "index.html"

if _assets_dir.exists():
    # When frontend/dist exists, FastAPI serves the built SPA directly.
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="frontend-assets")


@app.middleware("http")
async def frontend_cache_control(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path
    content_type = response.headers.get("content-type", "")

    if path.startswith("/api/"):
        return response

    if path == "/" or path.endswith(".html") or content_type.startswith("text/html"):
        # Force the shell document to revalidate so browser tabs do not keep an
        # older SPA bundle after deployment.
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    elif path.startswith("/assets/"):
        # Built assets are content-hashed, so they can be cached aggressively.
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"

    return response


@app.api_route("/api/{full_path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"], include_in_schema=False)
def api_not_found(full_path: str):
    raise HTTPException(status_code=404, detail=f"API route not found: /api/{full_path}")


@app.get("/", include_in_schema=False)
def frontend_index():
    if _index_file.exists():
        return FileResponse(_index_file)
    return {"status": "frontend_not_built"}


@app.get("/{full_path:path}", include_in_schema=False)
def frontend_spa_fallback(full_path: str):
    # Vite builds a client-side routed SPA, so unknown paths should fall back
    # to index.html instead of returning a 404 from the backend.
    candidate = (_dist_dir / full_path).resolve()
    try:
        candidate.relative_to(_dist_dir.resolve())
    except (OSError, ValueError):
        candidate = _index_file

    if candidate.exists() and candidate.is_file():
        return FileResponse(candidate)
    if _index_file.exists():
        return FileResponse(_index_file)
    return {"status": "frontend_not_built", "path": full_path}
