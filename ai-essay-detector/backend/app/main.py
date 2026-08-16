"""FastAPI application entry point.

Both models are loaded exactly once here, in the lifespan handler, and stay
resident for the life of the process. Nothing in the request path ever calls a
loader.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import BACKEND_ROOT, get_settings
from app.models.schemas import ErrorResponse, HealthResponse, RootResponse
from app.routers import analyze as analyze_router
from app.services import classifier, model_loader

settings = get_settings()

# <repo>/ai-essay-detector/backend -> .../frontend/public/favicon.svg. Resolved
# once at import: the file either exists in this checkout or it never will.
_favicon = BACKEND_ROOT.parent / "frontend" / "public" / "favicon.svg"
FAVICON_PATH = _favicon if _favicon.is_file() else None

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("app")

# Third-party libraries are extremely chatty at INFO (every HuggingFace HTTP
# HEAD, every httpx request). Keep our own logs readable.
for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load GPT-2 and the trained classifier once, at boot.

    A failure here is logged but does not abort startup: the process stays up
    so ``/health`` can report ``model_loaded: false`` and ``/analyze`` can
    answer 503. A crash loop would tell an operator far less.
    """
    logger.info("Starting %s v%s", settings.api_title, settings.api_version)

    try:
        model_loader.load_gpt2(settings)
    except Exception:
        logger.exception("Failed to load GPT-2; /analyze will return 503.")

    try:
        classifier.load_classifier(settings)
    except Exception:
        logger.exception("Failed to load the classifier; /analyze will return 503.")

    if models_ready():
        logger.info("Startup complete; models resident and ready.")
    else:
        logger.error("Startup completed DEGRADED; see the errors above.")

    yield

    logger.info("Shutting down.")


def models_ready() -> bool:
    """True only when both GPT-2 and the classifier actually loaded."""
    return model_loader.is_loaded() and classifier.is_loaded()


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan,
)


# --- Error handling -------------------------------------------------------
# Registered as middleware rather than only as an app exception handler so the
# JSON body is produced *inside* the CORS layer. An exception that escapes to
# Starlette's outer error middleware would produce a response with no
# Access-Control-Allow-Origin header, and the browser would show the frontend a
# opaque CORS failure instead of the real 500.
@app.middleware("http")
async def catch_unhandled_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(error="Internal server error").model_dump(),
        )


# CORS is added last so it wraps the handler above and stamps its headers onto
# error responses too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Backstop for anything raised outside the middleware chain."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(error="Internal server error").model_dump(),
    )


# Registered against Starlette's HTTPException, not FastAPI's subclass: handler
# lookup walks the raised type's MRO, so registering the subclass would miss the
# framework's own 404s and 405s and leak the default {"detail": ...} shape.
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Keep deliberate errors in the same ``{"error": ...}`` envelope."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=str(exc.detail)).model_dump(),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422s in the same envelope, with a message a UI can display as-is."""
    problems = "; ".join(
        f"{'.'.join(str(p) for p in err['loc'][1:]) or 'body'}: {err['msg']}"
        for err in exc.errors()
    )
    # Literal 422: the constant was renamed between Starlette versions.
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(error=problems or "Invalid request.").model_dump(),
    )


# --- Routes ---------------------------------------------------------------
@app.get("/", response_model=RootResponse, tags=["ops"])
async def root() -> RootResponse:
    """Service index.

    This is an API, not a site, but the root is the first thing anyone types
    into a browser -- answering with the version, readiness and the endpoint
    map is more useful than the 404 a missing route would produce.
    """
    ready = models_ready()
    return RootResponse(
        service=settings.api_title,
        version=settings.api_version,
        status="ok" if ready else "degraded",
        model_loaded=ready,
        docs="/docs",
        endpoints={
            "health": "GET /health",
            "analyze": "POST /analyze",
            "docs": "GET /docs",
            "openapi": "GET /openapi.json",
        },
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    """Browsers request this unprompted on every root visit.

    The frontend ships the real icon; the API serves it too when that build is
    checked out beside the backend, and answers 204 otherwise so an unrelated
    request never shows up as an error in the log.
    """
    if FAVICON_PATH is not None:
        return FileResponse(FAVICON_PATH, media_type="image/svg+xml")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness plus a real readiness signal.

    ``model_loaded`` reflects both singletons actually being resident, not
    merely that this process is answering.
    """
    return HealthResponse(status="ok", model_loaded=models_ready())


app.include_router(analyze_router.router)
