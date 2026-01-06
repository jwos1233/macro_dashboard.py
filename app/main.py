"""
Epoch Macro Dashboard - FastAPI Application
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup: Pre-warm caches from disk
    print("=" * 60, flush=True)
    print("EPOCH MACRO STARTING UP", flush=True)
    print("=" * 60, flush=True)

    # Pre-load signals from disk cache
    try:
        from app.routes.dashboard import load_signals_from_disk
        signals = load_signals_from_disk()
        if signals:
            print("✓ Signals loaded from disk cache", flush=True)
        else:
            print("○ No signal cache found (will generate on first request)", flush=True)
    except Exception as e:
        print(f"○ Could not pre-load signals: {e}", flush=True)

    # Pre-load backtest from disk cache
    try:
        from app.data import load_backtest_results
        backtest = load_backtest_results()
        source = backtest.get('data_source', 'unknown')
        print(f"✓ Backtest loaded from {source}", flush=True)
    except Exception as e:
        print(f"○ Could not pre-load backtest: {e}", flush=True)

    print("=" * 60, flush=True)
    print("STARTUP COMPLETE - Ready for requests", flush=True)
    print("=" * 60, flush=True)

    yield  # App runs here

    # Shutdown
    print("Epoch Macro shutting down...", flush=True)


from app.routes import dashboard, api

# Initialize FastAPI app
app = FastAPI(
    title="Epoch Macro",
    description="Quantitative Macro Allocation Dashboard",
    version="1.0.0",
    lifespan=lifespan
)

# Get the app directory
APP_DIR = Path(__file__).parent

# Mount static files (only if directory exists)
static_dir = APP_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setup templates
templates = Jinja2Templates(directory=APP_DIR / "templates")

# Include routers
app.include_router(dashboard.router)
app.include_router(api.router, prefix="/api")


@app.get("/")
async def root(request: Request):
    """Redirect to dashboard"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
