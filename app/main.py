"""
Epoch Macro Dashboard - FastAPI Application
"""

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.routes import dashboard, api

# Initialize FastAPI app
app = FastAPI(
    title="Epoch Macro",
    description="Quantitative Macro Allocation Dashboard",
    version="1.0.0"
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
