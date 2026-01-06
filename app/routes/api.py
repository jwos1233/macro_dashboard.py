"""
API routes for Epoch Macro Dashboard
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

router = APIRouter(tags=["api"])


class SignalResponse(BaseModel):
    top_quadrants: tuple
    quadrant_scores: Dict[str, float]
    target_weights: Dict[str, float]
    current_regime: str
    timestamp: str
    total_leverage: float


class HealthResponse(BaseModel):
    status: str
    timestamp: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )


@router.get("/signals")
async def get_signals():
    """Get current trading signals"""
    from app.routes.dashboard import get_signals as fetch_signals

    signals = fetch_signals()

    return {
        "top_quadrants": signals['top_quadrants'],
        "quadrant_scores": {k: float(v) for k, v in signals['quadrant_scores'].items()},
        "target_weights": signals['target_weights'],
        "current_regime": signals['current_regime'],
        "timestamp": signals['timestamp'].isoformat(),
        "total_leverage": signals['total_leverage'],
    }


@router.get("/allocation")
async def get_allocation():
    """Get current portfolio allocation"""
    from app.routes.dashboard import get_signals as fetch_signals, calculate_asset_class_breakdown

    signals = fetch_signals()
    breakdown = calculate_asset_class_breakdown(signals['target_weights'])

    return {
        "weights": signals['target_weights'],
        "asset_class_breakdown": breakdown,
        "total_leverage": signals['total_leverage'],
        "timestamp": signals['timestamp'].isoformat(),
    }


@router.get("/regime")
async def get_regime():
    """Get current regime information"""
    from app.routes.dashboard import get_signals as fetch_signals
    from config import QUADRANT_DESCRIPTIONS

    signals = fetch_signals()

    return {
        "current_regime": signals['current_regime'],
        "primary_quadrant": signals['top_quadrants'][0],
        "secondary_quadrant": signals['top_quadrants'][1],
        "quadrant_scores": {k: float(v) for k, v in signals['quadrant_scores'].items()},
        "descriptions": QUADRANT_DESCRIPTIONS,
        "timestamp": signals['timestamp'].isoformat(),
    }


@router.post("/refresh")
async def refresh_signals():
    """Force refresh of signals (clears cache)"""
    from app.routes import dashboard

    dashboard._signal_cache = None
    dashboard._cache_time = None

    signals = dashboard.get_signals()

    return {
        "status": "refreshed",
        "timestamp": signals['timestamp'].isoformat(),
        "regime": signals['current_regime'],
    }


@router.get("/backtest/status")
async def backtest_status():
    """Get backtest data status and debug info"""
    from app.data import load_backtest_results, _backtest_cache, _backtest_cache_time, get_last_error
    import sys

    # Check which dependencies are available
    deps = {}
    for mod in ['numpy', 'pandas', 'yfinance']:
        try:
            __import__(mod)
            deps[mod] = True
        except ImportError:
            deps[mod] = False

    # Load current backtest data
    backtest = load_backtest_results()

    return {
        "data_source": backtest.get('data_source', 'unknown'),
        "generated_at": backtest.get('generated_at'),
        "summary": {
            "total_return": backtest.get('summary', {}).get('total_return', 0),
            "start_date": backtest.get('summary', {}).get('start_date', ''),
            "end_date": backtest.get('summary', {}).get('end_date', ''),
        },
        "dependencies_available": deps,
        "cache_time": _backtest_cache_time.isoformat() if _backtest_cache_time else None,
        "python_version": sys.version,
        "last_error": get_last_error(),
    }


@router.post("/backtest/refresh")
async def refresh_backtest():
    """Force refresh of backtest data"""
    from app.data import reload_backtest_results, get_last_error

    backtest = reload_backtest_results()

    return {
        "status": "refreshed",
        "data_source": backtest.get('data_source', 'unknown'),
        "generated_at": backtest.get('generated_at'),
        "total_return": backtest.get('summary', {}).get('total_return', 0),
        "last_error": get_last_error(),
    }


@router.post("/notes/generate")
async def generate_daily_note():
    """Generate a new daily note using AI"""
    from app.data.notes import create_daily_note, get_todays_note
    from app.routes.dashboard import get_signals

    # Check if today's note already exists
    existing = get_todays_note()
    if existing:
        return {
            "status": "exists",
            "message": "Today's note already exists",
            "note": existing
        }

    # Get current signals for context
    signals = get_signals()

    # Generate note
    note = create_daily_note(signals)

    if note:
        return {
            "status": "success",
            "message": "Note generated successfully",
            "note": note
        }
    else:
        return {
            "status": "error",
            "error": "Failed to generate note. Check ANTHROPIC_API_KEY environment variable."
        }


@router.get("/notes")
async def get_notes():
    """Get recent daily notes"""
    from app.data.notes import get_recent_notes

    notes = get_recent_notes(limit=10)

    return {
        "notes": notes,
        "count": len(notes)
    }


@router.delete("/notes/today")
async def delete_todays_note():
    """Delete today's note so it can be regenerated"""
    from app.data.notes import load_notes, save_notes
    from datetime import date

    today = date.today().isoformat()
    notes = load_notes()

    # Filter out today's note
    original_count = len(notes)
    notes = [n for n in notes if n.get('date') != today]

    if len(notes) < original_count:
        save_notes(notes)
        return {
            "status": "deleted",
            "message": f"Deleted note for {today}"
        }
    else:
        return {
            "status": "not_found",
            "message": f"No note found for {today}"
        }
