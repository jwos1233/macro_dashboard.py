"""
Dashboard routes for Epoch Macro
"""

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import signal generator (requires numpy, pandas, yfinance)
try:
    from signal_generator import SignalGenerator
    SIGNAL_GENERATOR_AVAILABLE = True
except ImportError:
    SIGNAL_GENERATOR_AVAILABLE = False
    print("Warning: SignalGenerator not available, using mock data")

from config import QUAD_ALLOCATIONS, QUADRANT_DESCRIPTIONS

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

# Cache for signals (refresh every 5 minutes in production)
_signal_cache = None
_cache_time = None


class MockSeries(dict):
    """Mock pandas Series for when pandas is not available"""
    def items(self):
        return super().items()


def get_mock_signals():
    """Return mock signals for development/demo"""
    return {
        'top_quadrants': ('Q1', 'Q3'),
        'quadrant_scores': MockSeries({
            'Q1': 8.5,
            'Q3': 4.2,
            'Q2': -1.3,
            'Q4': -5.8
        }),
        'target_weights': {
            'QQQ': 0.18,
            'ARKK': 0.12,
            'IWM': 0.08,
            'IBIT': 0.06,
            'GLD': 0.10,
            'XLE': 0.09,
            'DBC': 0.07,
            'TIP': 0.06,
            'XLV': 0.05,
            'XLU': 0.04,
        },
        'current_regime': 'Q1 + Q3',
        'timestamp': datetime.now(),
        'total_leverage': 1.45,
        'atr_data': {}
    }


def get_signals():
    """Get current signals from signal generator"""
    global _signal_cache, _cache_time

    # Use cache if less than 5 minutes old
    if _signal_cache and _cache_time and (datetime.now() - _cache_time) < timedelta(minutes=5):
        return _signal_cache

    # If signal generator not available, use mock data
    if not SIGNAL_GENERATOR_AVAILABLE:
        return get_mock_signals()

    try:
        sg = SignalGenerator()
        signals = sg.generate_signals()
        _signal_cache = signals
        _cache_time = datetime.now()
        return signals
    except Exception as e:
        print(f"Error generating signals: {e}")
        # Return mock data if signal generation fails
        return get_mock_signals()


def calculate_asset_class_breakdown(weights: dict) -> dict:
    """Calculate asset class breakdown from weights"""
    # Map tickers to asset classes
    equities = ['QQQ', 'ARKK', 'IWM', 'XLC', 'XLY', 'XLV', 'XLU', 'XLP', 'XLF', 'XLI', 'XLB', 'VTV', 'IWD']
    bonds = ['TLT', 'LQD', 'IEF', 'VGLT', 'MUB', 'TIP', 'VTIP']
    commodities = ['GLD', 'DBC', 'XLE', 'XOP', 'FCG', 'USO', 'GCC', 'DBA', 'REMX', 'URA', 'LIT', 'AA', 'PALL', 'VALT']
    crypto = ['IBIT', 'ETHA']
    real_assets = ['VNQ', 'PAVE']

    breakdown = {
        'Equities': 0,
        'Bonds': 0,
        'Commodities': 0,
        'Crypto': 0,
        'Real Assets': 0,
        'Cash': 0
    }

    total = sum(weights.values())

    for ticker, weight in weights.items():
        if ticker in equities:
            breakdown['Equities'] += weight
        elif ticker in bonds:
            breakdown['Bonds'] += weight
        elif ticker in commodities:
            breakdown['Commodities'] += weight
        elif ticker in crypto:
            breakdown['Crypto'] += weight
        elif ticker in real_assets:
            breakdown['Real Assets'] += weight

    # Cash is the remainder if total < 1.0
    if total < 1.0:
        breakdown['Cash'] = 1.0 - total

    return breakdown


@router.get("/dashboard")
@router.get("/dashboard/overview")
async def overview(request: Request):
    """Main dashboard overview page"""
    signals = get_signals()

    # Calculate asset class breakdown
    breakdown = calculate_asset_class_breakdown(signals['target_weights'])

    # Get top positions (sorted by weight)
    top_positions = sorted(
        signals['target_weights'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    return templates.TemplateResponse("dashboard/overview.html", {
        "request": request,
        "page": "overview",
        "signals": signals,
        "breakdown": breakdown,
        "top_positions": top_positions,
        "quad_descriptions": QUADRANT_DESCRIPTIONS,
    })


@router.get("/dashboard/signals")
async def signals_page(request: Request):
    """Full signals/momentum table page"""
    signals = get_signals()

    # Get all positions sorted by weight
    all_positions = sorted(
        signals['target_weights'].items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Build ticker to quadrant mapping
    ticker_quads = {}
    for quad, allocations in QUAD_ALLOCATIONS.items():
        for ticker in allocations.keys():
            if ticker not in ticker_quads:
                ticker_quads[ticker] = []
            ticker_quads[ticker].append(quad)

    return templates.TemplateResponse("dashboard/signals.html", {
        "request": request,
        "page": "signals",
        "signals": signals,
        "all_positions": all_positions,
        "ticker_quads": ticker_quads,
        "quad_descriptions": QUADRANT_DESCRIPTIONS,
    })


@router.get("/dashboard/allocation")
async def allocation_page(request: Request):
    """Detailed allocation page"""
    signals = get_signals()

    # Calculate asset class breakdown
    breakdown = calculate_asset_class_breakdown(signals['target_weights'])

    # Get all positions with quadrant info
    positions_with_info = []
    for ticker, weight in signals['target_weights'].items():
        quads = []
        for quad, allocations in QUAD_ALLOCATIONS.items():
            if ticker in allocations:
                quads.append(quad)

        positions_with_info.append({
            'ticker': ticker,
            'weight': weight,
            'quadrants': quads,
            'notional_10k': weight * 10000,
        })

    positions_with_info.sort(key=lambda x: x['weight'], reverse=True)

    return templates.TemplateResponse("dashboard/allocation.html", {
        "request": request,
        "page": "allocation",
        "signals": signals,
        "breakdown": breakdown,
        "positions": positions_with_info,
        "quad_descriptions": QUADRANT_DESCRIPTIONS,
        "quad_allocations": QUAD_ALLOCATIONS,
    })


@router.get("/dashboard/research")
async def research_page(request: Request):
    """Research/daily notes page"""
    signals = get_signals()

    # Mock research notes for now
    research_notes = [
        {
            'date': '2026-01-06',
            'title': 'Regime Shift: Q1 Dominance Continues',
            'summary': 'Growth momentum remains strong with tech leading. Goldilocks conditions persist as inflation moderates.',
            'regime': 'Q1 + Q3',
        },
        {
            'date': '2026-01-05',
            'title': 'Weekly Rebalance: Adding Commodity Exposure',
            'summary': 'Increasing allocation to energy and materials as Q3 scores improve.',
            'regime': 'Q1 + Q3',
        },
        {
            'date': '2026-01-04',
            'title': 'Market Update: Momentum Confirmation',
            'summary': 'All positions remain above 50-day EMA. No changes to current allocation.',
            'regime': 'Q1 + Q3',
        },
    ]

    return templates.TemplateResponse("dashboard/research.html", {
        "request": request,
        "page": "research",
        "signals": signals,
        "research_notes": research_notes,
        "quad_descriptions": QUADRANT_DESCRIPTIONS,
    })


@router.get("/dashboard/performance")
async def performance_page(request: Request):
    """Performance/track record page"""
    signals = get_signals()

    # Mock performance data
    performance = {
        'ytd_return': 12.5,
        'mtd_return': 3.2,
        'total_return': 45.8,
        'sharpe': 1.42,
        'max_drawdown': -15.3,
        'win_rate': 58.2,
        'avg_trade': 2.1,
    }

    # Mock monthly returns
    monthly_returns = [
        {'month': 'Jan 2026', 'return': 3.2},
        {'month': 'Dec 2025', 'return': 4.1},
        {'month': 'Nov 2025', 'return': 2.8},
        {'month': 'Oct 2025', 'return': -1.5},
        {'month': 'Sep 2025', 'return': 1.9},
        {'month': 'Aug 2025', 'return': 3.4},
    ]

    return templates.TemplateResponse("dashboard/performance.html", {
        "request": request,
        "page": "performance",
        "signals": signals,
        "performance": performance,
        "monthly_returns": monthly_returns,
        "quad_descriptions": QUADRANT_DESCRIPTIONS,
    })
