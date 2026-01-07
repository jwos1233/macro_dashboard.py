"""
Dashboard routes for Epoch Macro
"""

from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path
from datetime import datetime, timedelta
import sys
import json

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
from app.data import load_backtest_results

router = APIRouter()
templates = Jinja2Templates(directory=Path(__file__).parent.parent / "templates")

# Cache settings
SIGNAL_CACHE_HOURS = 24  # Cache signals for 24 hours (refreshes once per day)
SIGNALS_CACHE_FILE = Path(__file__).parent.parent / "data" / "signals_cache.json"

# In-memory cache for signals
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


def save_signals_to_disk(signals: dict):
    """Save signals to disk for persistence"""
    try:
        # Convert to JSON-serializable format
        cache_data = {
            'top_quadrants': signals['top_quadrants'],
            'quadrant_scores': dict(signals['quadrant_scores']) if hasattr(signals['quadrant_scores'], 'items') else signals['quadrant_scores'],
            'target_weights': dict(signals['target_weights']) if hasattr(signals['target_weights'], 'items') else signals['target_weights'],
            'current_regime': signals['current_regime'],
            'timestamp': signals['timestamp'].isoformat() if hasattr(signals['timestamp'], 'isoformat') else str(signals['timestamp']),
            'total_leverage': signals['total_leverage'],
            'cached_at': datetime.now().isoformat()
        }
        with open(SIGNALS_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"Signals cached to disk", flush=True)
    except Exception as e:
        print(f"Error saving signals to disk: {e}", flush=True)


def load_signals_from_disk():
    """Load signals from disk if fresh enough"""
    global _signal_cache, _cache_time

    if not SIGNALS_CACHE_FILE.exists():
        return None

    try:
        with open(SIGNALS_CACHE_FILE, 'r') as f:
            cache_data = json.load(f)

        # Check if cache is still fresh
        cached_at = datetime.fromisoformat(cache_data['cached_at'])
        if (datetime.now() - cached_at) > timedelta(hours=SIGNAL_CACHE_HOURS):
            print("Disk cache expired", flush=True)
            return None

        # Convert back to expected format
        signals = {
            'top_quadrants': tuple(cache_data['top_quadrants']),
            'quadrant_scores': cache_data['quadrant_scores'],
            'target_weights': cache_data['target_weights'],
            'current_regime': cache_data['current_regime'],
            'timestamp': datetime.fromisoformat(cache_data['timestamp']),
            'total_leverage': cache_data['total_leverage'],
            'atr_data': {}
        }

        _signal_cache = signals
        _cache_time = cached_at
        print(f"Loaded signals from disk cache (cached {cached_at})", flush=True)
        return signals
    except Exception as e:
        print(f"Error loading signals from disk: {e}", flush=True)
        return None


def get_signals():
    """Get current signals from signal generator with caching"""
    global _signal_cache, _cache_time

    # Check in-memory cache first (fastest)
    if _signal_cache and _cache_time and (datetime.now() - _cache_time) < timedelta(hours=SIGNAL_CACHE_HOURS):
        return _signal_cache

    # Try loading from disk cache
    disk_signals = load_signals_from_disk()
    if disk_signals:
        return disk_signals

    # If signal generator not available, use mock data
    if not SIGNAL_GENERATOR_AVAILABLE:
        return get_mock_signals()

    # Generate fresh signals
    try:
        print("Generating fresh signals (this may take 30-60 seconds)...", flush=True)
        sg = SignalGenerator()
        signals = sg.generate_signals()
        _signal_cache = signals
        _cache_time = datetime.now()

        # Save to disk for persistence
        save_signals_to_disk(signals)

        return signals
    except Exception as e:
        print(f"Error generating signals: {e}", flush=True)
        # Return mock data if signal generation fails
        return get_mock_signals()


def calculate_asset_class_breakdown(weights: dict) -> dict:
    """Calculate asset class breakdown from weights"""
    # Map tickers to asset classes
    equities = ['QQQ', 'ARKK', 'IWM', 'XLC', 'XLY', 'XLV', 'XLU', 'XLP', 'XLF', 'XLI', 'XLB', 'VTV', 'IWD',
                'MSTR', 'BMNR', 'ARKX', 'BOTZ', 'EEM']
    bonds = ['TLT', 'LQD', 'IEF', 'VGLT', 'MUB', 'TIP', 'VTIP']
    commodities = ['GLD', 'DBC', 'XLE', 'XOP', 'FCG', 'USO', 'GCC', 'DBA', 'REMX', 'URA', 'LIT', 'AA', 'PALL', 'VALT']
    crypto = []  # No crypto ETFs in current allocations
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
    backtest = load_backtest_results()

    # Calculate asset class breakdown
    breakdown = calculate_asset_class_breakdown(signals['target_weights'])

    # Get top positions (sorted by weight)
    top_positions = sorted(
        signals['target_weights'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    # Get performance summary from backtest
    summary = backtest.get('summary', {})
    monthly = backtest.get('monthly_returns', [])
    ytd_return = sum(m.get('return', 0) for m in monthly if m.get('month', '').startswith('2025'))

    performance_summary = {
        'total_return': summary.get('total_return', 0),
        'ytd_return': ytd_return,
        'sharpe': summary.get('sharpe', 0),
        'max_drawdown': summary.get('max_drawdown', 0),
    }

    return templates.TemplateResponse("dashboard/overview.html", {
        "request": request,
        "page": "overview",
        "signals": signals,
        "breakdown": breakdown,
        "top_positions": top_positions,
        "performance": performance_summary,
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
    backtest = load_backtest_results()

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
        "asset_class_history": backtest.get('asset_class_history', []),
    })


@router.get("/dashboard/research")
async def research_page(request: Request):
    """Research/daily notes page"""
    from app.data.notes import ensure_todays_note

    signals = get_signals()
    backtest = load_backtest_results()

    # Get daily notes (auto-generates today's if missing)
    research_notes = ensure_todays_note(signals)

    # Get regime history from backtest
    regime_history = backtest.get('regime_history', [])

    return templates.TemplateResponse("dashboard/research.html", {
        "request": request,
        "page": "research",
        "signals": signals,
        "research_notes": research_notes,
        "regime_history": regime_history,
        "quad_descriptions": QUADRANT_DESCRIPTIONS,
    })


@router.get("/dashboard/performance")
async def performance_page(request: Request):
    """Performance/track record page"""
    signals = get_signals()
    backtest = load_backtest_results()

    # Get summary stats from backtest
    summary = backtest.get('summary', {})
    benchmark = backtest.get('vs_benchmark', {})

    # Calculate YTD and MTD from monthly returns
    monthly = backtest.get('monthly_returns', [])
    ytd_return = sum(m.get('return', 0) for m in monthly if m.get('month', '').startswith('2025'))
    mtd_return = monthly[0].get('return', 0) if monthly else 0

    performance = {
        'ytd_return': ytd_return,
        'mtd_return': mtd_return,
        'total_return': summary.get('total_return', 0),
        'sharpe': summary.get('sharpe', 0),
        'max_drawdown': summary.get('max_drawdown', 0),
        'win_rate': summary.get('win_rate', 0),
        'volatility': summary.get('volatility', 0),
        'annual_return': summary.get('annual_return', 0),
        'total_trades': summary.get('total_trades', 0),
        'trading_costs': summary.get('trading_costs', 0),
        'initial_capital': summary.get('initial_capital', 50000),
        'final_value': summary.get('final_value', 50000),
        'start_date': summary.get('start_date', ''),
        'end_date': summary.get('end_date', ''),
    }

    # Format monthly returns for display
    monthly_returns = []
    for m in monthly[:12]:  # Last 12 months
        month_str = m.get('month', '')
        # Convert YYYY-MM to readable format
        if month_str:
            try:
                year, month = month_str.split('-')
                month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                formatted = f"{month_names[int(month)]} {year}"
            except:
                formatted = month_str
        else:
            formatted = 'Unknown'

        monthly_returns.append({
            'month': formatted,
            'return': m.get('return', 0),
            'drawdown': m.get('drawdown', 0),
            'sharpe': m.get('sharpe', 0)
        })

    return templates.TemplateResponse("dashboard/performance.html", {
        "request": request,
        "page": "performance",
        "signals": signals,
        "performance": performance,
        "monthly_returns": monthly_returns,
        "annual_returns": backtest.get('annual_returns', []),
        "regime_performance": backtest.get('regime_performance', {}),
        "vs_benchmark": benchmark,
        "equity_curve": backtest.get('equity_curve', []),
        "spy_curve": backtest.get('spy_curve', []),
        "quad_descriptions": QUADRANT_DESCRIPTIONS,
        "data_source": backtest.get('data_source', 'unknown'),
        "generated_at": backtest.get('generated_at'),
    })
