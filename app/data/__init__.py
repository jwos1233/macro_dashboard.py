"""
Data loading utilities for Epoch Macro Dashboard
"""

import json
from pathlib import Path
from typing import Optional
from datetime import datetime

DATA_DIR = Path(__file__).parent

# Cache for backtest results
_backtest_cache: Optional[dict] = None


def load_backtest_results() -> dict:
    """Load backtest results from JSON file"""
    global _backtest_cache

    if _backtest_cache is not None:
        return _backtest_cache

    json_path = DATA_DIR / "backtest_results.json"

    if not json_path.exists():
        return get_default_backtest_results()

    try:
        with open(json_path, 'r') as f:
            _backtest_cache = json.load(f)
        return _backtest_cache
    except Exception as e:
        print(f"Error loading backtest results: {e}")
        return get_default_backtest_results()


def get_default_backtest_results() -> dict:
    """Return default/placeholder backtest results"""
    return {
        "summary": {
            "initial_capital": 50000,
            "final_value": 50000,
            "total_return": 0,
            "annual_return": 0,
            "sharpe": 0,
            "max_drawdown": 0,
            "volatility": 0,
            "win_rate": 0,
            "total_trades": 0,
            "trading_costs": 0,
            "start_date": "2021-01-01",
            "end_date": datetime.now().strftime("%Y-%m-%d"),
            "trading_days": 0
        },
        "vs_benchmark": {
            "spy_total_return": 0,
            "spy_annual_return": 0,
            "spy_sharpe": 0,
            "spy_max_drawdown": 0,
            "alpha": 0,
            "beta": 0,
            "correlation": 0,
            "information_ratio": 0
        },
        "regime_performance": {},
        "annual_returns": [],
        "monthly_returns": [],
        "regime_history": [],
        "equity_curve": [],
        "generated_at": None
    }


def reload_backtest_results() -> dict:
    """Force reload of backtest results (clears cache)"""
    global _backtest_cache
    _backtest_cache = None
    return load_backtest_results()
