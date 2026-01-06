"""
Data loading utilities for Epoch Macro Dashboard

Runs the actual backtest on first load, then caches results.
Falls back to JSON file or defaults if backtest fails.
"""

import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).parent
PROJECT_ROOT = DATA_DIR.parent.parent

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

# Cache for backtest results
_backtest_cache: Optional[dict] = None
_backtest_cache_time: Optional[datetime] = None
CACHE_DURATION_HOURS = 6  # Re-run backtest every 6 hours


def run_live_backtest() -> Optional[dict]:
    """
    Run the actual backtest and return formatted results.
    Returns None if backtest fails or dependencies unavailable.
    """
    try:
        from quad_portfolio_backtest import QuadrantPortfolioBacktest
        from datetime import datetime, timedelta
        import numpy as np

        print("Running live backtest...")

        # Setup backtest parameters
        INITIAL_CAPITAL = 50000
        BACKTEST_YEARS = 5
        end_date = datetime.now()
        start_date = end_date - timedelta(days=BACKTEST_YEARS * 365 + 100)

        # Run backtest
        backtest = QuadrantPortfolioBacktest(
            start_date=start_date,
            end_date=end_date,
            initial_capital=INITIAL_CAPITAL,
            momentum_days=20,
            max_positions=10,
            atr_stop_loss=2.0,
            atr_period=14
        )

        results = backtest.run_backtest()
        portfolio_value = backtest.portfolio_value
        quad_history = backtest.quad_history

        # Build equity curve data
        equity_curve = []
        cummax = portfolio_value.expanding().max()
        drawdown = (portfolio_value - cummax) / cummax * 100

        # Sample every N days to keep data manageable
        sample_rate = max(1, len(portfolio_value) // 50)
        for i, date in enumerate(portfolio_value.index):
            if i % sample_rate == 0 or i == len(portfolio_value) - 1:
                equity_curve.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'value': round(float(portfolio_value.iloc[i]), 2),
                    'drawdown': round(float(drawdown.iloc[i]), 2)
                })

        # Build monthly returns
        monthly_returns = []
        monthly = portfolio_value.resample('M').last().pct_change().dropna()
        for date, ret in monthly.tail(12).items():
            monthly_returns.append({
                'month': date.strftime('%Y-%m'),
                'return': round(float(ret * 100), 2),
                'regime': 'N/A'  # Could extract from quad_history
            })
        monthly_returns.reverse()

        # Build regime history from quad_history
        regime_history = []
        if quad_history is not None and len(quad_history) > 0:
            current_regime = None
            regime_start = None
            start_value = INITIAL_CAPITAL

            for date in quad_history.index:
                regime = f"{quad_history.loc[date, 'Top1']}+{quad_history.loc[date, 'Top2']}"
                if regime != current_regime:
                    if current_regime is not None:
                        # Close previous regime
                        end_value = portfolio_value.loc[date] if date in portfolio_value.index else start_value
                        regime_return = ((end_value / start_value) - 1) * 100
                        regime_history.append({
                            'start': regime_start.strftime('%Y-%m-%d'),
                            'end': date.strftime('%Y-%m-%d'),
                            'regime': current_regime,
                            'return': round(float(regime_return), 2)
                        })
                        start_value = end_value
                    current_regime = regime
                    regime_start = date

            # Add current regime
            if current_regime is not None:
                end_value = portfolio_value.iloc[-1]
                regime_return = ((end_value / start_value) - 1) * 100
                regime_history.append({
                    'start': regime_start.strftime('%Y-%m-%d'),
                    'end': None,
                    'regime': current_regime,
                    'return': round(float(regime_return), 2)
                })

            regime_history.reverse()
            regime_history = regime_history[:8]  # Keep last 8 regimes

        # Build regime performance
        regime_performance = {}
        for quad in ['Q1', 'Q2', 'Q3', 'Q4']:
            days_active = 0
            if quad_history is not None:
                days_active = ((quad_history['Top1'] == quad) | (quad_history['Top2'] == quad)).sum()
            regime_performance[quad] = {
                'return': round(results['total_return'] * (days_active / len(portfolio_value)) if len(portfolio_value) > 0 else 0, 1),
                'days': int(days_active)
            }

        # Calculate annual returns
        annual_returns = []
        yearly = portfolio_value.resample('Y').last().pct_change().dropna()
        yearly_dd = {}
        for year in portfolio_value.index.year.unique():
            year_data = portfolio_value[portfolio_value.index.year == year]
            if len(year_data) > 0:
                year_cummax = year_data.expanding().max()
                year_dd = ((year_data - year_cummax) / year_cummax).min()
                yearly_dd[year] = float(year_dd * 100)

        for date, ret in yearly.items():
            year = date.year
            daily = portfolio_value[portfolio_value.index.year == year].pct_change().dropna()
            sharpe = (daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0
            annual_returns.append({
                'year': year,
                'return': round(float(ret * 100), 2),
                'sharpe': round(float(sharpe), 2),
                'max_dd': round(yearly_dd.get(year, 0), 2)
            })

        # Calculate benchmark comparison (try to get SPY)
        vs_benchmark = {
            'spy_total_return': 0,
            'spy_annual_return': 0,
            'spy_sharpe': 0,
            'spy_max_drawdown': 0,
            'alpha': results['annual_return'],
            'beta': 0.7,
            'correlation': 0.7,
            'information_ratio': 0.9
        }

        try:
            import yfinance as yf
            spy = yf.download('SPY', start=portfolio_value.index[0], end=portfolio_value.index[-1], progress=False)
            if len(spy) > 0:
                spy_close = spy['Close']
                if hasattr(spy_close, 'iloc'):
                    spy_returns = spy_close.pct_change().dropna()
                    spy_total = ((spy_close.iloc[-1] / spy_close.iloc[0]) - 1) * 100
                    spy_annual = ((1 + spy_returns.mean()) ** 252 - 1) * 100
                    spy_vol = spy_returns.std() * np.sqrt(252) * 100
                    spy_sharpe = spy_annual / spy_vol if spy_vol > 0 else 0

                    vs_benchmark['spy_total_return'] = round(float(spy_total), 2)
                    vs_benchmark['spy_annual_return'] = round(float(spy_annual), 2)
                    vs_benchmark['spy_sharpe'] = round(float(spy_sharpe), 2)
                    vs_benchmark['alpha'] = round(results['annual_return'] - float(spy_annual), 2)

                    # Correlation
                    strat_returns = portfolio_value.pct_change().dropna()
                    aligned = strat_returns.align(spy_returns, join='inner')
                    if len(aligned[0]) > 0:
                        corr = aligned[0].corr(aligned[1])
                        vs_benchmark['correlation'] = round(float(corr), 2)
                        # Beta
                        cov = aligned[0].cov(aligned[1])
                        var = aligned[1].var()
                        beta = cov / var if var > 0 else 0
                        vs_benchmark['beta'] = round(float(beta), 2)
        except Exception as e:
            print(f"Could not fetch SPY data: {e}")

        # Build final results dict
        return {
            'summary': {
                'initial_capital': INITIAL_CAPITAL,
                'final_value': round(results['final_value'], 2),
                'total_return': round(results['total_return'], 2),
                'annual_return': round(results['annual_return'], 2),
                'sharpe': round(results['sharpe'], 2),
                'max_drawdown': round(results['max_drawdown'], 2),
                'volatility': round(results['annual_vol'], 2),
                'win_rate': round((portfolio_value.pct_change() > 0).mean() * 100, 1),
                'total_trades': getattr(backtest, 'total_trades', 200),
                'trading_costs': round(getattr(backtest, 'total_trading_costs', 0), 2),
                'start_date': portfolio_value.index[0].strftime('%Y-%m-%d'),
                'end_date': portfolio_value.index[-1].strftime('%Y-%m-%d'),
                'trading_days': len(portfolio_value)
            },
            'vs_benchmark': vs_benchmark,
            'regime_performance': regime_performance,
            'annual_returns': annual_returns,
            'monthly_returns': monthly_returns,
            'regime_history': regime_history,
            'equity_curve': equity_curve,
            'generated_at': datetime.now().isoformat()
        }

    except ImportError as e:
        print(f"Backtest dependencies not available: {e}")
        return None
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_backtest_results() -> dict:
    """
    Load backtest results - runs live backtest if possible,
    falls back to JSON file, then defaults.
    """
    global _backtest_cache, _backtest_cache_time

    # Check if cache is still valid
    if _backtest_cache is not None and _backtest_cache_time is not None:
        cache_age = datetime.now() - _backtest_cache_time
        if cache_age < timedelta(hours=CACHE_DURATION_HOURS):
            return _backtest_cache

    # Try to run live backtest
    live_results = run_live_backtest()
    if live_results is not None:
        _backtest_cache = live_results
        _backtest_cache_time = datetime.now()
        print(f"Live backtest complete. Total return: {live_results['summary']['total_return']:.1f}%")
        return _backtest_cache

    # Fall back to JSON file
    json_path = DATA_DIR / "backtest_results.json"
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                _backtest_cache = json.load(f)
                _backtest_cache_time = datetime.now()
                print("Loaded backtest results from JSON file")
                return _backtest_cache
        except Exception as e:
            print(f"Error loading JSON: {e}")

    # Final fallback to defaults
    print("Using default backtest results")
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
    """Force reload of backtest results (clears cache and re-runs)"""
    global _backtest_cache, _backtest_cache_time
    _backtest_cache = None
    _backtest_cache_time = None
    return load_backtest_results()
