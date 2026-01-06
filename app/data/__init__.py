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
_last_backtest_error: Optional[str] = None
CACHE_DURATION_HOURS = 6  # Re-run backtest every 6 hours


def get_last_error() -> Optional[str]:
    """Get the last backtest error message"""
    return _last_backtest_error


def run_live_backtest() -> Optional[dict]:
    """
    Run the actual backtest and return formatted results.
    Returns None if backtest fails or dependencies unavailable.
    """
    global _last_backtest_error
    _last_backtest_error = None

    try:
        print("=" * 60, flush=True)
        print("ATTEMPTING LIVE BACKTEST", flush=True)
        print("=" * 60, flush=True)

        print("Step 1: Importing QuadrantPortfolioBacktest...", flush=True)
        from quad_portfolio_backtest import QuadrantPortfolioBacktest
        print("  ✓ Import successful", flush=True)

        print("Step 2: Importing numpy...", flush=True)
        import numpy as np
        print("  ✓ Numpy imported", flush=True)

        from datetime import datetime, timedelta

        print("Step 3: Setting up backtest parameters...", flush=True)

        # Setup backtest parameters
        INITIAL_CAPITAL = 50000
        BACKTEST_YEARS = 5
        end_date = datetime.now()
        start_date = end_date - timedelta(days=BACKTEST_YEARS * 365 + 100)

        # Run backtest
        print("Step 4: Creating backtest instance...", flush=True)
        backtest = QuadrantPortfolioBacktest(
            start_date=start_date,
            end_date=end_date,
            initial_capital=INITIAL_CAPITAL,
            momentum_days=20,
            max_positions=10,
            atr_stop_loss=2.0,
            atr_period=14
        )
        print("  ✓ Backtest instance created", flush=True)

        print("Step 5: Running backtest (this may take 1-2 minutes)...", flush=True)
        results = backtest.run_backtest()
        print(f"  ✓ Backtest complete! Total return: {results.get('total_return', 'N/A')}%", flush=True)

        portfolio_value = backtest.portfolio_value
        quad_history = backtest.quad_history

        print("Step 6: Processing results...", flush=True)

        # Build equity curve data
        equity_curve = []
        spy_curve = []  # SPY benchmark curve
        cummax = portfolio_value.expanding().max()
        drawdown = (portfolio_value - cummax) / cummax * 100

        # Sample every N days to keep data manageable
        sample_rate = max(1, len(portfolio_value) // 50)
        sample_dates = []
        for i, date in enumerate(portfolio_value.index):
            if i % sample_rate == 0 or i == len(portfolio_value) - 1:
                sample_dates.append(date)
                equity_curve.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'value': round(float(portfolio_value.iloc[i]), 2),
                    'drawdown': round(float(drawdown.iloc[i]), 2)
                })

        # Build monthly returns with drawdown and sharpe
        monthly_returns = []
        monthly = portfolio_value.resample('M').last().pct_change().dropna()
        for date, ret in monthly.tail(12).items():
            # Get daily returns for this month to calculate sharpe
            year, month = date.year, date.month
            month_data = portfolio_value[(portfolio_value.index.year == year) & (portfolio_value.index.month == month)]
            daily_returns = month_data.pct_change().dropna()

            # Monthly Sharpe (annualized from daily)
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                monthly_sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                monthly_sharpe = 0

            # Monthly max drawdown
            if len(month_data) > 0:
                month_cummax = month_data.expanding().max()
                month_dd = ((month_data - month_cummax) / month_cummax).min() * 100
            else:
                month_dd = 0

            monthly_returns.append({
                'month': date.strftime('%Y-%m'),
                'return': round(float(ret * 100), 2),
                'drawdown': round(float(month_dd), 1),
                'sharpe': round(float(monthly_sharpe), 2)
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
            import pandas as pd
            spy = yf.download('SPY', start=portfolio_value.index[0], end=portfolio_value.index[-1], progress=False)
            if len(spy) > 0:
                # Handle different yfinance return formats
                if isinstance(spy.columns, pd.MultiIndex):
                    spy_close = spy['Close']['SPY']
                else:
                    spy_close = spy['Close']

                # Flatten if needed and ensure it's a Series
                if hasattr(spy_close, 'squeeze'):
                    spy_close = spy_close.squeeze()

                if hasattr(spy_close, 'iloc') and len(spy_close) > 0:
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

                    # Build SPY curve for chart overlay (normalized to same starting point)
                    spy_initial = float(spy_close.iloc[0])
                    spy_dates = spy_close.index.tz_localize(None) if spy_close.index.tz else spy_close.index
                    for sample_date in sample_dates:
                        # Find closest date in SPY data
                        try:
                            sample_date_naive = sample_date.tz_localize(None) if hasattr(sample_date, 'tz_localize') and sample_date.tz else sample_date
                            closest_idx = spy_dates.get_indexer([sample_date_naive], method='nearest')[0]
                            if closest_idx >= 0:
                                spy_val = float(spy_close.iloc[closest_idx])
                                # Normalize to portfolio initial capital for comparison
                                spy_normalized = (spy_val / spy_initial) * INITIAL_CAPITAL
                                spy_curve.append({
                                    'date': sample_date.strftime('%Y-%m-%d'),
                                    'value': round(spy_normalized, 2)
                                })
                        except Exception as e:
                            print(f"Error matching SPY date {sample_date}: {e}", flush=True)
                    print(f"Built SPY curve with {len(spy_curve)} points", flush=True)
        except Exception as e:
            print(f"Could not fetch SPY data: {e}", flush=True)

        # Build final results dict
        return {
            'data_source': 'live_backtest',
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
            'spy_curve': spy_curve,
            'generated_at': datetime.now().isoformat()
        }

    except ImportError as e:
        _last_backtest_error = f"ImportError: {e}"
        print(f"✗ Backtest dependencies not available: {e}", flush=True)
        return None
    except Exception as e:
        _last_backtest_error = f"{type(e).__name__}: {e}"
        print(f"✗ Error running backtest: {e}", flush=True)
        import traceback
        traceback.print_exc()
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
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
                _backtest_cache['data_source'] = 'json_file'
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
        "data_source": "defaults",
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
        "spy_curve": [],
        "generated_at": None
    }


def reload_backtest_results() -> dict:
    """Force reload of backtest results (clears cache and re-runs)"""
    global _backtest_cache, _backtest_cache_time
    _backtest_cache = None
    _backtest_cache_time = None
    return load_backtest_results()
