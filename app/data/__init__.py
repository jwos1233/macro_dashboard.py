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

        print("Step 2: Importing numpy and pandas...", flush=True)
        import numpy as np
        import pandas as pd
        print("  ✓ Numpy and pandas imported", flush=True)

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
        spy_curve = []  # SPY benchmark curve (sampled)
        spy_curve_daily = []  # SPY benchmark curve (daily for 2yr)
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

        # Build DAILY equity curve for last 2 years (for detailed performance view)
        equity_curve_daily = []
        two_years_ago = datetime.now() - timedelta(days=730)
        for date in portfolio_value.index:
            if date >= two_years_ago:
                equity_curve_daily.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'value': round(float(portfolio_value.loc[date]), 2),
                    'drawdown': round(float(drawdown.loc[date]), 2)
                })
        print(f"Built daily equity curve with {len(equity_curve_daily)} points (2yr)", flush=True)

        # Build monthly returns with drawdown and sharpe
        monthly_returns = []
        monthly = portfolio_value.resample('M').last().pct_change().dropna()

        # Determine current month to mark as incomplete
        current_year = datetime.now().year
        current_month = datetime.now().month

        for date, ret in monthly.tail(12).items():
            # Get daily returns for this month to calculate sharpe
            year, month = date.year, date.month
            month_data = portfolio_value[(portfolio_value.index.year == year) & (portfolio_value.index.month == month)]
            daily_returns = month_data.pct_change().dropna()

            # Check if this is the current (incomplete) month
            is_current_month = (year == current_year and month == current_month)

            # Monthly Sharpe (annualized from daily) - only for complete months
            if is_current_month:
                monthly_sharpe = None  # Incomplete month - don't calculate
            elif len(daily_returns) > 5 and daily_returns.std() > 0:
                # Need at least 5 days of data for meaningful Sharpe
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
                'sharpe': round(float(monthly_sharpe), 2) if monthly_sharpe is not None else None
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

        # Build historical asset class allocation (sampled for overview)
        asset_class_history = []
        # Build daily asset class history for last 2 years (for allocation page)
        asset_class_daily = []
        if quad_history is not None and len(quad_history) > 0:
            try:
                from config import QUAD_ALLOCATIONS

                # Asset class categorization (same as dashboard.py)
                equities = ['QQQ', 'ARKK', 'IWM', 'XLC', 'XLY', 'XLV', 'XLU', 'XLP', 'XLF', 'XLI', 'XLB', 'VTV', 'IWD',
                            'ARKX', 'BOTZ', 'EEM', 'AA']
                bonds = ['TLT', 'LQD', 'IEF', 'VGLT', 'MUB', 'TIP', 'VTIP']
                commodities = ['GLD', 'DBC', 'XLE', 'XOP', 'FCG', 'USO', 'GCC', 'DBA', 'REMX', 'URA', 'LIT', 'PALL', 'VALT']
                crypto = ['BTC-USD']
                real_assets = ['VNQ', 'PAVE']

                def categorize_allocations(allocations):
                    """Categorize allocations into asset classes"""
                    breakdown = {'Equities': 0, 'Bonds': 0, 'Commodities': 0, 'Crypto': 0, 'Real Assets': 0}
                    for ticker, weight in allocations.items():
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
                    return breakdown

                # Sample at same rate as equity curve for consistency (overview)
                # Use actual target_weights from backtest (includes vol weighting, EMA filters, max_positions)
                if hasattr(backtest, 'target_weights') and backtest.target_weights is not None:
                    target_weights_df = backtest.target_weights
                    for sample_date in sample_dates:
                        if sample_date in target_weights_df.index:
                            # Get actual weights for this date
                            day_weights = target_weights_df.loc[sample_date]
                            actual_allocations = {}
                            for ticker in day_weights.index:
                                weight = day_weights[ticker]
                                if pd.notna(weight) and weight > 0.001:
                                    actual_allocations[ticker] = float(weight)

                            breakdown = categorize_allocations(actual_allocations)
                            breakdown = {k: round(v * 100, 1) for k, v in breakdown.items()}

                            asset_class_history.append({
                                'date': sample_date.strftime('%Y-%m-%d'),
                                **breakdown
                            })
                else:
                    # Fallback to QUAD_ALLOCATIONS if target_weights not available
                    for sample_date in sample_dates:
                        if sample_date in quad_history.index:
                            top1 = quad_history.loc[sample_date, 'Top1']
                            top2 = quad_history.loc[sample_date, 'Top2']

                            combined = {}
                            for quad in [top1, top2]:
                                if quad in QUAD_ALLOCATIONS:
                                    for ticker, weight in QUAD_ALLOCATIONS[quad].items():
                                        combined[ticker] = combined.get(ticker, 0) + weight * 0.5

                            breakdown = categorize_allocations(combined)
                            breakdown = {k: round(v * 100, 1) for k, v in breakdown.items()}

                            asset_class_history.append({
                                'date': sample_date.strftime('%Y-%m-%d'),
                                **breakdown
                            })

                # Daily history for last 2 years using ACTUAL target_weights from backtest
                # (not theoretical QUAD_ALLOCATIONS - this shows what we actually held)
                two_years_ago = datetime.now() - timedelta(days=730)

                # Use actual target_weights from backtest (includes vol weighting, EMA filters, max_positions)
                if hasattr(backtest, 'target_weights') and backtest.target_weights is not None:
                    target_weights = backtest.target_weights
                    recent_dates = [d for d in target_weights.index if d >= two_years_ago]

                    for date in recent_dates:
                        # Get actual weights for this date (these are the real positions)
                        day_weights = target_weights.loc[date]

                        # Build allocations dict from actual non-zero positions
                        actual_allocations = {}
                        for ticker in day_weights.index:
                            weight = day_weights[ticker]
                            if pd.notna(weight) and weight > 0.001:  # Only include actual positions
                                actual_allocations[ticker] = float(weight)

                        breakdown = categorize_allocations(actual_allocations)
                        breakdown = {k: round(v * 100, 1) for k, v in breakdown.items()}
                        # Calculate total exposure for this day
                        total_exposure = sum(breakdown.values())

                        asset_class_daily.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'total': round(total_exposure, 1),
                            **breakdown
                        })
                else:
                    # Fallback: use quad_history with QUAD_ALLOCATIONS if target_weights not available
                    recent_dates = [d for d in quad_history.index if d >= two_years_ago]
                    for date in recent_dates:
                        top1 = quad_history.loc[date, 'Top1']
                        top2 = quad_history.loc[date, 'Top2']

                        combined = {}
                        for quad in [top1, top2]:
                            if quad in QUAD_ALLOCATIONS:
                                for ticker, weight in QUAD_ALLOCATIONS[quad].items():
                                    combined[ticker] = combined.get(ticker, 0) + weight * 0.5

                        breakdown = categorize_allocations(combined)
                        breakdown = {k: round(v * 100, 1) for k, v in breakdown.items()}
                        total_exposure = sum(breakdown.values())

                        asset_class_daily.append({
                            'date': date.strftime('%Y-%m-%d'),
                            'total': round(total_exposure, 1),
                            **breakdown
                        })

                using_actual = hasattr(backtest, 'target_weights') and backtest.target_weights is not None
                print(f"Built asset class history with {len(asset_class_history)} points (sampled, {'actual positions' if using_actual else 'theoretical'})", flush=True)
                print(f"Built daily asset class history with {len(asset_class_daily)} points (2yr, {'actual positions' if using_actual else 'theoretical'})", flush=True)
            except Exception as e:
                print(f"Could not build asset class history: {e}", flush=True)

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

                    # Build DAILY SPY curve for last 2 years
                    spy_curve_daily = []
                    two_years_ago_spy = datetime.now() - timedelta(days=730)
                    # Get SPY initial value for 2yr period
                    spy_2yr_dates = [d for d in spy_dates if d >= two_years_ago_spy]
                    if len(spy_2yr_dates) > 0:
                        spy_2yr_initial_idx = spy_dates.get_indexer([spy_2yr_dates[0]], method='nearest')[0]
                        spy_2yr_initial = float(spy_close.iloc[spy_2yr_initial_idx]) if spy_2yr_initial_idx >= 0 else spy_initial
                        for spy_date in spy_2yr_dates:
                            try:
                                idx = spy_dates.get_indexer([spy_date], method='nearest')[0]
                                if idx >= 0:
                                    spy_val = float(spy_close.iloc[idx])
                                    # Normalize to $10k starting point for "Growth of $10k" view
                                    spy_normalized = (spy_val / spy_2yr_initial) * 10000
                                    spy_curve_daily.append({
                                        'date': spy_date.strftime('%Y-%m-%d'),
                                        'value': round(spy_normalized, 2)
                                    })
                            except Exception as e:
                                pass
                        print(f"Built daily SPY curve with {len(spy_curve_daily)} points (2yr)", flush=True)
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
                'total_trades': getattr(backtest, 'total_trades', 0),
                'rebalance_days': getattr(backtest, 'rebalance_count', 0),
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
            'equity_curve_daily': equity_curve_daily,
            'spy_curve_daily': spy_curve_daily,
            'asset_class_history': asset_class_history,
            'asset_class_daily': asset_class_daily,
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
            "rebalance_days": 0,
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
        "equity_curve_daily": [],
        "spy_curve_daily": [],
        "asset_class_history": [],
        "asset_class_daily": [],
        "generated_at": None
    }


def reload_backtest_results() -> dict:
    """Force reload of backtest results (clears cache and re-runs)"""
    global _backtest_cache, _backtest_cache_time
    _backtest_cache = None
    _backtest_cache_time = None
    return load_backtest_results()


# Cache for EMA window comparison
_ema_comparison_cache = None
_ema_comparison_cache_time = None


def run_ema_window_comparison() -> dict:
    """
    Run backtests with different EMA smoothing windows to find optimal period.
    Tests: 5, 10, 15, 20, 30, 50 day EMA windows.
    Results are cached for the same duration as regular backtest.
    """
    global _ema_comparison_cache, _ema_comparison_cache_time

    # Check cache
    if _ema_comparison_cache is not None and _ema_comparison_cache_time is not None:
        cache_age = datetime.now() - _ema_comparison_cache_time
        if cache_age < timedelta(hours=CACHE_DURATION_HOURS):
            return _ema_comparison_cache

    try:
        print("=" * 60, flush=True)
        print("RUNNING EMA WINDOW COMPARISON", flush=True)
        print("=" * 60, flush=True)

        from quad_portfolio_backtest import QuadrantPortfolioBacktest

        # Setup parameters
        INITIAL_CAPITAL = 50000
        BACKTEST_YEARS = 5
        MOMENTUM_DAYS = 50

        # EMA windows to test
        EMA_WINDOWS = [5, 10, 15, 20, 30, 50]

        end_date = datetime.now()
        start_date = end_date - timedelta(days=BACKTEST_YEARS * 365 + 100)

        results = []
        best_sharpe = -999
        best_window = None

        for i, ema_window in enumerate(EMA_WINDOWS):
            print(f"\n[{i+1}/{len(EMA_WINDOWS)}] Testing EMA window = {ema_window}...", flush=True)

            backtest = QuadrantPortfolioBacktest(
                start_date=start_date,
                end_date=end_date,
                initial_capital=INITIAL_CAPITAL,
                momentum_days=MOMENTUM_DAYS,
                max_positions=10,
                atr_stop_loss=2.0,
                atr_period=14,
                ema_smoothing_period=ema_window
            )
            bt_results = backtest.run_backtest()

            # Count regime changes
            regime_changes = 0
            if hasattr(backtest, 'quad_history') and backtest.quad_history is not None:
                prev_top2 = None
                for idx in backtest.quad_history.index:
                    current_top2 = (
                        backtest.quad_history.loc[idx, 'Top1'],
                        backtest.quad_history.loc[idx, 'Top2']
                    )
                    if prev_top2 is not None and current_top2 != prev_top2:
                        regime_changes += 1
                    prev_top2 = current_top2

            # Build equity curve
            equity_curve = []
            if backtest.portfolio_value is not None:
                for date, value in backtest.portfolio_value.items():
                    equity_curve.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'value': float(value)
                    })

            result = {
                'ema_window': ema_window,
                'total_return': bt_results.get('total_return', 0),
                'annual_return': bt_results.get('annual_return', 0),
                'sharpe': bt_results.get('sharpe', 0),
                'max_drawdown': bt_results.get('max_drawdown', 0),
                'volatility': bt_results.get('annual_vol', 0),
                'final_value': bt_results.get('final_value', INITIAL_CAPITAL),
                'regime_changes': regime_changes,
                'equity_curve': equity_curve,
            }
            results.append(result)

            if bt_results.get('sharpe', 0) > best_sharpe:
                best_sharpe = bt_results.get('sharpe', 0)
                best_window = ema_window

            print(f"  → Return: {bt_results.get('total_return', 0):.1f}%, "
                  f"Sharpe: {bt_results.get('sharpe', 0):.2f}, "
                  f"Regime Changes: {regime_changes}", flush=True)

        _ema_comparison_cache = {
            'results': results,
            'best_window': best_window,
            'best_sharpe': best_sharpe,
            'parameters': {
                'initial_capital': INITIAL_CAPITAL,
                'backtest_years': BACKTEST_YEARS,
                'momentum_days': MOMENTUM_DAYS,
                'windows_tested': EMA_WINDOWS,
            },
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        _ema_comparison_cache_time = datetime.now()

        print(f"\n✓ Comparison complete! Best EMA window: {best_window} (Sharpe: {best_sharpe:.2f})", flush=True)

        return _ema_comparison_cache

    except Exception as e:
        print(f"Error running EMA comparison: {e}", flush=True)
        import traceback
        traceback.print_exc()

        return {
            'results': [],
            'error': str(e),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }


# Cache for BTC digital assets framework
_btc_framework_cache = None
_btc_framework_cache_time = None


def run_btc_framework_backtest() -> dict:
    """
    Run BTC-only backtest based on quad framework.

    Allocation rules:
    - Q1 in top 2 + above EMA → 200% (Overweight)
    - Q1 in top 2 but below EMA → 50% (Neutral)
    - Q1 not in top 2 + above EMA → 0% (Underweight)
    - Q1 not in top 2 + below EMA → -25% (Short)
    """
    global _btc_framework_cache, _btc_framework_cache_time

    # Check cache
    if _btc_framework_cache is not None and _btc_framework_cache_time is not None:
        cache_age = datetime.now() - _btc_framework_cache_time
        if cache_age < timedelta(hours=CACHE_DURATION_HOURS):
            return _btc_framework_cache

    try:
        print("=" * 60, flush=True)
        print("RUNNING BTC DIGITAL ASSETS FRAMEWORK BACKTEST", flush=True)
        print("=" * 60, flush=True)

        import numpy as np
        import pandas as pd
        import yfinance as yf
        from config import QUAD_INDICATORS

        # Parameters
        INITIAL_CAPITAL = 10000
        BACKTEST_YEARS = 5
        MOMENTUM_DAYS = 50
        EMA_PERIOD = 50
        EMA_SMOOTHING = 20

        end_date = datetime.now()
        start_date = end_date - timedelta(days=BACKTEST_YEARS * 365 + 150)

        print(f"Fetching BTC and indicator data...", flush=True)

        # Fetch BTC data
        btc_data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(btc_data.columns, pd.MultiIndex):
            btc_close = btc_data['Close']['BTC-USD'] if 'BTC-USD' in btc_data['Close'].columns else btc_data['Close'].iloc[:, 0]
        else:
            btc_close = btc_data['Close']

        # Fetch all indicator tickers
        all_indicators = set()
        for indicators in QUAD_INDICATORS.values():
            all_indicators.update(indicators)

        indicator_data = {}
        for ticker in all_indicators:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if len(data) > 0:
                    if isinstance(data.columns, pd.MultiIndex):
                        indicator_data[ticker] = data['Close'].iloc[:, 0]
                    else:
                        indicator_data[ticker] = data['Close']
            except:
                pass

        indicator_df = pd.DataFrame(indicator_data)
        indicator_df = indicator_df.ffill().bfill()

        # Calculate momentum for quad scoring
        momentum = indicator_df.pct_change(MOMENTUM_DAYS)

        # Calculate quad scores
        quad_scores = pd.DataFrame(index=momentum.index)
        for quad, indicators in QUAD_INDICATORS.items():
            quad_tickers = [t for t in indicators if t in momentum.columns]
            if quad_tickers:
                quad_scores[quad] = momentum[quad_tickers].mean(axis=1)

        # Apply EMA smoothing to quad scores
        smoothed_scores = quad_scores.ewm(span=EMA_SMOOTHING, adjust=False).mean()

        # Calculate BTC EMA for trend filter
        btc_ema = btc_close.ewm(span=EMA_PERIOD, adjust=False).mean()

        # Align all data
        common_dates = btc_close.index.intersection(smoothed_scores.index).intersection(btc_ema.index)
        btc_close = btc_close.loc[common_dates]
        btc_ema = btc_ema.loc[common_dates]
        smoothed_scores = smoothed_scores.loc[common_dates]

        # Warmup period
        warmup = max(MOMENTUM_DAYS, EMA_PERIOD, EMA_SMOOTHING) + 10
        btc_close = btc_close.iloc[warmup:]
        btc_ema = btc_ema.iloc[warmup:]
        smoothed_scores = smoothed_scores.iloc[warmup:]

        print(f"Running strategy simulation...", flush=True)

        # Simulate strategy
        portfolio_value = pd.Series(index=btc_close.index, dtype=float)
        btc_buy_hold = pd.Series(index=btc_close.index, dtype=float)
        positions = pd.Series(index=btc_close.index, dtype=str)
        allocations = pd.Series(index=btc_close.index, dtype=float)

        cash = INITIAL_CAPITAL
        btc_holdings = 0
        buy_hold_btc = INITIAL_CAPITAL / btc_close.iloc[0]

        regime_history = []
        prev_allocation = 0

        for i, date in enumerate(btc_close.index):
            btc_price = btc_close.loc[date]
            ema_value = btc_ema.loc[date]
            scores = smoothed_scores.loc[date].sort_values(ascending=False)

            top1 = scores.index[0]
            top2 = scores.index[1]
            above_ema = btc_price > ema_value

            # Determine position
            if top1 == 'Q1':
                if above_ema:
                    position = 'Overweight'
                    target_allocation = 2.0  # 200%
                else:
                    position = 'Neutral'
                    target_allocation = 0.5  # 50%
            elif top2 == 'Q1':
                if above_ema:
                    position = 'Overweight'
                    target_allocation = 2.0  # 200%
                else:
                    position = 'Neutral'
                    target_allocation = 0.5  # 50%
            else:
                # Q1 not in top 2
                if above_ema:
                    position = 'Underweight'
                    target_allocation = 0.0
                else:
                    position = 'Short'
                    target_allocation = -0.25  # -25% (short)

            positions.loc[date] = position
            allocations.loc[date] = target_allocation

            # Rebalance if allocation changed
            if target_allocation != prev_allocation:
                # Calculate current portfolio value
                current_value = cash + btc_holdings * btc_price

                # Calculate target BTC value
                target_btc_value = current_value * target_allocation
                target_btc_holdings = target_btc_value / btc_price

                # Adjust holdings
                btc_holdings = target_btc_holdings
                cash = current_value - target_btc_value

                regime_history.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'position': position,
                    'allocation': target_allocation,
                    'btc_price': btc_price
                })

                prev_allocation = target_allocation

            # Calculate portfolio value
            portfolio_value.loc[date] = cash + btc_holdings * btc_price
            btc_buy_hold.loc[date] = buy_hold_btc * btc_price

        # Calculate performance metrics
        strategy_returns = portfolio_value.pct_change().dropna()
        buyhold_returns = btc_buy_hold.pct_change().dropna()

        # Strategy metrics
        total_return = (portfolio_value.iloc[-1] / INITIAL_CAPITAL - 1) * 100
        annual_return = ((portfolio_value.iloc[-1] / INITIAL_CAPITAL) ** (252 / len(portfolio_value)) - 1) * 100
        volatility = strategy_returns.std() * np.sqrt(252) * 100
        sharpe = (annual_return - 5) / volatility if volatility > 0 else 0  # Assuming 5% risk-free rate

        # Max drawdown
        rolling_max = portfolio_value.cummax()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        # Buy & hold metrics
        bh_total_return = (btc_buy_hold.iloc[-1] / INITIAL_CAPITAL - 1) * 100
        bh_annual_return = ((btc_buy_hold.iloc[-1] / INITIAL_CAPITAL) ** (252 / len(btc_buy_hold)) - 1) * 100
        bh_volatility = buyhold_returns.std() * np.sqrt(252) * 100
        bh_sharpe = (bh_annual_return - 5) / bh_volatility if bh_volatility > 0 else 0
        bh_rolling_max = btc_buy_hold.cummax()
        bh_drawdown = (btc_buy_hold - bh_rolling_max) / bh_rolling_max
        bh_max_drawdown = bh_drawdown.min() * 100

        # Build chart data
        chart_data = []
        for date in btc_close.index:
            chart_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'btc_price': float(btc_close.loc[date]),
                'ema': float(btc_ema.loc[date]),
                'position': positions.loc[date],
                'allocation': float(allocations.loc[date]),
                'portfolio_value': float(portfolio_value.loc[date]),
                'buyhold_value': float(btc_buy_hold.loc[date]),
            })

        # Position breakdown
        position_counts = positions.value_counts()
        position_pcts = {
            'Overweight': position_counts.get('Overweight', 0) / len(positions) * 100,
            'Neutral': position_counts.get('Neutral', 0) / len(positions) * 100,
            'Underweight': position_counts.get('Underweight', 0) / len(positions) * 100,
            'Short': position_counts.get('Short', 0) / len(positions) * 100,
        }

        _btc_framework_cache = {
            'strategy': {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'final_value': portfolio_value.iloc[-1],
            },
            'buyhold': {
                'total_return': bh_total_return,
                'annual_return': bh_annual_return,
                'sharpe': bh_sharpe,
                'max_drawdown': bh_max_drawdown,
                'volatility': bh_volatility,
                'final_value': btc_buy_hold.iloc[-1],
            },
            'current_position': positions.iloc[-1],
            'current_allocation': allocations.iloc[-1],
            'position_breakdown': position_pcts,
            'chart_data': chart_data,
            'regime_history': regime_history[-20:],  # Last 20 regime changes
            'parameters': {
                'initial_capital': INITIAL_CAPITAL,
                'backtest_years': BACKTEST_YEARS,
                'momentum_days': MOMENTUM_DAYS,
                'ema_period': EMA_PERIOD,
                'ema_smoothing': EMA_SMOOTHING,
            },
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        _btc_framework_cache_time = datetime.now()

        print(f"✓ BTC Framework complete!", flush=True)
        print(f"  Strategy: {total_return:.1f}% return, Sharpe {sharpe:.2f}", flush=True)
        print(f"  Buy&Hold: {bh_total_return:.1f}% return, Sharpe {bh_sharpe:.2f}", flush=True)

        return _btc_framework_cache

    except Exception as e:
        print(f"Error running BTC framework: {e}", flush=True)
        import traceback
        traceback.print_exc()

        return {
            'error': str(e),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }


# Cache for multi-asset volatility framework
_vol_framework_cache = None
_vol_framework_cache_time = None


def run_volatility_weighted_backtest() -> dict:
    """
    Run multi-asset (BTC/ETH/SOL) backtest with inverse volatility weighting.

    Uses the same quad framework signals as BTC-only, but splits exposure
    across BTC, ETH, SOL based on inverse volatility (lower vol = higher weight).

    This is a "volatility chasing" strategy in the sense that it dynamically
    rebalances to favor assets with lower recent volatility (risk parity style).
    """
    global _vol_framework_cache, _vol_framework_cache_time

    # Check cache
    if _vol_framework_cache is not None and _vol_framework_cache_time is not None:
        cache_age = datetime.now() - _vol_framework_cache_time
        if cache_age < timedelta(hours=CACHE_DURATION_HOURS):
            return _vol_framework_cache

    try:
        print("=" * 60, flush=True)
        print("RUNNING MULTI-ASSET VOLATILITY WEIGHTED BACKTEST", flush=True)
        print("=" * 60, flush=True)

        import numpy as np
        import pandas as pd
        import yfinance as yf
        from config import QUAD_INDICATORS

        # Parameters
        INITIAL_CAPITAL = 10000
        BACKTEST_YEARS = 5
        MOMENTUM_DAYS = 50
        EMA_PERIOD = 50
        EMA_SMOOTHING = 20
        VOL_LOOKBACK = 30  # Days for volatility calculation

        # Top 10 cryptos by market cap
        ASSETS = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BNB-USD', 'SOL-USD',
                  'TRX-USD', 'DOGE-USD', 'ADA-USD', 'BCH-USD', 'LINK-USD']
        ASSET_NAMES = {
            'BTC-USD': 'BTC', 'ETH-USD': 'ETH', 'XRP-USD': 'XRP', 'BNB-USD': 'BNB',
            'SOL-USD': 'SOL', 'TRX-USD': 'TRX', 'DOGE-USD': 'DOGE', 'ADA-USD': 'ADA',
            'BCH-USD': 'BCH', 'LINK-USD': 'LINK'
        }

        end_date = datetime.now()
        start_date = end_date - timedelta(days=BACKTEST_YEARS * 365 + 150)

        print(f"Fetching crypto and indicator data for {len(ASSETS)} assets...", flush=True)

        # Fetch all crypto data
        crypto_data = {}
        for ticker in ASSETS:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if len(data) > 0:
                if isinstance(data.columns, pd.MultiIndex):
                    crypto_data[ticker] = data['Close'].iloc[:, 0]
                else:
                    crypto_data[ticker] = data['Close']

        crypto_df = pd.DataFrame(crypto_data)
        crypto_df = crypto_df.ffill().bfill()

        # Calculate rolling volatility for each asset
        crypto_returns = crypto_df.pct_change()
        rolling_vol = crypto_returns.rolling(window=VOL_LOOKBACK).std() * np.sqrt(252)  # Annualized

        # Calculate inverse volatility weights
        def calc_inv_vol_weights(vol_row):
            """Calculate inverse volatility weights (lower vol = higher weight)"""
            valid_vols = vol_row.dropna()
            if len(valid_vols) == 0:
                return pd.Series({ticker: 1/len(ASSETS) for ticker in ASSETS})

            # Inverse vol
            inv_vol = 1 / valid_vols
            # Normalize to sum to 1
            weights = inv_vol / inv_vol.sum()

            # Fill missing with equal weight
            for ticker in ASSETS:
                if ticker not in weights.index or pd.isna(weights[ticker]):
                    weights[ticker] = 1 / len(ASSETS)

            return weights[ASSETS]

        inv_vol_weights = rolling_vol.apply(calc_inv_vol_weights, axis=1)

        # Fetch indicator data for quad scoring
        all_indicators = set()
        for indicators in QUAD_INDICATORS.values():
            all_indicators.update(indicators)

        indicator_data = {}
        for ticker in all_indicators:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if len(data) > 0:
                    if isinstance(data.columns, pd.MultiIndex):
                        indicator_data[ticker] = data['Close'].iloc[:, 0]
                    else:
                        indicator_data[ticker] = data['Close']
            except:
                pass

        indicator_df = pd.DataFrame(indicator_data)
        indicator_df = indicator_df.ffill().bfill()

        # Calculate momentum for quad scoring
        momentum = indicator_df.pct_change(MOMENTUM_DAYS)

        # Calculate quad scores
        quad_scores = pd.DataFrame(index=momentum.index)
        for quad, indicators in QUAD_INDICATORS.items():
            quad_tickers = [t for t in indicators if t in momentum.columns]
            if quad_tickers:
                quad_scores[quad] = momentum[quad_tickers].mean(axis=1)

        # Apply EMA smoothing
        smoothed_scores = quad_scores.ewm(span=EMA_SMOOTHING, adjust=False).mean()

        # Calculate EMA for each asset (for individual trend confirmation)
        btc_close = crypto_df['BTC-USD']
        asset_emas = {}
        for ticker in ASSETS:
            asset_emas[ticker] = crypto_df[ticker].ewm(span=EMA_PERIOD, adjust=False).mean()
        btc_ema = asset_emas['BTC-USD']  # For backward compatibility

        # Align all data
        common_dates = (btc_close.index
                       .intersection(smoothed_scores.index)
                       .intersection(btc_ema.index)
                       .intersection(inv_vol_weights.index)
                       .intersection(crypto_df.index))

        btc_close = btc_close.loc[common_dates]
        btc_ema = btc_ema.loc[common_dates]
        smoothed_scores = smoothed_scores.loc[common_dates]
        inv_vol_weights = inv_vol_weights.loc[common_dates]
        crypto_df = crypto_df.loc[common_dates]

        # Warmup period
        warmup = max(MOMENTUM_DAYS, EMA_PERIOD, EMA_SMOOTHING, VOL_LOOKBACK) + 10
        btc_close = btc_close.iloc[warmup:]
        btc_ema = btc_ema.iloc[warmup:]
        smoothed_scores = smoothed_scores.iloc[warmup:]
        inv_vol_weights = inv_vol_weights.iloc[warmup:]
        crypto_df = crypto_df.iloc[warmup:]

        print(f"Running multi-asset simulation...", flush=True)

        # Initialize tracking
        portfolio_value = pd.Series(index=btc_close.index, dtype=float)
        btc_buy_hold = pd.Series(index=btc_close.index, dtype=float)
        positions = pd.Series(index=btc_close.index, dtype=str)
        allocations = pd.Series(index=btc_close.index, dtype=float)

        # Track individual asset weights over time
        asset_weights_history = {ticker: pd.Series(index=btc_close.index, dtype=float) for ticker in ASSETS}
        asset_allocations_history = {ticker: pd.Series(index=btc_close.index, dtype=float) for ticker in ASSETS}
        vol_history = {ticker: pd.Series(index=btc_close.index, dtype=float) for ticker in ASSETS}

        # Portfolio state
        cash = INITIAL_CAPITAL
        holdings = {ticker: 0 for ticker in ASSETS}
        buy_hold_btc = INITIAL_CAPITAL / btc_close.iloc[0]

        regime_history = []
        prev_allocation = 0
        prev_weights = None

        for i, date in enumerate(btc_close.index):
            prices = {ticker: crypto_df.loc[date, ticker] for ticker in ASSETS}
            ema_values = {ticker: asset_emas[ticker].loc[date] for ticker in ASSETS}
            scores = smoothed_scores.loc[date].sort_values(ascending=False)
            base_weights = inv_vol_weights.loc[date]

            top1 = scores.index[0]
            top2 = scores.index[1]
            btc_above_ema = prices['BTC-USD'] > ema_values['BTC-USD']

            # Determine position based on BTC (same logic as BTC-only)
            if top1 == 'Q1' or top2 == 'Q1':
                if btc_above_ema:
                    position = 'Overweight'
                    target_allocation = 2.0
                else:
                    position = 'Neutral'
                    target_allocation = 0.5  # 50%
            else:
                if btc_above_ema:
                    position = 'Underweight'
                    target_allocation = 0.0
                else:
                    position = 'Short'
                    target_allocation = -0.25

            positions.loc[date] = position
            allocations.loc[date] = target_allocation

            # Apply individual EMA confirmations to each asset
            # For longs: only include assets above their EMA
            # For shorts: only include assets below their EMA
            confirmed_assets = []
            if target_allocation > 0:  # Long position
                confirmed_assets = [t for t in ASSETS if prices[t] > ema_values[t]]
            elif target_allocation < 0:  # Short position
                confirmed_assets = [t for t in ASSETS if prices[t] < ema_values[t]]

            # Calculate adjusted weights (redistribute among confirmed assets only)
            adjusted_weights = {t: 0.0 for t in ASSETS}
            if confirmed_assets:
                total_confirmed_weight = sum(base_weights[t] for t in confirmed_assets)
                if total_confirmed_weight > 0:
                    for t in confirmed_assets:
                        adjusted_weights[t] = base_weights[t] / total_confirmed_weight
            weights = pd.Series(adjusted_weights)

            # Store individual weights and vols
            for ticker in ASSETS:
                asset_weights_history[ticker].loc[date] = weights[ticker]
                asset_allocations_history[ticker].loc[date] = target_allocation * weights[ticker]
                vol_history[ticker].loc[date] = rolling_vol.loc[date, ticker] if date in rolling_vol.index else 0

            # Rebalance if allocation or weights changed significantly
            weights_changed = prev_weights is None or any(abs(weights[t] - prev_weights.get(t, 0)) > 0.05 for t in ASSETS)

            if target_allocation != prev_allocation or (target_allocation != 0 and weights_changed):
                # Calculate current portfolio value
                current_value = cash + sum(holdings[t] * prices[t] for t in ASSETS)

                # Calculate target holdings for each asset (only confirmed assets get allocation)
                for ticker in ASSETS:
                    target_asset_value = current_value * target_allocation * weights[ticker]
                    holdings[ticker] = target_asset_value / prices[ticker]

                cash = current_value - sum(holdings[t] * prices[t] for t in ASSETS)

                regime_history.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'position': position,
                    'allocation': target_allocation,
                    'weights': {ASSET_NAMES[t]: round(weights[t] * 100, 1) for t in ASSETS},
                    'confirmed': [ASSET_NAMES[t] for t in confirmed_assets],
                    'btc_price': prices['BTC-USD']
                })

                prev_allocation = target_allocation
                prev_weights = weights.to_dict()

            # Calculate portfolio value
            portfolio_value.loc[date] = cash + sum(holdings[t] * prices[t] for t in ASSETS)
            btc_buy_hold.loc[date] = buy_hold_btc * prices['BTC-USD']

        # Calculate performance metrics
        strategy_returns = portfolio_value.pct_change().dropna()
        buyhold_returns = btc_buy_hold.pct_change().dropna()

        # Strategy metrics
        total_return = (portfolio_value.iloc[-1] / INITIAL_CAPITAL - 1) * 100
        annual_return = ((portfolio_value.iloc[-1] / INITIAL_CAPITAL) ** (252 / len(portfolio_value)) - 1) * 100
        volatility = strategy_returns.std() * np.sqrt(252) * 100
        sharpe = (annual_return - 5) / volatility if volatility > 0 else 0

        rolling_max = portfolio_value.cummax()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        # Buy & hold metrics
        bh_total_return = (btc_buy_hold.iloc[-1] / INITIAL_CAPITAL - 1) * 100
        bh_annual_return = ((btc_buy_hold.iloc[-1] / INITIAL_CAPITAL) ** (252 / len(btc_buy_hold)) - 1) * 100
        bh_volatility = buyhold_returns.std() * np.sqrt(252) * 100
        bh_sharpe = (bh_annual_return - 5) / bh_volatility if bh_volatility > 0 else 0
        bh_rolling_max = btc_buy_hold.cummax()
        bh_max_drawdown = ((btc_buy_hold - bh_rolling_max) / bh_rolling_max).min() * 100

        # Build chart data
        chart_data = []
        for date in btc_close.index:
            chart_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'btc_price': float(crypto_df.loc[date, 'BTC-USD']),
                'eth_price': float(crypto_df.loc[date, 'ETH-USD']),
                'sol_price': float(crypto_df.loc[date, 'SOL-USD']),
                'ema': float(btc_ema.loc[date]),
                'position': positions.loc[date],
                'allocation': float(allocations.loc[date]),
                'portfolio_value': float(portfolio_value.loc[date]),
                'buyhold_value': float(btc_buy_hold.loc[date]),
                'btc_weight': float(asset_weights_history['BTC-USD'].loc[date]),
                'eth_weight': float(asset_weights_history['ETH-USD'].loc[date]),
                'sol_weight': float(asset_weights_history['SOL-USD'].loc[date]),
                'btc_alloc': float(asset_allocations_history['BTC-USD'].loc[date]),
                'eth_alloc': float(asset_allocations_history['ETH-USD'].loc[date]),
                'sol_alloc': float(asset_allocations_history['SOL-USD'].loc[date]),
                'btc_vol': float(vol_history['BTC-USD'].loc[date]) * 100 if not pd.isna(vol_history['BTC-USD'].loc[date]) else 0,
                'eth_vol': float(vol_history['ETH-USD'].loc[date]) * 100 if not pd.isna(vol_history['ETH-USD'].loc[date]) else 0,
                'sol_vol': float(vol_history['SOL-USD'].loc[date]) * 100 if not pd.isna(vol_history['SOL-USD'].loc[date]) else 0,
            })

        # Position breakdown
        position_counts = positions.value_counts()
        position_pcts = {
            'Overweight': position_counts.get('Overweight', 0) / len(positions) * 100,
            'Neutral': position_counts.get('Neutral', 0) / len(positions) * 100,
            'Underweight': position_counts.get('Underweight', 0) / len(positions) * 100,
            'Short': position_counts.get('Short', 0) / len(positions) * 100,
        }

        # Current weights
        current_weights = {
            'BTC': asset_weights_history['BTC-USD'].iloc[-1] * 100,
            'ETH': asset_weights_history['ETH-USD'].iloc[-1] * 100,
            'SOL': asset_weights_history['SOL-USD'].iloc[-1] * 100,
        }

        # Current volatilities
        current_vols = {
            'BTC': vol_history['BTC-USD'].iloc[-1] * 100 if not pd.isna(vol_history['BTC-USD'].iloc[-1]) else 0,
            'ETH': vol_history['ETH-USD'].iloc[-1] * 100 if not pd.isna(vol_history['ETH-USD'].iloc[-1]) else 0,
            'SOL': vol_history['SOL-USD'].iloc[-1] * 100 if not pd.isna(vol_history['SOL-USD'].iloc[-1]) else 0,
        }

        _vol_framework_cache = {
            'strategy': {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'final_value': portfolio_value.iloc[-1],
            },
            'buyhold': {
                'total_return': bh_total_return,
                'annual_return': bh_annual_return,
                'sharpe': bh_sharpe,
                'max_drawdown': bh_max_drawdown,
                'volatility': bh_volatility,
                'final_value': btc_buy_hold.iloc[-1],
            },
            'current_position': positions.iloc[-1],
            'current_allocation': allocations.iloc[-1],
            'current_weights': current_weights,
            'current_vols': current_vols,
            'position_breakdown': position_pcts,
            'chart_data': chart_data,
            'regime_history': regime_history[-20:],
            'parameters': {
                'initial_capital': INITIAL_CAPITAL,
                'backtest_years': BACKTEST_YEARS,
                'momentum_days': MOMENTUM_DAYS,
                'ema_period': EMA_PERIOD,
                'ema_smoothing': EMA_SMOOTHING,
                'vol_lookback': VOL_LOOKBACK,
                'assets': list(ASSET_NAMES.values()),
            },
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        _vol_framework_cache_time = datetime.now()

        print(f"✓ Multi-Asset Vol Framework complete!", flush=True)
        print(f"  Strategy: {total_return:.1f}% return, Sharpe {sharpe:.2f}", flush=True)
        print(f"  Current weights: BTC {current_weights['BTC']:.1f}%, ETH {current_weights['ETH']:.1f}%, SOL {current_weights['SOL']:.1f}%", flush=True)

        return _vol_framework_cache

    except Exception as e:
        print(f"Error running vol framework: {e}", flush=True)
        import traceback
        traceback.print_exc()

        return {
            'error': str(e),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }


# Cache for volatility-weighted (high vol = high weight) framework
_vol_chase_cache = None
_vol_chase_cache_time = None


def run_volatility_chase_backtest() -> dict:
    """
    Run multi-asset (BTC/ETH/SOL) backtest with volatility weighting.

    Uses the same quad framework signals as BTC-only, but splits exposure
    across BTC, ETH, SOL based on volatility (higher vol = higher weight).

    This is a "volatility chasing" strategy that allocates more to
    assets with higher recent volatility (momentum/trend following style).
    """
    global _vol_chase_cache, _vol_chase_cache_time

    # Check cache
    if _vol_chase_cache is not None and _vol_chase_cache_time is not None:
        cache_age = datetime.now() - _vol_chase_cache_time
        if cache_age < timedelta(hours=CACHE_DURATION_HOURS):
            return _vol_chase_cache

    try:
        print("=" * 60, flush=True)
        print("RUNNING VOLATILITY CHASE (HIGH VOL) BACKTEST", flush=True)
        print("=" * 60, flush=True)

        import numpy as np
        import pandas as pd
        import yfinance as yf
        from config import QUAD_INDICATORS

        # Parameters
        INITIAL_CAPITAL = 10000
        BACKTEST_YEARS = 5
        MOMENTUM_DAYS = 50
        EMA_PERIOD = 50
        EMA_SMOOTHING = 20
        VOL_LOOKBACK = 30

        # Top 10 cryptos by market cap
        ASSETS = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BNB-USD', 'SOL-USD',
                  'TRX-USD', 'DOGE-USD', 'ADA-USD', 'BCH-USD', 'LINK-USD']
        ASSET_NAMES = {
            'BTC-USD': 'BTC', 'ETH-USD': 'ETH', 'XRP-USD': 'XRP', 'BNB-USD': 'BNB',
            'SOL-USD': 'SOL', 'TRX-USD': 'TRX', 'DOGE-USD': 'DOGE', 'ADA-USD': 'ADA',
            'BCH-USD': 'BCH', 'LINK-USD': 'LINK'
        }

        end_date = datetime.now()
        start_date = end_date - timedelta(days=BACKTEST_YEARS * 365 + 150)

        print(f"Fetching crypto and indicator data for {len(ASSETS)} assets...", flush=True)

        # Fetch all crypto data
        crypto_data = {}
        for ticker in ASSETS:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if len(data) > 0:
                if isinstance(data.columns, pd.MultiIndex):
                    crypto_data[ticker] = data['Close'].iloc[:, 0]
                else:
                    crypto_data[ticker] = data['Close']

        crypto_df = pd.DataFrame(crypto_data)
        crypto_df = crypto_df.ffill().bfill()

        # Calculate rolling volatility for each asset
        crypto_returns = crypto_df.pct_change()
        rolling_vol = crypto_returns.rolling(window=VOL_LOOKBACK).std() * np.sqrt(252)

        # Calculate volatility weights (higher vol = higher weight)
        def calc_vol_weights(vol_row):
            """Calculate volatility weights (higher vol = higher weight)"""
            valid_vols = vol_row.dropna()
            if len(valid_vols) == 0:
                return pd.Series({ticker: 1/len(ASSETS) for ticker in ASSETS})

            # Normalize to sum to 1
            weights = valid_vols / valid_vols.sum()

            # Fill missing with equal weight
            for ticker in ASSETS:
                if ticker not in weights.index or pd.isna(weights[ticker]):
                    weights[ticker] = 1 / len(ASSETS)

            return weights[ASSETS]

        vol_weights = rolling_vol.apply(calc_vol_weights, axis=1)

        # Fetch indicator data for quad scoring
        all_indicators = set()
        for indicators in QUAD_INDICATORS.values():
            all_indicators.update(indicators)

        indicator_data = {}
        for ticker in all_indicators:
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                if len(data) > 0:
                    if isinstance(data.columns, pd.MultiIndex):
                        indicator_data[ticker] = data['Close'].iloc[:, 0]
                    else:
                        indicator_data[ticker] = data['Close']
            except:
                pass

        indicator_df = pd.DataFrame(indicator_data)
        indicator_df = indicator_df.ffill().bfill()

        # Calculate momentum for quad scoring
        momentum = indicator_df.pct_change(MOMENTUM_DAYS)

        # Calculate quad scores
        quad_scores = pd.DataFrame(index=momentum.index)
        for quad, indicators in QUAD_INDICATORS.items():
            quad_tickers = [t for t in indicators if t in momentum.columns]
            if quad_tickers:
                quad_scores[quad] = momentum[quad_tickers].mean(axis=1)

        # Apply EMA smoothing
        smoothed_scores = quad_scores.ewm(span=EMA_SMOOTHING, adjust=False).mean()

        # Calculate EMA for each asset (for individual trend confirmation)
        btc_close = crypto_df['BTC-USD']
        asset_emas = {}
        for ticker in ASSETS:
            asset_emas[ticker] = crypto_df[ticker].ewm(span=EMA_PERIOD, adjust=False).mean()
        btc_ema = asset_emas['BTC-USD']  # For backward compatibility

        # Align all data
        common_dates = (btc_close.index
                       .intersection(smoothed_scores.index)
                       .intersection(btc_ema.index)
                       .intersection(vol_weights.index)
                       .intersection(crypto_df.index))

        btc_close = btc_close.loc[common_dates]
        btc_ema = btc_ema.loc[common_dates]
        smoothed_scores = smoothed_scores.loc[common_dates]
        vol_weights = vol_weights.loc[common_dates]
        crypto_df = crypto_df.loc[common_dates]

        # Warmup period
        warmup = max(MOMENTUM_DAYS, EMA_PERIOD, EMA_SMOOTHING, VOL_LOOKBACK) + 10
        btc_close = btc_close.iloc[warmup:]
        btc_ema = btc_ema.iloc[warmup:]
        smoothed_scores = smoothed_scores.iloc[warmup:]
        vol_weights = vol_weights.iloc[warmup:]
        crypto_df = crypto_df.iloc[warmup:]

        print(f"Running volatility chase simulation...", flush=True)

        # Initialize tracking
        portfolio_value = pd.Series(index=btc_close.index, dtype=float)
        btc_buy_hold = pd.Series(index=btc_close.index, dtype=float)
        positions = pd.Series(index=btc_close.index, dtype=str)
        allocations = pd.Series(index=btc_close.index, dtype=float)

        # Track individual asset weights over time
        asset_weights_history = {ticker: pd.Series(index=btc_close.index, dtype=float) for ticker in ASSETS}
        asset_allocations_history = {ticker: pd.Series(index=btc_close.index, dtype=float) for ticker in ASSETS}
        vol_history = {ticker: pd.Series(index=btc_close.index, dtype=float) for ticker in ASSETS}

        # Portfolio state
        cash = INITIAL_CAPITAL
        holdings = {ticker: 0 for ticker in ASSETS}
        buy_hold_btc = INITIAL_CAPITAL / btc_close.iloc[0]

        regime_history = []
        prev_allocation = 0
        prev_weights = None

        for i, date in enumerate(btc_close.index):
            prices = {ticker: crypto_df.loc[date, ticker] for ticker in ASSETS}
            ema_values = {ticker: asset_emas[ticker].loc[date] for ticker in ASSETS}
            scores = smoothed_scores.loc[date].sort_values(ascending=False)
            base_weights = vol_weights.loc[date]

            top1 = scores.index[0]
            top2 = scores.index[1]
            btc_above_ema = prices['BTC-USD'] > ema_values['BTC-USD']

            # Determine position based on BTC (same logic as BTC-only)
            if top1 == 'Q1' or top2 == 'Q1':
                if btc_above_ema:
                    position = 'Overweight'
                    target_allocation = 2.0
                else:
                    position = 'Neutral'
                    target_allocation = 0.5  # 50%
            else:
                if btc_above_ema:
                    position = 'Underweight'
                    target_allocation = 0.0
                else:
                    position = 'Short'
                    target_allocation = -0.25

            positions.loc[date] = position
            allocations.loc[date] = target_allocation

            # Apply individual EMA confirmations to each asset
            # For longs: only include assets above their EMA
            # For shorts: only include assets below their EMA
            confirmed_assets = []
            if target_allocation > 0:  # Long position
                confirmed_assets = [t for t in ASSETS if prices[t] > ema_values[t]]
            elif target_allocation < 0:  # Short position
                confirmed_assets = [t for t in ASSETS if prices[t] < ema_values[t]]

            # Calculate adjusted weights (redistribute among confirmed assets only)
            adjusted_weights = {t: 0.0 for t in ASSETS}
            if confirmed_assets:
                total_confirmed_weight = sum(base_weights[t] for t in confirmed_assets)
                if total_confirmed_weight > 0:
                    for t in confirmed_assets:
                        adjusted_weights[t] = base_weights[t] / total_confirmed_weight
            weights = pd.Series(adjusted_weights)

            # Store individual weights and vols
            for ticker in ASSETS:
                asset_weights_history[ticker].loc[date] = weights[ticker]
                asset_allocations_history[ticker].loc[date] = target_allocation * weights[ticker]
                vol_history[ticker].loc[date] = rolling_vol.loc[date, ticker] if date in rolling_vol.index else 0

            # Rebalance if allocation or weights changed significantly
            weights_changed = prev_weights is None or any(abs(weights[t] - prev_weights.get(t, 0)) > 0.05 for t in ASSETS)

            if target_allocation != prev_allocation or (target_allocation != 0 and weights_changed):
                # Calculate current portfolio value
                current_value = cash + sum(holdings[t] * prices[t] for t in ASSETS)

                # Calculate target holdings for each asset (only confirmed assets get allocation)
                for ticker in ASSETS:
                    target_asset_value = current_value * target_allocation * weights[ticker]
                    holdings[ticker] = target_asset_value / prices[ticker]

                cash = current_value - sum(holdings[t] * prices[t] for t in ASSETS)

                regime_history.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'position': position,
                    'allocation': target_allocation,
                    'weights': {ASSET_NAMES[t]: round(weights[t] * 100, 1) for t in ASSETS},
                    'confirmed': [ASSET_NAMES[t] for t in confirmed_assets],
                    'btc_price': prices['BTC-USD']
                })

                prev_allocation = target_allocation
                prev_weights = weights.to_dict()

            # Calculate portfolio value
            portfolio_value.loc[date] = cash + sum(holdings[t] * prices[t] for t in ASSETS)
            btc_buy_hold.loc[date] = buy_hold_btc * prices['BTC-USD']

        # Calculate performance metrics
        strategy_returns = portfolio_value.pct_change().dropna()
        buyhold_returns = btc_buy_hold.pct_change().dropna()

        # Strategy metrics
        total_return = (portfolio_value.iloc[-1] / INITIAL_CAPITAL - 1) * 100
        annual_return = ((portfolio_value.iloc[-1] / INITIAL_CAPITAL) ** (252 / len(portfolio_value)) - 1) * 100
        volatility = strategy_returns.std() * np.sqrt(252) * 100
        sharpe = (annual_return - 5) / volatility if volatility > 0 else 0

        rolling_max = portfolio_value.cummax()
        drawdown = (portfolio_value - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100

        # Buy & hold metrics
        bh_total_return = (btc_buy_hold.iloc[-1] / INITIAL_CAPITAL - 1) * 100
        bh_annual_return = ((btc_buy_hold.iloc[-1] / INITIAL_CAPITAL) ** (252 / len(btc_buy_hold)) - 1) * 100
        bh_volatility = buyhold_returns.std() * np.sqrt(252) * 100
        bh_sharpe = (bh_annual_return - 5) / bh_volatility if bh_volatility > 0 else 0
        bh_rolling_max = btc_buy_hold.cummax()
        bh_max_drawdown = ((btc_buy_hold - bh_rolling_max) / bh_rolling_max).min() * 100

        # Build chart data
        chart_data = []
        for date in btc_close.index:
            chart_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'btc_price': float(crypto_df.loc[date, 'BTC-USD']),
                'eth_price': float(crypto_df.loc[date, 'ETH-USD']),
                'sol_price': float(crypto_df.loc[date, 'SOL-USD']),
                'ema': float(btc_ema.loc[date]),
                'position': positions.loc[date],
                'allocation': float(allocations.loc[date]),
                'portfolio_value': float(portfolio_value.loc[date]),
                'buyhold_value': float(btc_buy_hold.loc[date]),
                'btc_weight': float(asset_weights_history['BTC-USD'].loc[date]),
                'eth_weight': float(asset_weights_history['ETH-USD'].loc[date]),
                'sol_weight': float(asset_weights_history['SOL-USD'].loc[date]),
                'btc_alloc': float(asset_allocations_history['BTC-USD'].loc[date]),
                'eth_alloc': float(asset_allocations_history['ETH-USD'].loc[date]),
                'sol_alloc': float(asset_allocations_history['SOL-USD'].loc[date]),
                'btc_vol': float(vol_history['BTC-USD'].loc[date]) * 100 if not pd.isna(vol_history['BTC-USD'].loc[date]) else 0,
                'eth_vol': float(vol_history['ETH-USD'].loc[date]) * 100 if not pd.isna(vol_history['ETH-USD'].loc[date]) else 0,
                'sol_vol': float(vol_history['SOL-USD'].loc[date]) * 100 if not pd.isna(vol_history['SOL-USD'].loc[date]) else 0,
            })

        # Position breakdown
        position_counts = positions.value_counts()
        position_pcts = {
            'Overweight': position_counts.get('Overweight', 0) / len(positions) * 100,
            'Neutral': position_counts.get('Neutral', 0) / len(positions) * 100,
            'Underweight': position_counts.get('Underweight', 0) / len(positions) * 100,
            'Short': position_counts.get('Short', 0) / len(positions) * 100,
        }

        # Current weights
        current_weights = {
            'BTC': asset_weights_history['BTC-USD'].iloc[-1] * 100,
            'ETH': asset_weights_history['ETH-USD'].iloc[-1] * 100,
            'SOL': asset_weights_history['SOL-USD'].iloc[-1] * 100,
        }

        # Current volatilities
        current_vols = {
            'BTC': vol_history['BTC-USD'].iloc[-1] * 100 if not pd.isna(vol_history['BTC-USD'].iloc[-1]) else 0,
            'ETH': vol_history['ETH-USD'].iloc[-1] * 100 if not pd.isna(vol_history['ETH-USD'].iloc[-1]) else 0,
            'SOL': vol_history['SOL-USD'].iloc[-1] * 100 if not pd.isna(vol_history['SOL-USD'].iloc[-1]) else 0,
        }

        _vol_chase_cache = {
            'strategy': {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe': sharpe,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'final_value': portfolio_value.iloc[-1],
            },
            'buyhold': {
                'total_return': bh_total_return,
                'annual_return': bh_annual_return,
                'sharpe': bh_sharpe,
                'max_drawdown': bh_max_drawdown,
                'volatility': bh_volatility,
                'final_value': btc_buy_hold.iloc[-1],
            },
            'current_position': positions.iloc[-1],
            'current_allocation': allocations.iloc[-1],
            'current_weights': current_weights,
            'current_vols': current_vols,
            'position_breakdown': position_pcts,
            'chart_data': chart_data,
            'regime_history': regime_history[-20:],
            'parameters': {
                'initial_capital': INITIAL_CAPITAL,
                'backtest_years': BACKTEST_YEARS,
                'momentum_days': MOMENTUM_DAYS,
                'ema_period': EMA_PERIOD,
                'ema_smoothing': EMA_SMOOTHING,
                'vol_lookback': VOL_LOOKBACK,
                'assets': list(ASSET_NAMES.values()),
            },
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        _vol_chase_cache_time = datetime.now()

        print(f"✓ Volatility Chase Framework complete!", flush=True)
        print(f"  Strategy: {total_return:.1f}% return, Sharpe {sharpe:.2f}", flush=True)
        print(f"  Current weights: BTC {current_weights['BTC']:.1f}%, ETH {current_weights['ETH']:.1f}%, SOL {current_weights['SOL']:.1f}%", flush=True)

        return _vol_chase_cache

    except Exception as e:
        print(f"Error running vol chase framework: {e}", flush=True)
        import traceback
        traceback.print_exc()

        return {
            'error': str(e),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
