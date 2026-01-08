"""
Test: EMA-Smoothed Quad Scores Backtest
=======================================

Tests the hypothesis that smoothing quad scores with an EMA
before determining top quadrants improves performance by reducing whipsaws.

Run this script and compare results with run_production_backtest.py
"""

from quad_portfolio_backtest import QuadrantPortfolioBacktest
from config import QUAD_INDICATORS  # Use config.py to stay in sync
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np


class EMASmoothedQuadBacktest(QuadrantPortfolioBacktest):
    """Backtest with EMA-smoothed quad scores to reduce whipsaws"""

    def __init__(self, *args, ema_smoothing_period=20, **kwargs):
        """
        Args:
            ema_smoothing_period: Period for EMA smoothing of quad scores (default 20)
            *args, **kwargs: Same as QuadrantPortfolioBacktest
        """
        super().__init__(*args, **kwargs)
        self.ema_smoothing_period = ema_smoothing_period

    def calculate_quad_scores(self):
        """
        Calculate quad scores using QUAD_INDICATORS (same as signal_generator.py)

        For each date, calculates momentum for each indicator, then averages.
        """
        print(f"\nCalculating {self.momentum_days}-day momentum scores (using QUAD_INDICATORS)...")

        # Calculate momentum for all assets (as decimal, not percentage)
        momentum = self.price_data.pct_change(self.momentum_days)

        # Score each quadrant using QUAD_INDICATORS
        quad_scores = pd.DataFrame(index=momentum.index)

        for quad, indicators in QUAD_INDICATORS.items():
            quad_score_series = pd.Series(index=momentum.index, dtype=float)

            for date_idx in momentum.index:
                quad_scores_list = []
                for ticker in indicators:
                    if ticker in momentum.columns:
                        ticker_momentum = momentum.loc[date_idx, ticker]
                        if pd.notna(ticker_momentum):
                            quad_scores_list.append(ticker_momentum)

                if quad_scores_list:
                    quad_score_series.loc[date_idx] = np.mean(quad_scores_list)
                else:
                    quad_score_series.loc[date_idx] = 0.0

            quad_scores[quad] = quad_score_series

        return quad_scores

    def determine_top_quads(self, quad_scores):
        """
        Determine top 2 quadrants using EMA-smoothed scores

        Smooths raw quad scores with EMA to reduce noise and whipsaws.
        """
        print(f"\nApplying {self.ema_smoothing_period}-period EMA smoothing to quad scores...")

        # Apply EMA smoothing to each quadrant's score series
        smoothed_scores = pd.DataFrame(index=quad_scores.index, columns=quad_scores.columns)
        for quad in quad_scores.columns:
            smoothed_scores[quad] = quad_scores[quad].ewm(
                span=self.ema_smoothing_period,
                adjust=False
            ).mean()

        # Determine top quads based on smoothed scores
        top_quads = pd.DataFrame(index=smoothed_scores.index)

        for date_idx in smoothed_scores.index:
            scores = smoothed_scores.loc[date_idx].sort_values(ascending=False)
            top_quads.loc[date_idx, 'Top1'] = scores.index[0]
            top_quads.loc[date_idx, 'Top2'] = scores.index[1]
            top_quads.loc[date_idx, 'Score1'] = scores.iloc[0]
            top_quads.loc[date_idx, 'Score2'] = scores.iloc[1]

        # Store for analysis
        self.smoothed_quad_scores = smoothed_scores
        self.raw_quad_scores = quad_scores

        # Count regime changes to measure whipsaw reduction
        regime_changes = 0
        prev_top2 = None
        for date_idx in top_quads.index:
            current_top2 = (top_quads.loc[date_idx, 'Top1'], top_quads.loc[date_idx, 'Top2'])
            if prev_top2 is not None and current_top2 != prev_top2:
                regime_changes += 1
            prev_top2 = current_top2

        self.regime_changes = regime_changes
        print(f"✓ Smoothed scores calculated ({regime_changes} regime changes)")

        return top_quads


def run_comparison():
    """Run both standard and EMA-smoothed backtests for comparison"""

    # Setup - match production parameters
    INITIAL_CAPITAL = 50000
    BACKTEST_YEARS = 5
    MOMENTUM_DAYS = 50  # Match production
    EMA_SMOOTHING_PERIOD = 20

    today = date.today()
    end_date = datetime.combine(today - timedelta(days=1), datetime.min.time())
    start_date = end_date - timedelta(days=BACKTEST_YEARS * 365 + 100)

    print("="*70)
    print("EMA-SMOOTHED QUAD SCORES BACKTEST - TEST")
    print("="*70)
    print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"Momentum Lookback: {MOMENTUM_DAYS} days")
    print(f"EMA Smoothing Period: {EMA_SMOOTHING_PERIOD} days")
    print(f"Max Positions: 10")
    print(f"Stop Loss: 2.0x ATR (14-day)")
    print(f"Period: ~{BACKTEST_YEARS} years")
    print()
    print("Hypothesis: Smoothing quad scores reduces whipsaws and improves returns")
    print("="*70)

    # Run EMA-smoothed strategy
    print("\n" + "="*70)
    print("RUNNING EMA-SMOOTHED BACKTEST...")
    print("="*70)

    smoothed_backtest = EMASmoothedQuadBacktest(
        start_date=start_date,
        end_date=end_date,
        initial_capital=INITIAL_CAPITAL,
        momentum_days=MOMENTUM_DAYS,
        max_positions=10,
        atr_stop_loss=2.0,
        atr_period=14,
        ema_smoothing_period=EMA_SMOOTHING_PERIOD
    )

    smoothed_results = smoothed_backtest.run_backtest()
    smoothed_regime_changes = getattr(smoothed_backtest, 'regime_changes', 'N/A')

    # Run standard strategy for comparison
    print("\n" + "="*70)
    print("RUNNING STANDARD BACKTEST (no smoothing)...")
    print("="*70)

    standard_backtest = QuadrantPortfolioBacktest(
        start_date=start_date,
        end_date=end_date,
        initial_capital=INITIAL_CAPITAL,
        momentum_days=MOMENTUM_DAYS,
        max_positions=10,
        atr_stop_loss=2.0,
        atr_period=14
    )

    standard_results = standard_backtest.run_backtest()

    # Count regime changes in standard backtest
    standard_regime_changes = 0
    if hasattr(standard_backtest, 'quad_history') and standard_backtest.quad_history is not None:
        prev_top2 = None
        for date_idx in standard_backtest.quad_history.index:
            current_top2 = (
                standard_backtest.quad_history.loc[date_idx, 'Top1'],
                standard_backtest.quad_history.loc[date_idx, 'Top2']
            )
            if prev_top2 is not None and current_top2 != prev_top2:
                standard_regime_changes += 1
            prev_top2 = current_top2

    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON: EMA-SMOOTHED vs STANDARD")
    print("="*70)
    print(f"{'Metric':<25} {'Standard':>15} {'EMA-Smoothed':>15} {'Difference':>15}")
    print("-"*70)

    metrics = [
        ('Total Return %', 'total_return'),
        ('Annualized Return %', 'annual_return'),
        ('Sharpe Ratio', 'sharpe'),
        ('Max Drawdown %', 'max_drawdown'),
        ('Volatility %', 'annual_vol'),
    ]

    for label, key in metrics:
        std_val = standard_results[key]
        smooth_val = smoothed_results[key]
        diff = smooth_val - std_val
        diff_str = f"+{diff:.2f}" if diff > 0 else f"{diff:.2f}"
        print(f"{label:<25} {std_val:>15.2f} {smooth_val:>15.2f} {diff_str:>15}")

    print("-"*70)
    print(f"{'Regime Changes':<25} {standard_regime_changes:>15} {smoothed_regime_changes:>15} {smoothed_regime_changes - standard_regime_changes:>+15}")
    print(f"{'Final Value $':<25} {standard_results['final_value']:>15,.0f} {smoothed_results['final_value']:>15,.0f}")
    print("="*70)

    # Verdict
    print("\nVERDICT:")
    if smoothed_results['sharpe'] > standard_results['sharpe']:
        print("  ✓ EMA smoothing IMPROVED risk-adjusted returns (higher Sharpe)")
    else:
        print("  ✗ EMA smoothing REDUCED risk-adjusted returns (lower Sharpe)")

    if smoothed_regime_changes < standard_regime_changes:
        reduction = (1 - smoothed_regime_changes / standard_regime_changes) * 100
        print(f"  ✓ Regime changes reduced by {reduction:.0f}%")

    print("\n" + "="*70)

    # Plot smoothed results
    print("\nGenerating P/L Chart for EMA-Smoothed strategy...")
    smoothed_backtest.plot_results()


if __name__ == "__main__":
    run_comparison()
