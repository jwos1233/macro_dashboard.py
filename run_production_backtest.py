"""
Run Production v3.0 Backtest - Top 10 + ATR 2.0x
"""
from quad_portfolio_backtest import QuadrantPortfolioBacktest
from datetime import datetime, timedelta

# Setup
INITIAL_CAPITAL = 50000
BACKTEST_YEARS = 5
end_date = datetime.now()
start_date = end_date - timedelta(days=BACKTEST_YEARS * 365 + 100)

print("="*70)
print("PRODUCTION v3.0: TOP 10 + ATR 2.0x STOP LOSS")
print("="*70)
print(f"Initial Capital: ${INITIAL_CAPITAL:,}")
print(f"Momentum Lookback: 20 days")
print(f"Max Positions: 10")
print(f"Stop Loss: 2.0x ATR (14-day)")
print(f"Period: ~{BACKTEST_YEARS} years")
print("="*70)
print()

# Run production strategy
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

# Print summary
print("\n" + "="*70)
print("PRODUCTION v3.0 PERFORMANCE")
print("="*70)
print(f"Total Return:      {results['total_return']:.2f}%")
print(f"Annualized Return: {results['annual_return']:.2f}%")
print(f"Sharpe Ratio:      {results['sharpe']:.2f}")
print(f"Max Drawdown:      {results['max_drawdown']:.2f}%")
print(f"Volatility:        {results['annual_vol']:.2f}%")
print(f"Final Value:       ${results['final_value']:,.2f}")

print("="*70)

# Show comparison to buy & hold SPY
backtest.print_spy_comparison()

# Annual breakdown
backtest.print_annual_breakdown()

# Export to history.json for dashboard FIRST (before plot which may fail in CI)
print("\nExporting to dashboard...")

# Build performance history (daily)
portfolio_value = backtest.portfolio_value
performance_history = []

for date in portfolio_value.index:
    value = portfolio_value.loc[date]
    returns = (value / INITIAL_CAPITAL - 1) * 100
    performance_history.append({
        'date': date.strftime('%Y-%m-%d'),
        'value': round(float(value), 2),
        'totalReturn': round(float(returns), 2),
    })

# Build history data
history_data = {
    'events': [],  # Could extract regime changes here if needed
    'performance': performance_history,
    'summary': {
        'totalReturn': round(results['total_return'], 2),
        'annualReturn': round(results['annual_return'], 2),
        'sharpe': round(results['sharpe'], 2),
        'maxDrawdown': round(results['max_drawdown'], 2),
        'finalValue': round(results['final_value'], 2),
    },
    'generatedAt': datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ'),
}

# Write to dashboard data folder
output_path = Path(__file__).parent / 'apps' / 'web' / 'data' / 'history.json'
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(history_data, f, indent=2)

print(f"✅ Exported to {output_path}")

# Plot results (optional - may fail in headless CI environment)
try:
    print("\nGenerating P/L Chart...")
    backtest.plot_results()
    print("\n" + "="*70)
    print("Chart displayed! Close the chart window to continue.")
    print("="*70)
except Exception as e:
    print(f"\nSkipping chart display (headless mode): {e}")
