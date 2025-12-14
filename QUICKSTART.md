# Quick Start Guide

Get the backtest engine running in 5 minutes!

## Prerequisites

- Python 3.10+
- pip
- Active API credentials for at least one exchange

## Step 1: Clone & Setup (2 minutes)

```bash
# Clone repository
git clone https://github.com/yourusername/crypto-trading-backtest-engine.git
cd crypto-trading-backtest-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Configure Credentials (1 minute)

```bash
# Create config directory
mkdir -p config

# Create .env file
cat > config/.env << 'EOF'
# Add your exchange API credentials
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
EOF
```

## Step 3: Run Your First Backtest (2 minutes)

Create a file named `run_backtest.py`:

```python
from exchange_manager import ExchangeManager
from backtest_strategies import BacktestEngine
from config import StrategyConfig, GlobalConfig
from datetime import datetime
import logging

# Setup
logging.basicConfig(level=logging.INFO)
GlobalConfig.create_directories()

# Initialize
ex_mgr = ExchangeManager()

print("Step 1: Validating Exchange Connection...")
status = ex_mgr.validate_connectivity("BINANCE")
print(f"  Status: {status['status']}")

print("\nStep 2: Fetching Historical Data...")
candles = ex_mgr.fetch_ohlc(
    exchange_name="BINANCE",
    symbol="BTCUSDT",
    timeframe="1H",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 3, 31)
)
print(f"  Fetched {len(candles)} candles")

print("\nStep 3: Running Backtest...")
engine = BacktestEngine(
    candles=candles,
    strategy_config=StrategyConfig(),  # Default strategy
    order_qty=0.1,
    exchange="BINANCE",
    symbol="BTCUSDT",
    timeframe="1H"
)

result = engine.run()

print("\n" + "="*50)
print("BACKTEST RESULTS")
print("="*50)
print(f"Total Trades: {result.total_trades}")
print(f"Win Rate: {result.win_rate:.2f}%")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2f}%")
print(f"Gross P&L: ${result.gross_pnl:.2f}")
print(f"Net P&L: ${result.net_pnl:.2f}")
print(f"Recovery Factor: {result.recovery_factor:.2f}")
print(f"Profit Factor: {result.profit_factor:.2f}")
print("="*50)
```

Run it:

```bash
python run_backtest.py
```

**Expected Output:**

```
Step 1: Validating Exchange Connection...
  Status: READY

Step 2: Fetching Historical Data...
  Fetched 2160 candles

Step 3: Running Backtest...

==================================================
BACKTEST RESULTS
==================================================
Total Trades: 145
Win Rate: 52.41%
Sharpe Ratio: 1.42
Max Drawdown: -8.30%
Gross P&L: $2720.00
Net P&L: $2415.00
Recovery Factor: 29.08
Profit Factor: 1.43
==================================================
```

## What's Next?

### Option A: Optimize Parameters

```python
from optimization_engine import OptimizationEngine
from config import OptimizationConfig, OptimizationProfile

print("Running parameter optimization...")
opt_engine = OptimizationEngine(
    candles=candles,
    base_config=StrategyConfig(),
    order_qty=0.1,
    exchange="BINANCE",
    symbol="BTCUSDT",
    timeframe="1H",
    opt_config=OptimizationConfig(profile=OptimizationProfile.FAST)
)

# Stage 7.1: Integer parameters
print("\nStage 7.1: Optimizing integer parameters...")
results_7_1 = opt_engine.optimize_stage_7_1()
print(f"  Best Sharpe: {results_7_1['best_sharpe']:.4f}")

# Stage 7.2: Float parameters  
print("\nStage 7.2: Optimizing float parameters...")
results_7_2 = opt_engine.optimize_stage_7_2()
print(f"  Best Sharpe: {results_7_2['best_sharpe']:.4f}")

# Stage 7.3: Hybrid timeouts
print("\nStage 7.3: Optimizing hybrid timeouts...")
results_7_3 = opt_engine.optimize_stage_7_3()
print(f"  Best Sharpe: {results_7_3['best_sharpe']:.4f}")

print("\nSaving results...")
opt_engine.save_results()
print("  Done! Results saved to ./data/results/")
```

### Option B: Validate with Walk-Forward

```python
from walk_forward_engine import WalForwardEngine
from config import WalForwardConfig

print("Running walk-forward validation...")
wfa_engine = WalForwardEngine(
    candles=candles,
    optimized_config=opt_engine.get_final_config(),
    order_qty=0.1,
    exchange="BINANCE",
    symbol="BTCUSDT",
    timeframe="1H",
    wfa_config=WalForwardConfig()
)

print("Generating rolling windows...")
windows = wfa_engine.generate_windows()
print(f"  Generated {len(windows)} windows")

print("\nRunning walk-forward analysis...")
result = wfa_engine.run()

print("\nWalk-Forward Results:")
print(f"  Consistency Score: {result.consistency_score:.1%}")
print(f"  Stability Score: {result.stability_score:.2f}")
print(f"  Overfitting: {result.overfitting_indicator:.1%}")
print(f"  Confidence: {result.confidence_score:.1%}")

wfa_engine.save_results(result)
print("\n  Results saved to ./data/results/")
```

## Common Tasks

### Change Strategy Parameters

```python
from config import StrategyConfig

custom_strategy = StrategyConfig(
    ema_fast=15,        # Changed from 20
    ema_slow=45,        # Changed from 50
    ema_trend=180,      # Changed from 200
    rsi_long_threshold=30,   # Changed from 35
    rsi_short_threshold=70,  # Changed from 65
    atr_sl_multiplier=2.0,   # Changed from 1.5
    atr_tp_multiplier=3.0    # Changed from 2.0
)

engine = BacktestEngine(
    candles=candles,
    strategy_config=custom_strategy,
    order_qty=0.1
)

result = engine.run()
```

### Test Multiple Exchanges

```python
exchanges = ["BINANCE", "DELTA"]

for exchange in exchanges:
    print(f"\nTesting {exchange}...")
    try:
        status = ex_mgr.validate_connectivity(exchange)
        if status['status'] == "READY":
            # Fetch data and run backtest
            print(f"  {exchange}: OK")
        else:
            print(f"  {exchange}: FAILED - {status['message']}")
    except Exception as e:
        print(f"  {exchange}: ERROR - {str(e)}")
```

### Multi-Timeframe Backtest

```python
timeframes = ["1H", "4H", "1D"]
results = {}

for tf in timeframes:
    print(f"\nBacktesting {tf}...")
    candles = ex_mgr.fetch_ohlc(
        "BINANCE", "BTCUSDT", tf,
        datetime(2025, 1, 1), datetime(2025, 3, 31)
    )
    
    engine = BacktestEngine(candles, StrategyConfig(), 0.1,
                           "BINANCE", "BTCUSDT", tf)
    results[tf] = engine.run()
    print(f"  Sharpe: {results[tf].sharpe_ratio:.2f}")

# Compare results
print("\nSummary:")
for tf, result in results.items():
    print(f"{tf}: Win Rate {result.win_rate:.1f}% | "
          f"Sharpe {result.sharpe_ratio:.2f}")
```

## Troubleshooting

### Error: "Missing BINANCE_API_KEY"

Ensure your `config/.env` file exists and has credentials:

```bash
cat config/.env
# Should show: BINANCE_API_KEY=...
```

### Error: "Connection failed"

Check your internet connection and API key validity. Test with:

```python
status = ex_mgr.validate_connectivity("BINANCE")
print(status)  # Should show status: READY
```

### Error: "Insufficient data"

Extend your date range:

```python
candles = ex_mgr.fetch_ohlc(
    "BINANCE", "BTCUSDT", "1H",
    datetime(2024, 1, 1),   # Earlier start
    datetime(2025, 3, 31)
)
```

## Next Steps

1. **Read Full Documentation**: Check [README.md](README.md)
2. **Explore Examples**: See [examples/](examples/) directory
3. **Advanced Optimization**: Try BALANCED/THOROUGH profiles
4. **Interactive Dashboard**: Launch Flask dashboard
5. **Contributing**: Submit PRs or issues on GitHub

---

**Enjoy backtesting! Happy trading! 🚀**
