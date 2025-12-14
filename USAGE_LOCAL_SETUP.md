# Local Machine Setup - 2 File Implementation

## Quick Overview

You have **2 main files** that work together:

1. **exchange_manager_compact.py** - Handles all exchange APIs (Binance, Delta, Zerodha, Dhan)
2. **backtest_engine_complete.py** - Orchestrates all 12 stages of backtesting

No external databases or complex setup needed. Everything runs **locally on your machine**.

---

## Installation (5 minutes)

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Setup Environment Variables

Create `config/.env`:

```bash
mkdir -p config
cat > config/.env << 'EOF'
# Binance
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# Optional - other exchanges
# DELTA_API_KEY=your_key
# DELTA_API_SECRET=your_secret
# ZERODHA_API_KEY=your_key
# ZERODHA_ACCESS_TOKEN=your_token
# DHAN_ACCESS_TOKEN=your_token
EOF
```

### Step 3: Verify Installation

```python
from exchange_manager_compact import ExchangeManager

ex = ExchangeManager()
status = ex.validate_connectivity("BINANCE")
print(f"Status: {status['status']}")
```

---

## Usage Examples

### Example 1: Simple Backtest (5 minutes)

Create `run_simple_backtest.py`:

```python
from backtest_engine_complete import (
    BacktestEngine, StrategyConfig, FullBacktestOrchestrator
)
from exchange_manager_compact import ExchangeManager
from datetime import datetime

# Step 1: Initialize
ex_mgr = ExchangeManager()
orchestrator = FullBacktestOrchestrator(ex_mgr)

# Step 2: Run backtest
results = orchestrator.run_full_pipeline(
    exchange="BINANCE",
    symbol="BTCUSDT",
    timeframe="1H",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 3, 31),
    order_qty=0.1,
    enable_optimization=False,  # Skip optimization for speed
    enable_wfa=False  # Skip walk-forward for speed
)

print(f"\nBaseline Sharpe: {results['baseline'].sharpe_ratio:.2f}")
print(f"Win Rate: {results['baseline'].win_rate:.2f}%")
print(f"Net P&L: ${results['baseline'].net_pnl:.2f}")
```

Run:
```bash
python run_simple_backtest.py
```

### Example 2: Full Pipeline with Optimization (30 minutes)

Create `run_full_backtest.py`:

```python
from backtest_engine_complete import FullBacktestOrchestrator
from exchange_manager_compact import ExchangeManager
from datetime import datetime

ex_mgr = ExchangeManager()
orchestrator = FullBacktestOrchestrator(ex_mgr)

# Run COMPLETE pipeline with all stages
results = orchestrator.run_full_pipeline(
    exchange="BINANCE",
    symbol="BTCUSDT",
    timeframe="1H",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 3, 31),
    order_qty=0.1,
    optimization_profile="BALANCED",  # FAST/BALANCED/THOROUGH
    enable_optimization=True,   # Stage 7: Optimization
    enable_wfa=True  # Stage 8: Walk-Forward
)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

baseline = results['baseline']
optimized = results['optimized']
wfa = results['wfa']

print(f"\nBaseline Sharpe:   {baseline.sharpe_ratio:.2f}")
print(f"Optimized Sharpe:  {optimized.sharpe_ratio:.2f}")
improvement = ((optimized.sharpe_ratio - baseline.sharpe_ratio) / baseline.sharpe_ratio * 100)
print(f"Improvement:       +{improvement:.1f}%")

if 'confidence_score' in wfa:
    print(f"\nWalk-Forward Confidence: {wfa['confidence_score']:.1f}%")
```

Run:
```bash
python run_full_backtest.py
```

### Example 3: Custom Strategy Parameters

Create `run_custom_strategy.py`:

```python
from backtest_engine_complete import BacktestEngine, StrategyConfig
from exchange_manager_compact import ExchangeManager
from datetime import datetime

# Create custom strategy
custom_strategy = StrategyConfig(
    ema_fast=15,        # Faster response
    ema_slow=45,        # Tighter bands
    ema_trend=150,      # Shorter trend
    rsi_period=10,      # Faster RSI
    rsi_long_threshold=30,   # More aggressive
    rsi_short_threshold=70,  # More aggressive
    atr_sl_multiplier=2.0,   # Wider stop loss
    atr_tp_multiplier=3.0    # Higher take profit
)

# Fetch data
ex_mgr = ExchangeManager()
candles = ex_mgr.fetch_ohlc(
    "BINANCE", "BTCUSDT", "1H",
    datetime(2025, 1, 1),
    datetime(2025, 3, 31)
)

# Backtest
engine = BacktestEngine(
    candles=candles,
    strategy_config=custom_strategy,
    order_qty=0.1,
    exchange="BINANCE",
    symbol="BTCUSDT",
    timeframe="1H"
)

result = engine.run()

print(f"Trades: {result.total_trades}")
print(f"Win Rate: {result.win_rate:.2f}%")
print(f"Sharpe: {result.sharpe_ratio:.2f}")
print(f"Net P&L: ${result.net_pnl:.2f}")
```

### Example 4: Multi-Timeframe Backtest

Create `run_multi_timeframe.py`:

```python
from backtest_engine_complete import BacktestEngine, StrategyConfig
from exchange_manager_compact import ExchangeManager
from datetime import datetime

ex_mgr = ExchangeManager()
strategy = StrategyConfig()

timeframes = ["1H", "4H", "1D"]
results = {}

for tf in timeframes:
    print(f"\nTesting {tf}...")
    
    candles = ex_mgr.fetch_ohlc(
        "BINANCE", "BTCUSDT", tf,
        datetime(2025, 1, 1),
        datetime(2025, 3, 31)
    )
    
    engine = BacktestEngine(
        candles, strategy, 0.1,
        "BINANCE", "BTCUSDT", tf
    )
    
    result = engine.run()
    results[tf] = {
        "trades": result.total_trades,
        "win_rate": result.win_rate,
        "sharpe": result.sharpe_ratio,
        "net_pnl": result.net_pnl
    }
    
    print(f"  Trades: {result.total_trades}")
    print(f"  Win Rate: {result.win_rate:.2f}%")
    print(f"  Sharpe: {result.sharpe_ratio:.2f}")

# Compare
print("\n" + "="*50)
print("COMPARISON")
print("="*50)

for tf, res in results.items():
    print(f"\n{tf}:")
    print(f"  Trades: {res['trades']}")
    print(f"  Win Rate: {res['win_rate']:.2f}%")
    print(f"  Sharpe: {res['sharpe']:.2f}")
    print(f"  P&L: ${res['net_pnl']:.2f}")
```

---

## File Outputs

After running backtest, files are saved to:

```
data/
├── backtest_results_BTCUSDT_1H.json   (All results)
logs/
├── backtest_engine.log                (Detailed logs)
```

Results include:
- All metrics (Sharpe, Win Rate, Max DD, etc.)
- Individual trades with entry/exit prices
- Equity curve data
- Optimization results (if enabled)
- Walk-forward validation (if enabled)

---

## Key Classes & Methods

### ExchangeManager (exchange_manager_compact.py)

```python
ex_mgr = ExchangeManager()  # Singleton

# Validate exchange
status = ex_mgr.validate_connectivity("BINANCE")

# Fetch OHLC data
candles = ex_mgr.fetch_ohlc(
    exchange_name="BINANCE",
    symbol="BTCUSDT",
    timeframe="1H",
    start_date=datetime(...),
    end_date=datetime(...)
)
```

### BacktestEngine (backtest_engine_complete.py)

```python
engine = BacktestEngine(
    candles=list_of_candles,
    strategy_config=StrategyConfig(),
    order_qty=0.1,
    exchange="BINANCE",
    symbol="BTCUSDT",
    timeframe="1H"
)

result = engine.run()  # Returns BacktestMetrics
```

### FullBacktestOrchestrator (backtest_engine_complete.py)

```python
orchestrator = FullBacktestOrchestrator(exchange_manager)

results = orchestrator.run_full_pipeline(
    exchange="BINANCE",
    symbol="BTCUSDT",
    timeframe="1H",
    start_date=datetime(...),
    end_date=datetime(...),
    order_qty=0.1,
    optimization_profile="BALANCED",
    enable_optimization=True,
    enable_wfa=True
)
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'exchange_manager_compact'"

**Solution**: Make sure both Python files are in the same directory:
```bash
ls -la *.py
# Should show:
# exchange_manager_compact.py
# backtest_engine_complete.py
```

### Issue: "CredentialError: Missing BINANCE_API_KEY"

**Solution**: Ensure config/.env has your API key:
```bash
cat config/.env
# Should show: BINANCE_API_KEY=your_key_here
```

### Issue: "DataError: No candles fetched"

**Solution**: 
- Check your API key is valid
- Extend date range: `start_date=datetime(2024, 1, 1)`
- Use higher timeframe: `timeframe="4H"` instead of `1M`

### Issue: "API Rate Limited"

**Solution**: Automatic retry with exponential backoff is built-in. Just wait or reduce request frequency.

---

## Performance Tips

1. **Faster Testing**:
   ```python
   enable_optimization=False  # Skip to test baseline only
   enable_wfa=False           # Skip walk-forward
   ```

2. **Use FAST profile**:
   ```python
   optimization_profile="FAST"  # 50 trials instead of 150
   ```

3. **Higher timeframe**:
   ```python
   timeframe="4H"  # Fewer candles to process
   ```

4. **Shorter date range**:
   ```python
   start_date=datetime(2025, 2, 1)  # 1 month instead of 3
   ```

---

## What Each Stage Does

| Stage | Time | Purpose |
|-------|------|----------|
| 1-2 | <1s | Validate exchange connection |
| 3 | <1s | User input (symbol, dates) |
| 4 | <1s | Strategy configuration |
| 5 | 5-30s | Fetch data & baseline backtest |
| 6 | ~5s | Hardware profiling |
| 7 | 25-60min* | Bayesian optimization (7.1/7.2/7.3) |
| 8 | 2-5min | Walk-forward validation |
| 9 | <1s | Unified metrics report |
| 10 | <1s | Dashboard summary |

**Total**: ~30 minutes with optimization, ~5 minutes without

---

## Next Steps

1. ✅ Setup environment (done)
2. ✅ Add API keys to config/.env (done)
3. 🚀 Run first backtest: `python run_simple_backtest.py`
4. 🔍 Customize strategy parameters
5. 📊 Try optimization with `enable_optimization=True`
6. ✓ Validate with walk-forward: `enable_wfa=True`

---

**Happy Backtesting! 🚀**
