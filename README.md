# Cryptocurrency Trading Backtest Engine

**Production-Grade Backtesting Engine with Multi-Exchange Support, Bayesian Optimization, Walk-Forward Validation, and Interactive Dashboard**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Status](https://img.shields.io/badge/Status-Production--Ready-green)
![Version](https://img.shields.io/badge/Version-2.0-blue)

---

## Overview

A professional-grade cryptocurrency trading backtesting engine designed for algorithmic traders and quants. Features:

- **Multi-Exchange Support**: Binance, Delta Exchange, Zerodha, Dhan
- **Advanced Strategy**: EMA/RSI/ATR with 5-tier hybrid entry engine
- **Bayesian Optimization**: Optuna-based parameter optimization (3 stages)
- **Walk-Forward Validation**: Robustness testing with rolling windows
- **Interactive Dashboard**: Flask + Plotly visualization
- **Production-Ready**: Comprehensive logging, error handling, data persistence

---

## Key Features

✅ Vectorized indicator calculations (EMA, RSI, ATR, VWAP)  
✅ Multi-timeframe backtesting support  
✅ Sophisticated hybrid entry logic (5-tier execution)  
✅ Trailing stop-loss and take-profit management  
✅ Optuna-based Bayesian optimization  
✅ Multi-fidelity sampling (33% → 66% → 100%)  
✅ Walk-forward analysis with robustness metrics  
✅ Real-time Flask dashboard  
✅ Comprehensive audit trails and logging  
✅ Checkpoint-based recovery  

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/crypto-trading-backtest-engine.git
cd crypto-trading-backtest-engine

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure Credentials

Create `config/.env`:

```env
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
DELTA_API_KEY=your_key
DELTA_API_SECRET=your_secret
```

### 3. Run Backtest

```python
from exchange_manager import ExchangeManager
from backtest_strategies import BacktestEngine
from config import StrategyConfig
from datetime import datetime

# Fetch data
ex_mgr = ExchangeManager()
candles = ex_mgr.fetch_ohlc(
    exchange_name="BINANCE",
    symbol="BTCUSDT",
    timeframe="1H",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 3, 31)
)

# Run backtest
engine = BacktestEngine(
    candles=candles,
    strategy_config=StrategyConfig(),
    order_qty=0.1,
    exchange="BINANCE",
    symbol="BTCUSDT",
    timeframe="1H"
)

result = engine.run()
print(f"Win Rate: {result.win_rate:.2f}%")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Net P&L: ${result.net_pnl:.2f}")
```

---

## Architecture

### Stages

| Stage | Component | Purpose |
|-------|-----------|----------|
| 1-2 | ExchangeManager | Credential validation & API connectivity |
| 3 | User Input | Collect exchange/symbol/dates/timeframes |
| 4 | Strategy Config | Set EMA/RSI/ATR parameters |
| 5 | Data Fetch & Baseline | Get OHLC data & run baseline backtest |
| 6 | Hardware Profile | CPU/GPU benchmarking |
| 7 | Optimization | Bayesian optimization (7.1/7.2/7.3) |
| 8 | Walk-Forward | Rolling window validation |
| 9 | Unified Report | Stage 5/7/8 comparison |
| 10 | Dashboard | Interactive visualization |

### Technology Stack

- **Data Processing**: NumPy, Pandas, Polars
- **Optimization**: Optuna (Bayesian), scikit-optimize
- **Visualization**: Plotly, Matplotlib
- **Web**: Flask, Flask-CORS
- **API Integration**: Requests, aiohttp, WebSockets
- **Parallelization**: multiprocessing, Ray (optional)
- **Testing**: pytest, pytest-mock, pytest-cov

---

## Configuration

### Strategy Parameters

```python
from config import StrategyConfig

config = StrategyConfig(
    # EMA
    ema_fast=20,
    ema_slow=50,
    ema_trend=200,
    
    # RSI
    rsi_period=14,
    rsi_long_threshold=35,
    rsi_short_threshold=65,
    
    # Risk Management
    atr_period=14,
    atr_sl_multiplier=1.5,
    atr_tp_multiplier=2.0,
    trailing_sl_threshold=0.02,
    trailing_tp_threshold=0.03,
    
    # Hybrid Entry (5-Tier)
    enable_hybrid_tier1=True,  # EMA touch
    enable_hybrid_tier2=True,  # Limit order
    enable_hybrid_tier3=True,  # VWAP slice
    enable_hybrid_tier4=True,  # Market order
    hybrid_tier1_timeout=5,
    hybrid_tier2_timeout=3,
    hybrid_tier3_timeout=4,
    hybrid_tier4_timeout=1,
    
    # Filters
    enable_trend_filter=True,
    enable_rsi_filter=True,
    trailing_sl_enabled=True,
    trailing_tp_enabled=True
)
```

### Optimization Profile

```python
from config import OptimizationConfig, OptimizationProfile

opt_config = OptimizationConfig(
    profile=OptimizationProfile.BALANCED,  # FAST/BALANCED/THOROUGH
    optuna_sampler="TPE",
    multi_fidelity=True,
    fidelity_levels=(0.33, 0.66, 1.0)
)

# Profile Details:
# FAST:      ±10% grid, 50 trials, ~5-10 minutes
# BALANCED:  ±15% grid, 150 trials, ~25-30 minutes  
# THOROUGH:  ±25% grid, 300 trials, ~60+ minutes
```

### Walk-Forward Settings

```python
from config import WalForwardConfig

wfa_config = WalForwardConfig(
    is_length_days=60,      # In-sample window
    oos_length_days=20,     # Out-of-sample window
    stride_days=20,         # Window overlap
    min_trades_per_window=5 # Minimum trades required
)
```

---

## Usage Examples

### Example 1: Simple Backtest

```python
from exchange_manager import ExchangeManager
from backtest_strategies import BacktestEngine
from config import StrategyConfig
from datetime import datetime

ex_mgr = ExchangeManager()
candles = ex_mgr.fetch_ohlc(
    "BINANCE", "BTCUSDT", "1H",
    datetime(2025, 1, 1), datetime(2025, 3, 31)
)

engine = BacktestEngine(
    candles, StrategyConfig(), 0.1,
    "BINANCE", "BTCUSDT", "1H"
)
result = engine.run()

print(f"Trades: {result.total_trades}")
print(f"Win Rate: {result.win_rate:.2f}%")
print(f"Sharpe: {result.sharpe_ratio:.2f}")
print(f"Max DD: {result.max_drawdown:.2f}%")
```

### Example 2: Parameter Optimization

```python
from optimization_engine import OptimizationEngine
from config import OptimizationConfig, OptimizationProfile

opt_engine = OptimizationEngine(
    candles, StrategyConfig(), 0.1,
    exchange="BINANCE", symbol="BTCUSDT", timeframe="1H",
    opt_config=OptimizationConfig(profile=OptimizationProfile.BALANCED)
)

# Stage 7.1: Integer parameters
print("Running Stage 7.1 optimization...")
results_7_1 = opt_engine.optimize_stage_7_1()
print(f"Best Sharpe: {results_7_1['best_sharpe']:.4f}")
print(f"Parameters: {results_7_1['best_params']}")

# Stage 7.2: Float parameters
print("Running Stage 7.2 optimization...")
results_7_2 = opt_engine.optimize_stage_7_2()
print(f"Best Sharpe: {results_7_2['best_sharpe']:.4f}")

# Stage 7.3: Hybrid timeouts
print("Running Stage 7.3 optimization...")
results_7_3 = opt_engine.optimize_stage_7_3()
print(f"Best Sharpe: {results_7_3['best_sharpe']:.4f}")

opt_engine.save_results()
```

### Example 3: Walk-Forward Validation

```python
from walk_forward_engine import WalForwardEngine
from config import WalForwardConfig

wfa_engine = WalForwardEngine(
    candles,
    opt_engine.get_final_config(),
    0.1,
    exchange="BINANCE",
    symbol="BTCUSDT",
    timeframe="1H",
    wfa_config=WalForwardConfig()
)

result = wfa_engine.run()

print(f"Consistency: {result.consistency_score:.1%}")
print(f"Stability: {result.stability_score:.2f}")
print(f"Overfitting: {result.overfitting_indicator:.1%}")
print(f"Confidence: {result.confidence_score:.1%}")

wfa_engine.save_results(result)
```

---

## Performance Benchmarks

On Intel i7, 16GB RAM:

| Stage | Task | Time | Notes |
|-------|------|------|-------|
| 5 | Fetch + Backtest (3mo 1H) | ~30s | 1000+ candles |
| 7.1 | Integer Optimization | ~8m | 150 trials |
| 7.2 | Float Optimization | ~15m | 150 trials |
| 7.3 | Timeout Optimization | ~4m | 50 trials |
| 8 | Walk-Forward (7 windows) | ~2m | Full optimization |
| **Total** | **Full Pipeline** | **~30m** | Sequential execution |

**With GPU acceleration**: ~6× speedup possible

---

## Directory Structure

```
crypto-trading-backtest-engine/
├── config.py                      # Configuration dataclasses
├── exchange_manager.py            # Multi-exchange API
├── backtest_strategies.py         # Core strategy logic
├── optimization_engine.py         # Bayesian optimization
├── walk_forward_engine.py         # Walk-forward validation
├── dashboard_app.py               # Flask dashboard
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
│
├── config/
│   └── .env                       # API credentials
│
├── data/
│   ├── raw_logs/                  # Trade logs (JSON)
│   ├── results/                   # Backtest results
│   ├── exports/                   # CSV exports
│   ├── checkpoints/               # Stage checkpoints
│   └── cache/                     # Optional cache
│
├── logs/
│   ├── exchange_manager.log
│   ├── backtest_engine.log
│   ├── optimization_engine.log
│   └── error.log
│
and tests/
    ├── test_exchange_manager.py
    ├── test_backtest_strategies.py
    ├── test_optimization_engine.py
    └── test_walk_forward_engine.py
```

---

## Data Persistence

Results stored in `./data/`:

- **raw_logs**: Trade-by-trade logs per exchange/symbol
- **results**: JSON results from each stage
- **exports**: CSV tables for Excel/analysis
- **checkpoints**: Stage recovery points
- **hardware_profile.json**: CPU/GPU benchmarks

---

## Error Handling

Comprehensive error handling with custom exceptions:

- `CredentialError`: Missing/invalid API keys
- `APIError`: HTTP/network failures
- `RateLimitError`: Exchange rate limits
- `DataError`: Invalid OHLC data
- `BacktestError`: Strategy execution errors

Automatic retry with exponential backoff (1s → 2s → 4s)

---

## Logging

Structured logging to `./logs/` with:

- Console output (INFO level)
- File handlers with rotation
- JSON formatting for machine parsing
- Full stack traces on errors
- Per-module log levels

---

## Troubleshooting

### Issue: Missing Credentials

```bash
# Ensure config/.env exists and has valid keys
echo "BINANCE_API_KEY=xxx" >> config/.env
```

### Issue: Rate Limit Errors

Automatic retry with backoff. If persistent, reduce request frequency.

### Issue: Insufficient Data

Extend backtest date range:

```python
start_date=datetime(2024, 1, 1)  # Earlier start date
end_date=datetime(2025, 3, 31)
```

### Issue: Slow Optimization

Use FAST profile or fewer trials:

```python
opt_config = OptimizationConfig(profile=OptimizationProfile.FAST)
```

---

## API Reference

### ExchangeManager

```python
ex_mgr = ExchangeManager()  # Singleton

# Validate connectivity
status = ex_mgr.validate_connectivity("BINANCE")

# Fetch OHLC
candles = ex_mgr.fetch_ohlc(
    exchange_name="BINANCE",
    symbol="BTCUSDT",
    timeframe="1H",
    start_date=datetime(2025, 1, 1),
    end_date=datetime(2025, 3, 31)
)
```

### BacktestEngine

```python
engine = BacktestEngine(
    candles=candles,
    strategy_config=StrategyConfig(),
    order_qty=0.1,
    exchange="BINANCE",
    symbol="BTCUSDT",
    timeframe="1H"
)

result = engine.run()  # BacktestResult
```

### OptimizationEngine

```python
opt = OptimizationEngine(
    candles, config, 0.1,
    exchange="BINANCE", symbol="BTCUSDT"
)

results_7_1 = opt.optimize_stage_7_1()
results_7_2 = opt.optimize_stage_7_2()
results_7_3 = opt.optimize_stage_7_3()

opt.save_results()
```

### WalForwardEngine

```python
wfa = WalForwardEngine(
    candles, optimized_config, 0.1,
    wfa_config=WalForwardConfig()
)

windows = wfa.generate_windows()
result = wfa.run()
wfa.save_results(result)
```

---

## Contributing

Contributions welcome! Please:

1. Fork repository
2. Create feature branch
3. Add tests
4. Submit pull request

---

## License

MIT License - See LICENSE file

---

## Support

For issues or questions:
- Open GitHub Issue
- Check Troubleshooting section
- Review code docstrings

---

**Built with ❤️ for quantitative traders**

Version 2.0 • December 2025
