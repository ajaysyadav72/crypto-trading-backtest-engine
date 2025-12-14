"""
QUICK START EXAMPLE - Run This First!

This script demonstrates the complete 2-file solution:
  1. exchange_manager_compact.py - Exchange API integration
  2. backtest_engine_complete.py - All 12 backtest stages

Run: python example_quickstart.py
"""

import os
import sys
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """
    Setup: Create example .env file
    """
    env_content = """# Binance API Keys
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# Optional: Other Exchanges
# DELTA_API_KEY=your_delta_api_key
# DELTA_API_SECRET=your_delta_api_secret
# ZERODHA_ACCESS_TOKEN=your_zerodha_token
# DHAN_ACCESS_TOKEN=your_dhan_token
"""
    
    os.makedirs('config', exist_ok=True)
    env_path = 'config/.env'
    
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write(env_content)
        logger.info(f"Created {env_path}")
        logger.warning("\u26a0  UPDATE YOUR API KEYS IN config/.env!")
    else:
        logger.info(f"Using existing {env_path}")


def example_1_simple_backtest():
    """
    EXAMPLE 1: Simple Baseline Backtest (5 minutes)
    
    This runs just Stage 5: Data Fetch & Baseline Backtest
    No optimization, no walk-forward validation
    """
    print("\n" + "="*70)
    print(" EXAMPLE 1: Simple Baseline Backtest")
    print("="*70)
    
    from backtest_engine_complete import FullBacktestOrchestrator
    from exchange_manager_compact import ExchangeManager
    
    try:
        # Initialize
        ex_mgr = ExchangeManager()
        orchestrator = FullBacktestOrchestrator(ex_mgr)
        
        # Run backtest (Stages 1-5 only)
        results = orchestrator.run_full_pipeline(
            exchange="BINANCE",
            symbol="BTCUSDT",
            timeframe="1H",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),  # 1 month for speed
            order_qty=0.1,
            enable_optimization=False,  # Skip optimization
            enable_wfa=False  # Skip walk-forward
        )
        
        # Print results
        baseline = results['baseline']
        print(f"\n✓ BASELINE RESULTS:")
        print(f"  Total Trades:    {baseline.total_trades}")
        print(f"  Win Rate:        {baseline.win_rate:.2f}%")
        print(f"  Sharpe Ratio:    {baseline.sharpe_ratio:.2f}")
        print(f"  Max Drawdown:    {baseline.max_drawdown:.2f}%")
        print(f"  Net P&L:         ${baseline.net_pnl:.2f}")
        print(f"  Recovery Factor: {baseline.recovery_factor:.2f}")
        
    except Exception as e:
        logger.error(f"Error in Example 1: {str(e)}")
        logger.warning("\u26a0  Make sure your API keys are set in config/.env")


def example_2_with_optimization():
    """
    EXAMPLE 2: Backtest with Optimization (30 minutes)
    
    This runs all stages:
      Stage 1-2: Exchange validation
      Stage 3-4: User input & strategy setup
      Stage 5: Baseline backtest
      Stage 6: Hardware profiling
      Stage 7: Bayesian optimization (7.1/7.2/7.3)
      Stage 8: Walk-forward validation
      Stage 9-10: Reports & dashboard
    """
    print("\n" + "="*70)
    print(" EXAMPLE 2: Full Pipeline with Optimization")
    print("="*70)
    
    from backtest_engine_complete import FullBacktestOrchestrator
    from exchange_manager_compact import ExchangeManager
    
    try:
        ex_mgr = ExchangeManager()
        orchestrator = FullBacktestOrchestrator(ex_mgr)
        
        # Run FULL pipeline with optimization
        results = orchestrator.run_full_pipeline(
            exchange="BINANCE",
            symbol="ETHUSDT",
            timeframe="4H",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 2, 28),  # 2 months
            order_qty=1.0,
            optimization_profile="FAST",  # FAST/BALANCED/THOROUGH
            enable_optimization=True,   # Stage 7
            enable_wfa=True  # Stage 8
        )
        
        # Compare baseline vs optimized
        baseline = results['baseline']
        optimized = results['optimized']
        wfa = results['wfa']
        
        print(f"\n✓ OPTIMIZATION RESULTS:")
        print(f"\n  BASELINE:")
        print(f"    Sharpe:      {baseline.sharpe_ratio:.2f}")
        print(f"    Win Rate:    {baseline.win_rate:.2f}%")
        print(f"    Net P&L:     ${baseline.net_pnl:.2f}")
        
        print(f"\n  OPTIMIZED:")
        print(f"    Sharpe:      {optimized.sharpe_ratio:.2f}")
        print(f"    Win Rate:    {optimized.win_rate:.2f}%")
        print(f"    Net P&L:     ${optimized.net_pnl:.2f}")
        
        if optimized.sharpe_ratio > 0 and baseline.sharpe_ratio > 0:
            improvement = (
                (optimized.sharpe_ratio - baseline.sharpe_ratio) / 
                baseline.sharpe_ratio * 100
            )
            print(f"\n  ↧ IMPROVEMENT: +{improvement:.1f}%")
        
        if 'confidence_score' in wfa:
            print(f"\n  WALK-FORWARD VALIDATION:")
            print(f"    Confidence:  {wfa.get('confidence_score', 0):.1f}%")
            print(f"    Consistency: {wfa.get('consistency_score', 0):.1f}%")
        
    except Exception as e:
        logger.error(f"Error in Example 2: {str(e)}")
        logger.warning("\u26a0  Full optimization requires sufficient data/time")


def example_3_custom_strategy():
    """
    EXAMPLE 3: Custom Strategy Parameters
    
    Tests a custom EMA configuration on your choice of symbol
    """
    print("\n" + "="*70)
    print(" EXAMPLE 3: Custom Strategy Parameters")
    print("="*70)
    
    from backtest_engine_complete import BacktestEngine, StrategyConfig
    from exchange_manager_compact import ExchangeManager
    
    try:
        # Create custom strategy
        custom_strategy = StrategyConfig(
            # EMA settings
            ema_fast=20,
            ema_slow=50,
            ema_trend=200,
            # RSI settings
            rsi_period=14,
            rsi_long_threshold=35,
            rsi_short_threshold=65,
            # Risk management
            atr_period=14,
            atr_sl_multiplier=1.5,
            atr_tp_multiplier=2.0,
            # Filters
            enable_trend_filter=True,
            enable_rsi_filter=True,
            trailing_sl_enabled=True,
            trailing_tp_enabled=True
        )
        
        logger.info("Custom Strategy Configuration:")
        logger.info(f"  EMA: {custom_strategy.ema_fast}/{custom_strategy.ema_slow}/{custom_strategy.ema_trend}")
        logger.info(f"  RSI: {custom_strategy.rsi_period} period")
        logger.info(f"  ATR SL/TP: {custom_strategy.atr_sl_multiplier}x / {custom_strategy.atr_tp_multiplier}x")
        
        # Fetch data
        ex_mgr = ExchangeManager()
        logger.info("\nFetching candles...")
        candles = ex_mgr.fetch_ohlc(
            exchange="BINANCE",
            symbol="SOLUSD",
            timeframe="1H",
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 2, 28)
        )
        
        # Run backtest
        logger.info(f"Running backtest with {len(candles)} candles...")
        engine = BacktestEngine(
            candles=candles,
            strategy_config=custom_strategy,
            order_qty=100,
            exchange="BINANCE",
            symbol="SOLUSD",
            timeframe="1H"
        )
        result = engine.run()
        
        # Print results
        print(f"\n✓ CUSTOM STRATEGY RESULTS (SOLUSD):")
        print(f"  Total Trades:    {result.total_trades}")
        print(f"  Win Rate:        {result.win_rate:.2f}%")
        print(f"  Sharpe Ratio:    {result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown:    {result.max_drawdown:.2f}%")
        print(f"  Net P&L:         ${result.net_pnl:.2f}")
        print(f"  Profit Factor:   {result.profit_factor:.2f}")
        
    except Exception as e:
        logger.error(f"Error in Example 3: {str(e)}")


def example_4_multi_timeframe():
    """
    EXAMPLE 4: Multi-Timeframe Comparison
    
    Tests same strategy across different timeframes
    """
    print("\n" + "="*70)
    print(" EXAMPLE 4: Multi-Timeframe Comparison")
    print("="*70)
    
    from backtest_engine_complete import BacktestEngine, StrategyConfig
    from exchange_manager_compact import ExchangeManager
    
    try:
        strategy = StrategyConfig()
        ex_mgr = ExchangeManager()
        
        results = {}
        timeframes = ["1H", "4H", "1D"]
        
        for tf in timeframes:
            logger.info(f"\nTesting {tf}...")
            
            # Adjust candle fetching for different timeframes
            candles = ex_mgr.fetch_ohlc(
                exchange="BINANCE",
                symbol="BTCUSDT",
                timeframe=tf,
                start_date=datetime(2025, 1, 1),
                end_date=datetime(2025, 2, 28)
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
                "max_dd": result.max_drawdown,
                "net_pnl": result.net_pnl
            }
        
        # Print comparison table
        print("\n✓ MULTI-TIMEFRAME COMPARISON:")
        print("\n  Timeframe | Trades | Win% | Sharpe | Max DD | Net P&L")
        print("  " + "-" * 62)
        for tf, res in results.items():
            print(
                f"  {tf:9} | {res['trades']:6d} | "
                f"{res['win_rate']:4.1f}% | {res['sharpe']:6.2f} | "
                f"{res['max_dd']:6.1f}% | ${res['net_pnl']:7.0f}"
            )
        
    except Exception as e:
        logger.error(f"Error in Example 4: {str(e)}")


def main():
    """
    Main menu - choose which example to run
    """
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  CRYPTO TRADING BACKTEST ENGINE v2.0 - QUICK START".center(68) + "#")
    print("#" + "  2-File Local Implementation (No Database)".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # Setup
    setup_environment()
    
    # Menu
    print("\nö CHOOSE AN EXAMPLE:")
    print("  1. Simple Backtest (5 min) - Stage 5 only")
    print("  2. Full Pipeline (30 min) - Stages 1-10 with optimization")
    print("  3. Custom Strategy - Test custom parameters")
    print("  4. Multi-Timeframe - Compare across 1H/4H/1D")
    print("  0. Run All Examples")
    print()
    
    choice = input("Enter choice (0-4): ").strip()
    
    if choice == "1":
        example_1_simple_backtest()
    elif choice == "2":
        example_2_with_optimization()
    elif choice == "3":
        example_3_custom_strategy()
    elif choice == "4":
        example_4_multi_timeframe()
    elif choice == "0":
        example_1_simple_backtest()
        example_3_custom_strategy()
        example_4_multi_timeframe()
    else:
        print("Invalid choice")
        return
    
    # Summary
    print("\n" + "="*70)
    print(" RESULTS SAVED TO:")
    print("="*70)
    print("  • data/backtest_results_*.json  (All metrics & trades)")
    print("  • logs/backtest_engine.log      (Detailed logs)")
    print("\n🊀 NEXT STEPS:")
    print("  1. Review results in data/ folder")
    print("  2. Adjust strategy parameters in example scripts")
    print("  3. Try different symbols/timeframes")
    print("  4. Implement your own strategies!")
    print("\nFor detailed guide, see: USAGE_LOCAL_SETUP.md")
    print("="*70)


if __name__ == "__main__":
    main()
