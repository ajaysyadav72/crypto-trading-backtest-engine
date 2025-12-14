"""Multi-stage parameter optimization engine using Optuna."""

import logging
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from typing import Dict, List, Tuple, Any, Callable
from multiprocessing import Pool, cpu_count
import json
from pathlib import Path
from datetime import datetime

from config import (
    StrategyConfig, OptimizationConfig, OptimizationProfile,
    GlobalConfig
)
from backtest_strategies import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


class OptimizationEngine:
    """Multi-stage Bayesian optimization using Optuna."""
    
    def __init__(
        self,
        candles: List[Dict],
        base_config: StrategyConfig,
        order_qty: float,
        exchange: str = "BINANCE",
        symbol: str = "BTCUSDT",
        timeframe: str = "1H",
        opt_config: OptimizationConfig = None
    ):
        """Initialize optimization engine."""
        self.candles = candles
        self.base_config = base_config
        self.order_qty = order_qty
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.opt_config = opt_config or OptimizationConfig()
        
        # Multi-fidelity data splits
        self.data_splits = self._calculate_data_splits()
        
        # Results tracking
        self.stage_7_1_results = {}
        self.stage_7_2_results = {}
        self.stage_7_3_results = {}
        
        logger.info(
            f"OptimizationEngine initialized ({self.opt_config.profile.value})"
        )
    
    def _calculate_data_splits(self) -> Dict[str, List[Dict]]:
        """Calculate multi-fidelity data splits."""
        total = len(self.candles)
        low_end = total // 3      # 33%
        mid_end = (2 * total) // 3  # 66%
        
        return {
            "low": self.candles[:low_end],
            "mid": self.candles[:mid_end],
            "high": self.candles
        }
    
    def optimize_stage_7_1(self) -> Dict[str, Any]:
        """Stage 7.1: Integer parameter grid optimization."""
        logger.info(
            "Stage 7.1: Starting integer grid optimization "
            f"(Trials: {self.opt_config.num_trials})"
        )
        
        delta = self.opt_config.get_grid_delta()
        
        # Define parameter ranges
        def objective(trial):
            ema_fast = trial.suggest_int(
                "ema_fast",
                int(self.base_config.ema_fast * (1 - delta)),
                int(self.base_config.ema_fast * (1 + delta))
            )
            ema_slow = trial.suggest_int(
                "ema_slow",
                int(self.base_config.ema_slow * (1 - delta)),
                int(self.base_config.ema_slow * (1 + delta))
            )
            ema_trend = trial.suggest_int(
                "ema_trend",
                int(self.base_config.ema_trend * (1 - delta)),
                int(self.base_config.ema_trend * (1 + delta))
            )
            rsi_period = trial.suggest_int(
                "rsi_period",
                int(self.base_config.rsi_period * (1 - delta)),
                int(self.base_config.rsi_period * (1 + delta))
            )
            atr_period = trial.suggest_int(
                "atr_period",
                int(self.base_config.atr_period * (1 - delta)),
                int(self.base_config.atr_period * (1 + delta))
            )
            
            # Validate parameter constraints
            if ema_fast >= ema_slow or ema_slow >= ema_trend:
                return -999
            
            # Multi-fidelity evaluation
            try:
                config = self.base_config.__class__(
                    **{
                        **vars(self.base_config),
                        "ema_fast": ema_fast,
                        "ema_slow": ema_slow,
                        "ema_trend": ema_trend,
                        "rsi_period": rsi_period,
                        "atr_period": atr_period
                    }
                )
                
                # Low fidelity
                result_low = self._run_backtest(
                    config, self.data_splits["low"]
                )
                sharpe_low = result_low.sharpe_ratio or 0
                trial.report(sharpe_low, step=0)
                
                # Pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                # Mid fidelity
                result_mid = self._run_backtest(
                    config, self.data_splits["mid"]
                )
                sharpe_mid = result_mid.sharpe_ratio or 0
                trial.report(sharpe_mid, step=1)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                # High fidelity
                result_high = self._run_backtest(
                    config, self.data_splits["high"]
                )
                sharpe_high = result_high.sharpe_ratio or 0
                
                return sharpe_high
            
            except Exception as e:
                logger.error(f"Trial failed: {str(e)}")
                return -999
        
        # Create and run study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner()
        
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner
        )
        
        study.optimize(
            objective,
            n_trials=self.opt_config.num_trials,
            n_jobs=min(4, cpu_count()),
            show_progress_bar=True
        )
        
        # Extract best results
        best_trial = study.best_trial
        self.stage_7_1_results = {
            "best_params": best_trial.params,
            "best_sharpe": best_trial.value,
            "num_trials": len(study.trials),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(
            f"Stage 7.1 Complete: Best Sharpe = {best_trial.value:.4f} "
            f"(Params: {best_trial.params})"
        )
        
        return self.stage_7_1_results
    
    def optimize_stage_7_2(self) -> Dict[str, Any]:
        """Stage 7.2: Float parameter optimization."""
        logger.info(
            "Stage 7.2: Starting float parameter optimization "
            f"(Trials: {self.opt_config.num_trials})"
        )
        
        def objective(trial):
            rsi_long = trial.suggest_float(
                "rsi_long", 20, 50, step=1
            )
            rsi_short = trial.suggest_float(
                "rsi_short", 50, 80, step=1
            )
            atr_sl_mult = trial.suggest_float(
                "atr_sl_mult", 0.5, 5.0, step=0.1
            )
            atr_tp_mult = trial.suggest_float(
                "atr_tp_mult", 1.0, 5.0, step=0.1
            )
            trailing_sl = trial.suggest_float(
                "trailing_sl", 0.01, 0.10, step=0.01
            )
            trailing_tp = trial.suggest_float(
                "trailing_tp", 0.01, 0.10, step=0.01
            )
            
            try:
                # Use Stage 7.1 best parameters
                params = self.stage_7_1_results.get("best_params", {})
                
                config = self.base_config.__class__(
                    **{
                        **vars(self.base_config),
                        **params,
                        "rsi_long_threshold": int(rsi_long),
                        "rsi_short_threshold": int(rsi_short),
                        "atr_sl_multiplier": atr_sl_mult,
                        "atr_tp_multiplier": atr_tp_mult,
                        "trailing_sl_threshold": trailing_sl,
                        "trailing_tp_threshold": trailing_tp
                    }
                )
                
                # Multi-fidelity evaluation
                result_high = self._run_backtest(
                    config, self.data_splits["high"]
                )
                
                return result_high.sharpe_ratio or 0
            
            except Exception as e:
                logger.error(f"Trial failed: {str(e)}")
                return -999
        
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler
        )
        
        study.optimize(
            objective,
            n_trials=self.opt_config.num_trials,
            n_jobs=min(4, cpu_count()),
            show_progress_bar=True
        )
        
        best_trial = study.best_trial
        self.stage_7_2_results = {
            "best_params": best_trial.params,
            "best_sharpe": best_trial.value,
            "num_trials": len(study.trials),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(
            f"Stage 7.2 Complete: Best Sharpe = {best_trial.value:.4f}"
        )
        
        return self.stage_7_2_results
    
    def optimize_stage_7_3(self) -> Dict[str, Any]:
        """Stage 7.3: Hybrid timeout optimization."""
        logger.info(
            "Stage 7.3: Starting hybrid timeout optimization "
            f"(Trials: {self.opt_config.num_trials // 3})"
        )
        
        def objective(trial):
            t1_timeout = trial.suggest_int("tier1_timeout", 2, 8)
            t2_timeout = trial.suggest_int("tier2_timeout", 1, 6)
            t3_timeout = trial.suggest_int("tier3_timeout", 2, 7)
            t4_timeout = trial.suggest_int("tier4_timeout", 1, 3)
            
            # Constraint: T1 ≤ T2 ≤ T3
            if t1_timeout > t2_timeout or t2_timeout > t3_timeout:
                return -999
            
            try:
                # Merge all optimized parameters
                params_7_1 = self.stage_7_1_results.get("best_params", {})
                params_7_2 = self.stage_7_2_results.get("best_params", {})
                
                config = self.base_config.__class__(
                    **{
                        **vars(self.base_config),
                        **params_7_1,
                        **params_7_2,
                        "hybrid_tier1_timeout": t1_timeout,
                        "hybrid_tier2_timeout": t2_timeout,
                        "hybrid_tier3_timeout": t3_timeout,
                        "hybrid_tier4_timeout": t4_timeout
                    }
                )
                
                result = self._run_backtest(
                    config, self.data_splits["high"]
                )
                
                return result.sharpe_ratio or 0
            
            except Exception as e:
                logger.error(f"Trial failed: {str(e)}")
                return -999
        
        sampler = TPESampler(seed=42)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler
        )
        
        study.optimize(
            objective,
            n_trials=self.opt_config.num_trials // 3,
            n_jobs=min(4, cpu_count()),
            show_progress_bar=True
        )
        
        best_trial = study.best_trial
        self.stage_7_3_results = {
            "best_params": best_trial.params,
            "best_sharpe": best_trial.value,
            "num_trials": len(study.trials),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(
            f"Stage 7.3 Complete: Best Sharpe = {best_trial.value:.4f}"
        )
        
        return self.stage_7_3_results
    
    def _run_backtest(
        self,
        config: StrategyConfig,
        candles: List[Dict]
    ) -> BacktestResult:
        """Run single backtest."""
        engine = BacktestEngine(
            candles=candles,
            strategy_config=config,
            order_qty=self.order_qty,
            exchange=self.exchange,
            symbol=self.symbol,
            timeframe=self.timeframe
        )
        return engine.run()
    
    def get_final_config(self) -> StrategyConfig:
        """Get final optimized configuration."""
        params_7_1 = self.stage_7_1_results.get("best_params", {})
        params_7_2 = self.stage_7_2_results.get("best_params", {})
        params_7_3 = self.stage_7_3_results.get("best_params", {})
        
        all_params = {
            **vars(self.base_config),
            **params_7_1,
            **params_7_2,
            **params_7_3
        }
        
        return StrategyConfig(**all_params)
    
    def save_results(self) -> str:
        """Save optimization results to file."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "exchange": self.exchange,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "stage_7_1": self.stage_7_1_results,
            "stage_7_2": self.stage_7_2_results,
            "stage_7_3": self.stage_7_3_results,
            "final_config": vars(self.get_final_config())
        }
        
        filepath = Path(
            GlobalConfig.RESULTS_DIR
        ) / f"stage_7_optimization_{self.symbol}_{self.timeframe}.json"
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
        return str(filepath)
