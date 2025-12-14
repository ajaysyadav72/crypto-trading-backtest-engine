"""Walk-forward analysis engine for strategy robustness testing."""

import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

from config import WalForwardConfig, GlobalConfig, StrategyConfig
from backtest_strategies import BacktestEngine, BacktestResult

logger = logging.getLogger(__name__)


class WalForwardResult:
    """Walk-forward analysis result."""
    
    def __init__(self):
        self.windows: List[Dict[str, Any]] = []
        self.is_metrics: List[Dict[str, float]] = []
        self.oos_metrics: List[Dict[str, float]] = []
        self.consistency_score: float = 0.0
        self.stability_score: float = 0.0
        self.overfitting_indicator: float = 0.0
        self.confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "windows": self.windows,
            "is_metrics_avg": self._avg_metrics(self.is_metrics),
            "oos_metrics_avg": self._avg_metrics(self.oos_metrics),
            "consistency_score": self.consistency_score,
            "stability_score": self.stability_score,
            "overfitting_indicator": self.overfitting_indicator,
            "confidence_score": self.confidence_score
        }
    
    @staticmethod
    def _avg_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate average metrics."""
        if not metrics:
            return {}
        
        keys = metrics[0].keys()
        avg = {}
        
        for key in keys:
            values = [m.get(key, 0) for m in metrics]
            avg[key] = np.mean(values) if values else 0
        
        return avg


class WalForwardEngine:
    """Walk-forward validation engine."""
    
    def __init__(
        self,
        candles: List[Dict],
        optimized_config: StrategyConfig,
        order_qty: float,
        exchange: str = "BINANCE",
        symbol: str = "BTCUSDT",
        timeframe: str = "1H",
        wfa_config: WalForwardConfig = None
    ):
        """Initialize walk-forward engine."""
        self.candles = candles
        self.config = optimized_config
        self.order_qty = order_qty
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.wfa_config = wfa_config or WalForwardConfig()
        
        # Convert candles to timestamps
        self.timestamps = [c["time"] for c in candles]
        
        logger.info(
            f"WalForwardEngine initialized with {len(candles)} candles"
        )
    
    def generate_windows(self) -> List[Tuple[int, int, int, int]]:
        """Generate rolling windows (IS start, IS end, OOS start, OOS end)."""
        # Approximate candles per day (1H = 24 candles/day)
        candles_per_day = 24  # For 1H timeframe
        
        if self.timeframe == "5M":
            candles_per_day = 24 * 12
        elif self.timeframe == "4H":
            candles_per_day = 6
        elif self.timeframe == "1D":
            candles_per_day = 1
        
        is_length = self.wfa_config.is_length_days * candles_per_day
        oos_length = self.wfa_config.oos_length_days * candles_per_day
        stride = self.wfa_config.stride_days * candles_per_day
        
        windows = []
        total = len(self.candles)
        t = 0
        
        while t + is_length + oos_length <= total:
            is_start = t
            is_end = t + is_length
            oos_start = is_end
            oos_end = oos_start + oos_length
            
            windows.append((is_start, is_end, oos_start, oos_end))
            t += stride
        
        logger.info(f"Generated {len(windows)} rolling windows")
        return windows
    
    def run(self) -> WalForwardResult:
        """Execute walk-forward analysis."""
        logger.info(
            f"Starting walk-forward analysis "
            f"(IS: {self.wfa_config.is_length_days}d, "
            f"OOS: {self.wfa_config.oos_length_days}d)"
        )
        
        windows = self.generate_windows()
        result = WalForwardResult()
        
        is_sharpes = []
        oos_sharpes = []
        is_win_rates = []
        oos_win_rates = []
        
        for idx, (is_start, is_end, oos_start, oos_end) in enumerate(windows):
            logger.info(f"Processing window {idx + 1}/{len(windows)}")
            
            # Extract window data
            is_data = self.candles[is_start:is_end]
            oos_data = self.candles[oos_start:oos_end]
            
            # Run in-sample backtest
            is_result = self._run_backtest(is_data)
            is_metrics = {
                "trades": is_result.total_trades,
                "win_rate": is_result.win_rate,
                "sharpe": is_result.sharpe_ratio,
                "max_dd": is_result.max_drawdown,
                "net_pnl": is_result.net_pnl
            }
            result.is_metrics.append(is_metrics)
            is_sharpes.append(is_result.sharpe_ratio or 0)
            is_win_rates.append(is_result.win_rate)
            
            # Run out-of-sample backtest
            oos_result = self._run_backtest(oos_data)
            oos_metrics = {
                "trades": oos_result.total_trades,
                "win_rate": oos_result.win_rate,
                "sharpe": oos_result.sharpe_ratio,
                "max_dd": oos_result.max_drawdown,
                "net_pnl": oos_result.net_pnl
            }
            result.oos_metrics.append(oos_metrics)
            oos_sharpes.append(oos_result.sharpe_ratio or 0)
            oos_win_rates.append(oos_result.win_rate)
            
            # Calculate degradation
            sharpe_degradation = (
                (oos_result.sharpe_ratio or 0) /
                (is_result.sharpe_ratio or 1)
            ) * 100 if is_result.sharpe_ratio else 0
            
            win_rate_degradation = (
                oos_result.win_rate / is_result.win_rate * 100
                if is_result.win_rate > 0 else 0
            )
            
            window_record = {
                "window": idx + 1,
                "is_start": is_start,
                "is_end": is_end,
                "oos_start": oos_start,
                "oos_end": oos_end,
                "is_metrics": is_metrics,
                "oos_metrics": oos_metrics,
                "sharpe_degradation_pct": sharpe_degradation,
                "win_rate_degradation_pct": win_rate_degradation
            }
            result.windows.append(window_record)
        
        # Calculate robustness metrics
        if len(windows) > 0:
            result.consistency_score = self._calculate_consistency(
                result.oos_metrics
            )
            result.stability_score = self._calculate_stability(oos_sharpes)
            result.overfitting_indicator = self._calculate_overfitting(
                is_sharpes, oos_sharpes
            )
            result.confidence_score = self._calculate_confidence(
                result.consistency_score,
                result.stability_score,
                result.overfitting_indicator
            )
        
        logger.info(
            f"Walk-Forward Complete: "
            f"Consistency={result.consistency_score:.1%}, "
            f"Stability={result.stability_score:.2f}, "
            f"Confidence={result.confidence_score:.1%}"
        )
        
        return result
    
    def _run_backtest(self, candles: List[Dict]) -> BacktestResult:
        """Run backtest on data subset."""
        if len(candles) < 100:
            logger.warning(f"Insufficient candles for backtest: {len(candles)}")
            return BacktestResult(
                exchange=self.exchange,
                symbol=self.symbol,
                timeframe=self.timeframe
            )
        
        engine = BacktestEngine(
            candles=candles,
            strategy_config=self.config,
            order_qty=self.order_qty,
            exchange=self.exchange,
            symbol=self.symbol,
            timeframe=self.timeframe
        )
        
        try:
            return engine.run()
        except Exception as e:
            logger.error(f"Backtest failed: {str(e)}")
            return BacktestResult(
                exchange=self.exchange,
                symbol=self.symbol,
                timeframe=self.timeframe
            )
    
    def _calculate_consistency(self, oos_metrics: List[Dict]) -> float:
        """Calculate consistency score (% of profitable windows)."""
        if not oos_metrics:
            return 0.0
        
        profitable = sum(1 for m in oos_metrics if m.get("net_pnl", 0) > 0)
        return profitable / len(oos_metrics)
    
    def _calculate_stability(self, oos_sharpes: List[float]) -> float:
        """Calculate stability score (1 - CV of sharpe ratios)."""
        oos_sharpes = [s for s in oos_sharpes if s is not None and s != 0]
        
        if not oos_sharpes or len(oos_sharpes) < 2:
            return 0.0
        
        mean_sharpe = np.mean(oos_sharpes)
        std_sharpe = np.std(oos_sharpes)
        
        if mean_sharpe == 0:
            return 0.0
        
        cv = std_sharpe / mean_sharpe
        return max(0.0, 1.0 - cv)
    
    def _calculate_overfitting(self, is_sharpes: List[float],
                              oos_sharpes: List[float]) -> float:
        """Calculate overfitting indicator (% degradation)."""
        is_sharpes = [s for s in is_sharpes if s is not None]
        oos_sharpes = [s for s in oos_sharpes if s is not None]
        
        if not is_sharpes:
            return 0.0
        
        mean_is = np.mean(is_sharpes)
        mean_oos = np.mean(oos_sharpes)
        
        if mean_is == 0:
            return 0.0
        
        degradation = (mean_is - mean_oos) / mean_is
        return max(0.0, degradation)
    
    def _calculate_confidence(self, consistency: float, stability: float,
                            overfitting: float) -> float:
        """Calculate overall confidence score."""
        # Weights: Consistency 40%, Stability 40%, Overfitting 20%
        confidence = (
            (consistency * 0.40) +
            (stability * 0.40) +
            ((1.0 - min(overfitting, 1.0)) * 0.20)
        )
        return confidence
    
    def save_results(self, wfa_result: WalForwardResult) -> str:
        """Save walk-forward results to file."""
        filepath = Path(
            GlobalConfig.RESULTS_DIR
        ) / f"stage_8_wfa_{self.symbol}_{self.timeframe}.json"
        
        with open(filepath, 'w') as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "exchange": self.exchange,
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "results": wfa_result.to_dict()
                },
                f,
                indent=2,
                default=str
            )
        
        logger.info(f"WFA results saved to {filepath}")
        return str(filepath)
