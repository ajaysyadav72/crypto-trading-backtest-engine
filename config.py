"""Configuration management for backtest engine."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
from datetime import datetime
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv('./config/.env')


class AuthType(str, Enum):
    """API authentication types."""
    HMAC = "hmac"
    OAUTH2 = "oauth2"
    TOKEN = "token"


class OptimizationProfile(str, Enum):
    """Optimization speed profiles."""
    FAST = "FAST"
    BALANCED = "BALANCED"
    THOROUGH = "THOROUGH"


@dataclass
class ExchangeConfig:
    """Exchange configuration."""
    name: str
    base_url: str
    auth_type: AuthType
    rate_limit: int  # requests per minute
    timeout: int = 30  # seconds
    
    def __post_init__(self):
        if not self.name or not self.base_url:
            raise ValueError("Exchange name and base_url required")
        if self.rate_limit <= 0:
            raise ValueError("Rate limit must be positive")


@dataclass
class UserInputConfig:
    """User input configuration."""
    mode: str  # "SINGLE" | "MULTI"
    exchanges: List[str]
    symbols: List[str]
    order_qty: List[float]
    backtest_start: datetime
    backtest_end: datetime
    timeframes: List[str]
    
    def __post_init__(self):
        if self.mode not in ["SINGLE", "MULTI"]:
            raise ValueError("Mode must be SINGLE or MULTI")
        if self.backtest_end <= self.backtest_start:
            raise ValueError("End date must be after start date")
        if any(qty <= 0 for qty in self.order_qty):
            raise ValueError("Order quantities must be positive")


@dataclass
class StrategyConfig:
    """Strategy configuration with EMA, RSI, ATR parameters."""
    # EMA Parameters
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200
    
    # RSI Parameters
    rsi_period: int = 14
    rsi_long_threshold: int = 35
    rsi_short_threshold: int = 65
    
    # Risk Management
    atr_period: int = 14
    atr_sl_multiplier: float = 1.5
    atr_tp_multiplier: float = 2.0
    trailing_sl_threshold: float = 0.02  # 2%
    trailing_tp_threshold: float = 0.03  # 3%
    
    # Hybrid Entry Engine (5-Tier)
    enable_hybrid_tier1: bool = True
    enable_hybrid_tier2: bool = True
    enable_hybrid_tier3: bool = True
    enable_hybrid_tier4: bool = True
    ema_touch_tolerance: float = 0.0015  # 0.15%
    hybrid_tier1_timeout: int = 5
    hybrid_tier2_timeout: int = 3
    hybrid_tier3_timeout: int = 4
    hybrid_tier4_timeout: int = 1
    
    # Filters
    enable_trend_filter: bool = True
    enable_rsi_filter: bool = True
    enable_hybrid_engine: bool = True
    trailing_sl_enabled: bool = True
    trailing_tp_enabled: bool = True
    
    # Feature Toggles
    enable_optimization: bool = True
    enable_walk_forward: bool = True
    enable_monte_carlo: bool = True
    enable_dashboard: bool = True
    
    def __post_init__(self):
        """Validate configuration values."""
        if self.ema_fast >= self.ema_slow:
            raise ValueError("EMA_FAST must be < EMA_SLOW")
        if self.ema_slow >= self.ema_trend:
            raise ValueError("EMA_SLOW must be < EMA_TREND")
        if not (20 <= self.rsi_long_threshold <= 50):
            raise ValueError("RSI_LONG_THRESHOLD must be 20-50")
        if not (50 <= self.rsi_short_threshold <= 80):
            raise ValueError("RSI_SHORT_THRESHOLD must be 50-80")
        if not (0.5 <= self.atr_sl_multiplier <= 5.0):
            raise ValueError("ATR_SL_MULTIPLIER must be 0.5-5.0")
        if not (1.0 <= self.atr_tp_multiplier <= 5.0):
            raise ValueError("ATR_TP_MULTIPLIER must be 1.0-5.0")


@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    profile: OptimizationProfile = OptimizationProfile.BALANCED
    optuna_sampler: str = "TPE"
    num_trials: int = 150
    multi_fidelity: bool = True
    fidelity_levels: Tuple[float, float, float] = (0.33, 0.66, 1.0)
    parallelization_workers: int = 4
    
    def __post_init__(self):
        """Set trial count based on profile."""
        profile_trials = {
            OptimizationProfile.FAST: 50,
            OptimizationProfile.BALANCED: 150,
            OptimizationProfile.THOROUGH: 300,
        }
        if isinstance(self.profile, str):
            self.profile = OptimizationProfile(self.profile)
        if self.num_trials == 150:  # default
            self.num_trials = profile_trials[self.profile]
    
    def get_grid_delta(self) -> float:
        """Get parameter grid delta based on profile."""
        deltas = {
            OptimizationProfile.FAST: 0.10,
            OptimizationProfile.BALANCED: 0.15,
            OptimizationProfile.THOROUGH: 0.25,
        }
        return deltas[self.profile]


@dataclass
class WalForwardConfig:
    """Walk-forward analysis configuration."""
    is_length_days: int = 60
    oos_length_days: int = 20
    stride_days: int = 20
    min_trades_per_window: int = 5


@dataclass
class HardwareConfig:
    """Hardware benchmarking configuration."""
    benchmark_iterations: int = 100000
    cpu_timeout: int = 300  # seconds
    gpu_enabled: bool = True


# Predefined exchange configurations
EXCHANGE_REGISTRY = {
    "BINANCE": ExchangeConfig(
        name="BINANCE",
        base_url="https://api.binance.com",
        auth_type=AuthType.HMAC,
        rate_limit=1200,
        timeout=30
    ),
    "DELTA": ExchangeConfig(
        name="DELTA",
        base_url="https://api.india.delta.exchange",
        auth_type=AuthType.HMAC,
        rate_limit=6000,
        timeout=30
    ),
    "ZERODHA": ExchangeConfig(
        name="ZERODHA",
        base_url="https://api.kite.trade",
        auth_type=AuthType.OAUTH2,
        rate_limit=6000,
        timeout=30
    ),
    "DHAN": ExchangeConfig(
        name="DHAN",
        base_url="https://api.dhan.co/v2",
        auth_type=AuthType.TOKEN,
        rate_limit=6000,
        timeout=30
    ),
}


# Timeframe validation
VALID_TIMEFRAMES = {"1M", "5M", "15M", "1H", "4H", "1D", "1W"}


# Global settings
class GlobalConfig:
    """Global application settings."""
    DATA_DIR = "./data"
    LOGS_DIR = "./logs"
    CONFIG_DIR = "./config"
    CACHE_DIR = os.path.join(DATA_DIR, "cache")
    RAW_LOGS_DIR = os.path.join(DATA_DIR, "raw_logs")
    RESULTS_DIR = os.path.join(DATA_DIR, "results")
    EXPORTS_DIR = os.path.join(DATA_DIR, "exports")
    CHECKPOINTS_DIR = os.path.join(DATA_DIR, "checkpoints")
    
    @staticmethod
    def create_directories():
        """Create all necessary directories."""
        dirs = [
            GlobalConfig.DATA_DIR,
            GlobalConfig.LOGS_DIR,
            GlobalConfig.CONFIG_DIR,
            GlobalConfig.CACHE_DIR,
            GlobalConfig.RAW_LOGS_DIR,
            GlobalConfig.RESULTS_DIR,
            GlobalConfig.EXPORTS_DIR,
            GlobalConfig.CHECKPOINTS_DIR,
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)
