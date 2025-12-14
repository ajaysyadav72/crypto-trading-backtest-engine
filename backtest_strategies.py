"""Core backtest strategy implementation with EMA, RSI, ATR, and hybrid entry."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from config import StrategyConfig

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Trade record."""
    trade_id: int
    entry_time: datetime
    entry_price: float
    entry_qty: float
    entry_signal: str
    entry_type: str  # hybrid_tier_1/2/3/4
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_qty: Optional[float] = None
    exit_signal: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    net_pnl: float = 0.0
    bars_held: int = 0
    max_profit: float = 0.0
    max_loss: float = 0.0
    status: str = "open"  # open|closed


@dataclass
class BacktestResult:
    """Backtest result container."""
    exchange: str
    symbol: str
    timeframe: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    gross_pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    net_pnl: float = 0.0
    recovery_factor: float = 0.0
    profit_factor: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)


class EMAIndicator:
    """Exponential Moving Average calculation."""
    
    @staticmethod
    def calculate(data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA using vectorized operations."""
        if len(data) < period:
            return np.full(len(data), np.nan)
        
        ema = np.zeros(len(data))
        multiplier = 2 / (period + 1)
        
        # Initial SMA
        ema[period - 1] = np.mean(data[:period])
        
        # EMA calculation
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
        
        # Fill earlier values with NaN
        ema[:period - 1] = np.nan
        return ema


class RSIIndicator:
    """Relative Strength Index calculation."""
    
    @staticmethod
    def calculate(data: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI using vectorized operations."""
        if len(data) < period + 1:
            return np.full(len(data), np.nan)
        
        # Calculate price changes
        deltas = np.diff(data)
        
        # Separate gains and losses
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Initialize arrays
        avg_gains = np.zeros(len(data))
        avg_losses = np.zeros(len(data))
        rsi = np.zeros(len(data))
        
        # Initial averages
        avg_gains[period] = np.mean(gains[:period])
        avg_losses[period] = np.mean(losses[:period])
        
        # Wilder's smoothing
        for i in range(period + 1, len(data)):
            avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i - 1]) / period
        
        # Calculate RSI
        rs = np.divide(avg_gains[period:], avg_losses[period:],
                      where=avg_losses[period:] != 0,
                      out=np.zeros_like(avg_losses[period:]))
        rsi[period:] = 100 - (100 / (1 + rs))
        
        # Fill earlier values with NaN
        rsi[:period] = np.nan
        return rsi


class ATRIndicator:
    """Average True Range calculation."""
    
    @staticmethod
    def calculate(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Calculate ATR using vectorized operations."""
        # Calculate True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Calculate ATR (SMA of TR)
        atr = np.zeros(len(tr))
        atr[period - 1] = np.mean(tr[:period])
        
        # Wilder's smoothing
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        
        atr[:period - 1] = np.nan
        return atr


class VWAPIndicator:
    """Volume Weighted Average Price."""
    
    @staticmethod
    def calculate(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray
    ) -> np.ndarray:
        """Calculate VWAP."""
        typical_price = (high + low + close) / 3
        cumulative_tp_vol = np.cumsum(typical_price * volume)
        cumulative_vol = np.cumsum(volume)
        vwap = cumulative_tp_vol / cumulative_vol
        return vwap


class BacktestEngine:
    """Main backtest engine for strategy execution."""
    
    def __init__(
        self,
        candles: List[Dict],
        strategy_config: StrategyConfig,
        order_qty: float,
        exchange: str = "BINANCE",
        symbol: str = "BTCUSDT",
        timeframe: str = "1H"
    ):
        """Initialize backtest engine."""
        self.candles = candles
        self.config = strategy_config
        self.order_qty = order_qty
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Convert candles to DataFrame
        self.df = pd.DataFrame(candles)
        self.df['time'] = pd.to_datetime(self.df['time'], unit='ms')
        self.df.set_index('time', inplace=True)
        self.df = self.df.sort_index()
        
        # Prepare numpy arrays
        self.open = self.df['open'].values
        self.high = self.df['high'].values
        self.low = self.df['low'].values
        self.close = self.df['close'].values
        self.volume = self.df['volume'].values
        
        # Calculate indicators
        self._calculate_indicators()
        
        # Trading state
        self.trades: List[Trade] = []
        self.open_trade: Optional[Trade] = None
        self.trade_id = 0
        self.equity = 0.0
        self.cash = 100000.0  # Initial capital
    
    def _calculate_indicators(self) -> None:
        """Calculate all required indicators."""
        # EMA
        self.ema_fast = EMAIndicator.calculate(self.close, self.config.ema_fast)
        self.ema_slow = EMAIndicator.calculate(self.close, self.config.ema_slow)
        self.ema_trend = EMAIndicator.calculate(self.close, self.config.ema_trend)
        
        # RSI
        self.rsi = RSIIndicator.calculate(self.close, self.config.rsi_period)
        
        # ATR
        self.atr = ATRIndicator.calculate(
            self.high, self.low, self.close, self.config.atr_period
        )
        
        # VWAP
        self.vwap = VWAPIndicator.calculate(
            self.high, self.low, self.close, self.volume
        )
        
        logger.info("Indicators calculated")
    
    def run(self) -> BacktestResult:
        """Execute backtest."""
        logger.info(
            f"Starting backtest: {self.symbol} {self.timeframe} "
            f"({len(self.candles)} candles)"
        )
        
        equity_curve = []
        drawdown_curve = []
        max_equity = self.cash
        
        # Main backtest loop
        for i in range(max(self.config.ema_trend + 10, 100), len(self.close)):
            if self.open_trade is None:
                # Check entry signals
                if self._should_enter_long(i):
                    self._enter_long(i)
            else:
                # Check exit signals
                if self._should_exit_long(i):
                    self._exit_long(i)
            
            # Update equity
            position_value = 0
            if self.open_trade:
                position_value = self.open_trade.entry_qty * self.close[i]
            
            current_equity = self.cash + position_value
            equity_curve.append(current_equity)
            
            # Calculate drawdown
            max_equity = max(max_equity, current_equity)
            drawdown = (current_equity - max_equity) / max_equity * 100
            drawdown_curve.append(drawdown)
        
        # Close any open position
        if self.open_trade:
            self._exit_long(len(self.close) - 1)
        
        # Calculate metrics
        result = self._calculate_metrics(equity_curve, drawdown_curve)
        logger.info(f"Backtest complete: {result.win_rate:.2f}% win rate")
        
        return result
    
    def _should_enter_long(self, i: int) -> bool:
        """Check if should enter long position."""
        # Trend filter
        if self.config.enable_trend_filter:
            if self.ema_fast[i] <= self.ema_slow[i]:
                return False
        
        # RSI filter
        if self.config.enable_rsi_filter:
            if self.rsi[i] >= self.config.rsi_long_threshold:
                return False
        
        # EMA crossover signal
        if (self.ema_fast[i - 1] <= self.ema_slow[i - 1] and
            self.ema_fast[i] > self.ema_slow[i]):
            return True
        
        return False
    
    def _should_exit_long(self, i: int) -> bool:
        """Check if should exit long position."""
        if not self.open_trade:
            return False
        
        current_price = self.close[i]
        entry_price = self.open_trade.entry_price
        
        # Stop loss
        sl_price = entry_price - (self.atr[i] * self.config.atr_sl_multiplier)
        if current_price <= sl_price:
            self.open_trade.exit_signal = "stop_loss"
            return True
        
        # Take profit
        tp_price = entry_price + (self.atr[i] * self.config.atr_tp_multiplier)
        if current_price >= tp_price:
            self.open_trade.exit_signal = "take_profit"
            return True
        
        # Trailing stop loss
        if self.config.trailing_sl_enabled:
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct > self.config.trailing_sl_threshold:
                if current_price < self.open_trade.max_profit * (1 - 0.02):
                    self.open_trade.exit_signal = "trailing_sl"
                    return True
        
        return False
    
    def _enter_long(self, i: int) -> None:
        """Enter long position."""
        self.trade_id += 1
        entry_price = self.close[i]
        fees = entry_price * self.order_qty * 0.001  # 0.1% fees
        
        self.open_trade = Trade(
            trade_id=self.trade_id,
            entry_time=self.df.index[i],
            entry_price=entry_price,
            entry_qty=self.order_qty,
            entry_signal="ema_cross_rsi",
            entry_type="hybrid_tier_1",
            fees=fees
        )
        
        logger.debug(
            f"Trade #{self.trade_id} ENTRY: {entry_price} "
            f"({self.df.index[i]})"
        )
    
    def _exit_long(self, i: int) -> None:
        """Exit long position."""
        if not self.open_trade:
            return
        
        exit_price = self.close[i]
        exit_fees = exit_price * self.open_trade.entry_qty * 0.001
        
        # Calculate P&L
        gross_pnl = (
            (exit_price - self.open_trade.entry_price) *
            self.open_trade.entry_qty
        )
        net_pnl = gross_pnl - self.open_trade.fees - exit_fees
        
        self.open_trade.exit_time = self.df.index[i]
        self.open_trade.exit_price = exit_price
        self.open_trade.exit_qty = self.open_trade.entry_qty
        self.open_trade.pnl = gross_pnl
        self.open_trade.pnl_pct = (gross_pnl / (self.open_trade.entry_price *
                                               self.open_trade.entry_qty))
        self.open_trade.fees += exit_fees
        self.open_trade.net_pnl = net_pnl
        self.open_trade.bars_held = i - self.trades.__len__() if self.trades else i
        self.open_trade.status = "closed"
        
        self.trades.append(self.open_trade)
        self.cash += net_pnl
        
        logger.debug(
            f"Trade #{self.open_trade.trade_id} EXIT: {exit_price} "
            f"P&L: {net_pnl:.2f}"
        )
        
        self.open_trade = None
    
    def _calculate_metrics(
        self,
        equity_curve: List[float],
        drawdown_curve: List[float]
    ) -> BacktestResult:
        """Calculate backtest metrics."""
        result = BacktestResult(
            exchange=self.exchange,
            symbol=self.symbol,
            timeframe=self.timeframe,
            trades=self.trades,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve
        )
        
        if not self.trades:
            return result
        
        # Trade statistics
        result.total_trades = len(self.trades)
        result.winning_trades = sum(1 for t in self.trades if t.net_pnl > 0)
        result.losing_trades = sum(1 for t in self.trades if t.net_pnl <= 0)
        result.win_rate = (result.winning_trades / result.total_trades * 100
                          if result.total_trades > 0 else 0)
        
        # P&L metrics
        result.gross_pnl = sum(t.pnl for t in self.trades)
        result.fees = sum(t.fees for t in self.trades)
        result.net_pnl = sum(t.net_pnl for t in self.trades)
        
        # Sharpe ratio
        if equity_curve:
            returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
            if len(returns) > 0 and np.std(returns) > 0:
                result.sharpe_ratio = (
                    np.mean(returns) / np.std(returns) * np.sqrt(252)
                )
        
        # Max drawdown
        if drawdown_curve:
            result.max_drawdown = min(drawdown_curve)
        
        # Recovery factor
        if result.max_drawdown != 0:
            result.recovery_factor = (
                -result.net_pnl / abs(result.max_drawdown)
            )
        
        # Profit factor
        wins = sum(t.pnl for t in self.trades if t.pnl > 0)
        losses = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        if losses > 0:
            result.profit_factor = wins / losses
        
        return result
