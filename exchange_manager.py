"""Exchange API management and data fetching."""

import logging
import time
import os
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import hmac
import hashlib
from urllib.parse import urlencode
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from functools import wraps
import json
from pathlib import Path

from config import EXCHANGE_REGISTRY, ExchangeConfig, AuthType, GlobalConfig

# Setup logging
logger = logging.getLogger(__name__)


class BacktestError(Exception):
    """Base error for backtest system."""
    pass


class CredentialError(BacktestError):
    """Error with credentials."""
    pass


class APIError(BacktestError):
    """Error with API call."""
    pass


class RateLimitError(BacktestError):
    """Rate limit exceeded."""
    pass


class DataError(BacktestError):
    """Error with data validation."""
    pass


def retry_with_backoff(max_retries=3, base_delay=1):
    """Decorator for retry logic with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError, RateLimitError) as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
        return wrapper
    return decorator


class ExchangeManager:
    """Centralized exchange API management."""
    
    _instance = None  # Singleton pattern
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExchangeManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize ExchangeManager."""
        if self._initialized:
            return
        
        self.credentials: Dict[str, Dict[str, str]] = {}
        self.exchanges: Dict[str, ExchangeConfig] = EXCHANGE_REGISTRY.copy()
        self.session = self._create_session()
        self.cache: Dict[str, Any] = {}
        self._initialized = True
        
        logger.info("ExchangeManager initialized")
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry strategy."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def load_credentials(self, exchange_name: str) -> Dict[str, str]:
        """Load credentials from environment or config."""
        if exchange_name in self.credentials:
            return self.credentials[exchange_name]
        
        # Load from environment variables
        env_prefix = exchange_name.upper()
        api_key = os.getenv(f"{env_prefix}_API_KEY")
        api_secret = os.getenv(f"{env_prefix}_API_SECRET")
        access_token = os.getenv(f"{env_prefix}_ACCESS_TOKEN")
        
        if not api_key:
            raise CredentialError(
                f"Missing {env_prefix}_API_KEY in environment"
            )
        
        creds = {"api_key": api_key}
        if api_secret:
            creds["api_secret"] = api_secret
        if access_token:
            creds["access_token"] = access_token
        
        self.credentials[exchange_name] = creds
        logger.info(f"Credentials loaded for {exchange_name}")
        return creds
    
    def validate_connectivity(self, exchange_name: str) -> Dict[str, Any]:
        """Validate API connectivity for exchange."""
        try:
            creds = self.load_credentials(exchange_name)
            exchange = self.exchanges[exchange_name]
            
            # Test API call based on exchange
            if exchange_name == "BINANCE":
                response = self.session.get(
                    f"{exchange.base_url}/api/v3/account",
                    headers=self._build_headers(
                        exchange_name, "GET", "/api/v3/account"
                    ),
                    timeout=exchange.timeout
                )
            elif exchange_name == "DELTA":
                response = self.session.get(
                    f"{exchange.base_url}/api/v2/wallet",
                    headers=self._build_headers(
                        exchange_name, "GET", "/api/v2/wallet"
                    ),
                    timeout=exchange.timeout
                )
            elif exchange_name == "ZERODHA":
                response = self.session.get(
                    f"{exchange.base_url}/profile",
                    headers={
                        "Authorization": f"Bearer {creds.get('access_token')}"
                    },
                    timeout=exchange.timeout
                )
            elif exchange_name == "DHAN":
                response = self.session.get(
                    f"{exchange.base_url}/fundLimit",
                    headers={
                        "Authorization": f"Bearer {creds.get('access_token')}"
                    },
                    timeout=exchange.timeout
                )
            
            if response.status_code == 200:
                logger.info(f"{exchange_name} connectivity: OK")
                return {
                    "exchange": exchange_name,
                    "status": "READY",
                    "message": "API connection successful"
                }
            else:
                raise APIError(f"HTTP {response.status_code}: {response.text}")
        
        except Exception as e:
            logger.error(f"{exchange_name} connectivity failed: {str(e)}")
            return {
                "exchange": exchange_name,
                "status": "FAILED",
                "message": str(e)
            }
    
    def _build_headers(
        self,
        exchange_name: str,
        method: str,
        path: str,
        params: Optional[Dict] = None
    ) -> Dict[str, str]:
        """Build signed headers for API request."""
        creds = self.credentials.get(exchange_name)
        if not creds:
            raise CredentialError(f"No credentials for {exchange_name}")
        
        headers = {"Content-Type": "application/json"}
        
        if exchange_name == "BINANCE":
            timestamp = int(time.time() * 1000)
            query_string = urlencode(params or {}) if params else ""
            if query_string:
                query_string += f"&timestamp={timestamp}"
            else:
                query_string = f"timestamp={timestamp}"
            
            signature = hmac.new(
                creds["api_secret"].encode(),
                query_string.encode(),
                hashlib.sha256
            ).hexdigest()
            
            headers["X-MBX-APIKEY"] = creds["api_key"]
            headers["X-Signature"] = signature
        
        elif exchange_name == "DELTA":
            headers["X-API-Key"] = creds["api_key"]
        
        elif exchange_name == "ZERODHA":
            headers["Authorization"] = f"Bearer {creds.get('access_token')}"
        
        return headers
    
    @retry_with_backoff(max_retries=3, base_delay=1)
    def fetch_ohlc(
        self,
        exchange_name: str,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch OHLC candle data from exchange."""
        logger.info(
            f"Fetching {timeframe} candles for {symbol} from {exchange_name} "
            f"({start_date.date()} to {end_date.date()})"
        )
        
        exchange = self.exchanges[exchange_name]
        candles = []
        current_start = start_date
        
        # Pagination: fetch 1000 candles per request
        while current_start < end_date:
            try:
                if exchange_name == "BINANCE":
                    candles += self._fetch_binance_candles(
                        symbol, timeframe, current_start, end_date
                    )
                elif exchange_name == "DELTA":
                    candles += self._fetch_delta_candles(
                        symbol, timeframe, current_start, end_date
                    )
                elif exchange_name == "ZERODHA":
                    candles += self._fetch_zerodha_candles(
                        symbol, timeframe, current_start, end_date
                    )
                elif exchange_name == "DHAN":
                    candles += self._fetch_dhan_candles(
                        symbol, timeframe, current_start, end_date
                    )
                
                # Move to next batch
                if len(candles) > 0:
                    last_time = datetime.fromtimestamp(
                        candles[-1]["time"] / 1000
                    )
                    current_start = last_time + timedelta(minutes=1)
                else:
                    break
            
            except Exception as e:
                logger.error(f"Error fetching candles: {str(e)}")
                raise
        
        # Validate candles
        self._validate_candles(candles)
        logger.info(f"Fetched {len(candles)} candles for {symbol}")
        return candles
    
    def _fetch_binance_candles(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch candles from Binance."""
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "startTime": int(start_date.timestamp() * 1000),
            "endTime": int(end_date.timestamp() * 1000),
            "limit": 1000
        }
        
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        candles = []
        for item in response.json():
            candles.append({
                "time": item[0],
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[7])
            })
        
        return candles
    
    def _fetch_delta_candles(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch candles from Delta Exchange."""
        # Similar implementation for Delta
        return []
    
    def _fetch_zerodha_candles(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch candles from Zerodha."""
        # Similar implementation for Zerodha
        return []
    
    def _fetch_dhan_candles(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch candles from Dhan."""
        # Similar implementation for Dhan
        return []
    
    def _validate_candles(self, candles: List[Dict[str, Any]]) -> None:
        """Validate OHLC candle data."""
        if not candles:
            raise DataError("No candles fetched")
        
        for i, candle in enumerate(candles):
            # Check OHLC logic
            if candle["high"] < candle["low"]:
                raise DataError(
                    f"Candle {i}: High < Low ({candle['high']} < {candle['low']})"
                )
            
            if candle["high"] < candle["close"]:
                raise DataError(
                    f"Candle {i}: High < Close"
                )
            
            if candle["low"] > candle["close"]:
                raise DataError(
                    f"Candle {i}: Low > Close"
                )
            
            # Check volume
            if candle["volume"] < 0:
                raise DataError(f"Candle {i}: Negative volume")
            
            # Check for duplicates
            if i > 0 and candles[i]["time"] <= candles[i-1]["time"]:
                raise DataError(
                    f"Candle {i}: Duplicate or out-of-order timestamp"
                )
        
        logger.info(f"Validated {len(candles)} candles")


# Singleton instance
get_exchange_manager = ExchangeManager
