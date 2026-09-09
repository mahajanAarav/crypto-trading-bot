"""
Asynchronous Multi-Market Shadow Trading Bot
============================================

Architecture:
- Event-driven WebSocket data ingestion (Coinbase)
- Async REST execution (Kalshi API)
- Dynamic market discovery (auto-detects all 15m crypto markets)
- High-performance rolling feature calculations (deque + NumPy)
- Non-blocking logging
- Order book depth modeling

Performance targets:
- <50ms feature calculation latency
- <100ms signal generation to execution
- Support 10+ concurrent markets without degradation
"""

import asyncio
import aiohttp
import json
import base64
import logging
import numpy as np
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Deque
import time
import os
from dataclasses import dataclass, field
from enum import Enum

from xgboost import XGBClassifier
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ============================================================================
# CONFIGURATION & DATA STRUCTURES
# ============================================================================

@dataclass
class MarketConfig:
    """Configuration for a single market"""
    series: str  # e.g., "KXBTC15M"
    asset: str   # e.g., "BTC-USD"
    granularity: int  # 900 for 15m
    active_ticker: Optional[str] = None
    last_price: float = 0.0
    order_book_depth: Dict[str, float] = field(default_factory=dict)

@dataclass
class OHLCV:
    """Single candle data point"""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float

class SignalType(Enum):
    """Trade signal types"""
    LONG = "UP"
    SHORT = "DOWN"
    HOLD = "HOLD"

@dataclass
class TradeSignal:
    """Generated trade signal"""
    market: str
    signal: SignalType
    model_prob: float
    edge: float
    confidence: float
    timestamp: float

# ============================================================================
# ASYNC LOGGING (NON-BLOCKING)
# ============================================================================

class AsyncLogger:
    """Non-blocking logger using asyncio queue"""
    
    def __init__(self, log_file: str = "async_bot.log"):
        self.queue = asyncio.Queue()
        self.log_file = log_file
        self.console_logger = logging.getLogger("AsyncBot")
        self.console_logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.console_logger.addHandler(ch)
        
        self.running = False
    
    async def start(self):
        """Start the logging worker"""
        self.running = True
        asyncio.create_task(self._log_worker())
        self.info("AsyncLogger started")
    
    async def _log_worker(self):
        """Background worker that writes logs"""
        with open(self.log_file, 'a') as f:
            while self.running:
                try:
                    message = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_entry = f"[{timestamp}] {message}\n"
                    f.write(log_entry)
                    f.flush()
                except asyncio.TimeoutError:
                    continue
    
    def info(self, message: str):
        """Log info message"""
        self.console_logger.info(message)
        try:
            self.queue.put_nowait(f"INFO: {message}")
        except asyncio.QueueFull:
            pass
    
    def error(self, message: str):
        """Log error message"""
        self.console_logger.error(message)
        try:
            self.queue.put_nowait(f"ERROR: {message}")
        except asyncio.QueueFull:
            pass
    
    def debug(self, message: str):
        """Log debug message"""
        self.console_logger.debug(message)
        try:
            self.queue.put_nowait(f"DEBUG: {message}")
        except asyncio.QueueFull:
            pass

    async def stop(self):
        """Stop the logger"""
        self.running = False
        await asyncio.sleep(1.1)  # Let worker finish

# ============================================================================
# HIGH-PERFORMANCE ROLLING FEATURE ENGINE
# ============================================================================

class RollingFeatureEngine:
    """
    Optimized feature calculation using deque and NumPy
    
    Performance: O(1) for appends, O(n) for rolling calculations
    Memory: Fixed size circular buffers
    """
    
    def __init__(self, max_window: int = 200):
        self.max_window = max_window
        
        # Circular buffers using deque (O(1) append/popleft)
        self.close_buffer: Deque[float] = deque(maxlen=max_window)
        self.high_buffer: Deque[float] = deque(maxlen=max_window)
        self.low_buffer: Deque[float] = deque(maxlen=max_window)
        self.open_buffer: Deque[float] = deque(maxlen=max_window)
        self.volume_buffer: Deque[float] = deque(maxlen=max_window)
        
        # Feature cache
        self.feature_cache: Dict[str, float] = {}
        self.last_update = 0.0
    
    def update(self, candle: OHLCV):
        """Update buffers with new candle - O(1) operation"""
        self.close_buffer.append(candle.close)
        self.high_buffer.append(candle.high)
        self.low_buffer.append(candle.low)
        self.open_buffer.append(candle.open)
        self.volume_buffer.append(candle.volume)
        self.last_update = candle.timestamp
    
    def calculate_features(self) -> Optional[Dict[str, float]]:
        """
        Calculate all features using vectorized NumPy operations
        
        Returns: Dict of feature values or None if insufficient data
        """
        if len(self.close_buffer) < 50:
            return None
        
        # Convert deques to NumPy arrays for vectorized operations
        close = np.array(self.close_buffer)
        high = np.array(self.high_buffer)
        low = np.array(self.low_buffer)
        open_prices = np.array(self.open_buffer)
        volume = np.array(self.volume_buffer)
        
        features = {}
        
        # === PRICE-BASED FEATURES (vectorized) ===
        for period in [1, 2, 3, 5, 10, 15, 20]:
            if len(close) > period:
                features[f"return_{period}"] = (close[-1] / close[-period-1] - 1)
        
        # Moving averages
        if len(close) >= 50:
            features["sma_5"] = np.mean(close[-5:])
            features["sma_10"] = np.mean(close[-10:])
            features["sma_20"] = np.mean(close[-20:])
            features["sma_50"] = np.mean(close[-50:])
            
            # EMA calculation (optimized)
            features["ema_10"] = self._ema(close, 10)
            features["ema_20"] = self._ema(close, 20)
            features["ema_50"] = self._ema(close, 50)
            
            # MA distances
            features["sma_dist_10"] = close[-1] / features["sma_10"] - 1
            features["sma_dist_20"] = close[-1] / features["sma_20"] - 1
            features["ema_dist_20"] = close[-1] / features["ema_20"] - 1
            
            # Crossovers
            features["ma_cross_5_10"] = 1.0 if features["sma_5"] > features["sma_10"] else 0.0
            features["ma_cross_10_20"] = 1.0 if features["sma_10"] > features["sma_20"] else 0.0
        
        # === VOLATILITY FEATURES ===
        returns_1 = np.diff(close) / close[:-1]
        
        if len(returns_1) >= 30:
            features["volatility_10"] = np.std(returns_1[-10:])
            features["volatility_20"] = np.std(returns_1[-20:])
            features["volatility_30"] = np.std(returns_1[-30:])
            
            # Bollinger Bands
            for period in [20, 50]:
                if len(close) >= period:
                    rolling_mean = np.mean(close[-period:])
                    rolling_std = np.std(close[-period:])
                    features[f"bb_position_{period}"] = (close[-1] - rolling_mean) / (2 * rolling_std + 1e-10)
                    features[f"bb_width_{period}"] = (2 * rolling_std) / (rolling_mean + 1e-10)
            
            # ATR (Average True Range)
            if len(close) >= 14:
                tr = np.maximum(
                    high[1:] - low[1:],
                    np.maximum(
                        np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:] - close[:-1])
                    )
                )
                features["atr_14"] = np.mean(tr[-14:])
                if len(tr) >= 20:
                    features["atr_20"] = np.mean(tr[-20:])
        
        # === MOMENTUM FEATURES ===
        for period in [7, 14, 21]:
            if len(close) >= period + 1:
                delta = np.diff(close[-period-1:])
                gains = np.where(delta > 0, delta, 0)
                losses = np.where(delta < 0, -delta, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                rs = avg_gain / (avg_loss + 1e-10)
                features[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        
        # Momentum
        for period in [5, 10, 20]:
            if len(close) > period:
                features[f"momentum_{period}"] = close[-1] - close[-period-1]
                features[f"roc_{period}"] = (close[-1] - close[-period-1]) / (close[-period-1] + 1e-10)
        
        # === VOLUME FEATURES ===
        if len(volume) >= 20:
            features["volume_ma_10"] = np.mean(volume[-10:])
            features["volume_ma_20"] = np.mean(volume[-20:])
            features["volume_ratio_10"] = volume[-1] / (features["volume_ma_10"] + 1e-10)
            features["volume_ratio_20"] = volume[-1] / (features["volume_ma_20"] + 1e-10)
            
            # Volume-price correlation
            if len(close) >= 20:
                features["volume_price_corr"] = np.corrcoef(volume[-20:], close[-20:])[0, 1]
        
        # === TREND FEATURES ===
        if "ema_20" in features and "sma_10" in features:
            features["trend_strength"] = abs(features["ema_20"] - features["sma_10"])
            features["trend_regime"] = 1.0 if features["ema_20"] > features["sma_10"] else 0.0
        
        if "momentum_10" in features and len(close) > 11:
            prev_momentum = close[-2] - close[-12]
            features["momentum_accel"] = features["momentum_10"] - prev_momentum
        
        if "return_1" in features and "volatility_20" in features:
            features["price_vol_ratio"] = features["return_1"] / (features["volatility_20"] + 1e-10)
        
        # === REGIME DETECTION ===
        if "volatility_20" in features and len(returns_1) >= 100:
            vol_mean_100 = np.mean([np.std(returns_1[i:i+20]) for i in range(len(returns_1)-100, len(returns_1)-20)])
            features["high_vol_regime"] = 1.0 if features["volatility_20"] > vol_mean_100 else 0.0
        
        if "bb_width_20" in features and len(close) >= 70:
            bb_widths = []
            for i in range(len(close)-50, len(close)):
                if i >= 20:
                    window = close[i-20:i]
                    bb_widths.append(2 * np.std(window) / (np.mean(window) + 1e-10))
            if bb_widths:
                features["consolidation_regime"] = 1.0 if features["bb_width_20"] < np.percentile(bb_widths, 30) else 0.0
        
        # === PATTERN FEATURES ===
        if len(high) >= 2:
            features["higher_high"] = 1.0 if high[-1] > high[-2] else 0.0
            features["lower_low"] = 1.0 if low[-1] < low[-2] else 0.0
        
        if len(close) >= 2 and len(open_prices) >= 1:
            gap = (open_prices[-1] - close[-2]) / (close[-2] + 1e-10)
            features["gap_up"] = 1.0 if gap > 0.005 else 0.0
            features["gap_down"] = 1.0 if gap < -0.005 else 0.0
        
        # === LAGGED FEATURES ===
        for lag in [1, 2, 3]:
            if f"return_1" in features and len(returns_1) > lag:
                features[f"return_1_lag{lag}"] = returns_1[-lag-1]
            if f"rsi_14" in features and len(close) >= 14 + lag + 1:
                # Approximate lagged RSI
                features[f"rsi_14_lag{lag}"] = features["rsi_14"]  # Simplified
            if f"volume_ratio_10" in features and len(volume) > lag:
                features[f"volume_ratio_10_lag{lag}"] = volume[-lag-1] / (np.mean(volume[-lag-11:-lag-1]) + 1e-10)
        
        self.feature_cache = features
        return features
    
    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        """Calculate EMA efficiently"""
        if len(data) < period:
            return np.mean(data)
        alpha = 2 / (period + 1)
        ema = data[-period]  # Start with SMA
        for price in data[-period+1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

# ============================================================================
# ORDER BOOK DEPTH MODEL
# ============================================================================

class OrderBookDepthModel:
    """
    Estimates slippage based on order book depth
    
    Without real book data, we model:
    - Spread widening under high volatility
    - Size-based slippage
    - Market impact cost
    """
    
    def __init__(self):
        self.base_spread = 0.02  # 2% base spread
        self.slippage_per_contract = 0.0001  # 0.01% per contract
    
    def estimate_execution_price(
        self, 
        mid_price: float, 
        side: SignalType, 
        size: int,
        volatility: float = 0.0
    ) -> Tuple[float, float]:
        """
        Estimate actual execution price accounting for slippage
        
        Returns: (execution_price, total_slippage)
        """
        # Dynamic spread based on volatility
        dynamic_spread = self.base_spread * (1 + volatility * 10)
        
        # Market impact (square root model)
        market_impact = self.slippage_per_contract * np.sqrt(size)
        
        # Total slippage
        total_slippage = (dynamic_spread / 2) + market_impact
        
        # Execution price
        if side == SignalType.LONG:
            execution_price = mid_price * (1 + total_slippage)
        else:
            execution_price = mid_price * (1 - total_slippage)
        
        return execution_price, total_slippage

# ============================================================================
# WEBSOCKET MANAGER FOR COINBASE
# ============================================================================

class CoinbaseWebSocketManager:
    """
    Event-driven WebSocket manager for Coinbase price feeds
    
    Architecture:
    - Maintains persistent WebSocket connections
    - Auto-reconnects on disconnect
    - Streams real-time ticker data
    - Updates market state on each message
    """
    
    def __init__(self, logger: AsyncLogger):
        self.logger = logger
        self.ws_url = "wss://ws-feed.exchange.coinbase.com"
        self.sessions: Dict[str, aiohttp.ClientSession] = {}
        self.ws_connections: Dict[str, aiohttp.ClientWebSocketResponse] = {}
        self.subscribed_products: List[str] = []
        self.running = False
        
        # Market state updated by WebSocket
        self.market_state: Dict[str, OHLCV] = {}
        self.state_lock = asyncio.Lock()
    
    async def start(self, products: List[str]):
        """Start WebSocket connections for given products"""
        self.subscribed_products = products
        self.running = True
        
        for product in products:
            asyncio.create_task(self._maintain_connection(product))
        
        self.logger.info(f"WebSocket manager started for {len(products)} products")
    
    async def _maintain_connection(self, product: str):
        """Maintain WebSocket connection with auto-reconnect"""
        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(self.ws_url) as ws:
                        self.logger.info(f"WebSocket connected: {product}")
                        
                        # Subscribe to ticker channel
                        subscribe_message = {
                            "type": "subscribe",
                            "product_ids": [product],
                            "channels": ["ticker"]
                        }
                        await ws.send_json(subscribe_message)
                        
                        # Event loop: process messages
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._on_message(product, msg.json())
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                self.logger.error(f"WebSocket error: {product}")
                                break
            
            except Exception as e:
                self.logger.error(f"WebSocket connection failed for {product}: {e}")
            
            # Reconnect backoff
            if self.running:
                await asyncio.sleep(5)
    
    async def _on_message(self, product: str, data: dict):
        """
        Event handler for WebSocket messages
        
        Critical path: <5ms processing time
        """
        if data.get("type") != "ticker":
            return
        
        try:
            # Extract price data
            price = float(data.get("price", 0))
            volume_24h = float(data.get("volume_24h", 0))
            
            # Create OHLCV snapshot (WebSocket doesn't provide OHLC, so we approximate)
            candle = OHLCV(
                timestamp=time.time(),
                open=price,  # Approximate
                high=price,
                low=price,
                close=price,
                volume=volume_24h
            )
            
            # Update shared state (thread-safe)
            async with self.state_lock:
                self.market_state[product] = candle
        
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")
    
    async def get_latest_price(self, product: str) -> Optional[OHLCV]:
        """Get latest price from WebSocket stream"""
        async with self.state_lock:
            return self.market_state.get(product)
    
    async def stop(self):
        """Stop all WebSocket connections"""
        self.running = False
        await asyncio.sleep(1)
        self.logger.info("WebSocket manager stopped")

# ============================================================================
# ASYNC KALSHI CLIENT
# ============================================================================

class AsyncKalshiClient:
    """
    Asynchronous Kalshi API client
    
    Features:
    - Connection pooling via aiohttp
    - Request signing
    - Rate limiting
    - Retry logic
    """
    
    def __init__(self, api_key: str, private_key_str: str, logger: AsyncLogger):
        self.api_key = api_key
        self.private_key = serialization.load_pem_private_key(
            private_key_str.encode(), 
            password=None
        )
        self.logger = logger
        self.base_url = "https://api.elections.kalshi.com"
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.rate_limiter = asyncio.Semaphore(10)  # 10 concurrent requests max
        self.last_request_time = 0.0
        self.min_request_interval = 0.1  # 100ms between requests
    
    async def __aenter__(self):
        """Context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            await self.session.close()
    
    def _sign_request(self, method: str, path: str, timestamp_str: str) -> str:
        """Sign Kalshi API request"""
        message = f"{timestamp_str}{method}{path}"
        signature = self.private_key.sign(
            message.encode(), 
            padding.PKCS1v15(), 
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()
    
    async def _rate_limit(self):
        """Enforce rate limiting"""
        async with self.rate_limiter:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - elapsed)
            self.last_request_time = time.time()
    
    async def get(self, path: str, params: Optional[Dict] = None) -> Optional[dict]:
        """Async GET request"""
        await self._rate_limit()
        
        timestamp_str = str(int(time.time() * 1000))
        signature = self._sign_request("GET", path, timestamp_str)
        
        headers = {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_str,
            "Content-Type": "application/json"
        }
        
        try:
            async with self.session.get(
                f"{self.base_url}{path}", 
                headers=headers, 
                params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    self.logger.error(f"Kalshi API error: {response.status} for {path}")
                    return None
        except Exception as e:
            self.logger.error(f"Kalshi request failed: {e}")
            return None
    
    async def discover_crypto_15m_markets(self) -> List[MarketConfig]:
        """
        Discover all active 15m crypto markets
        
        Returns: List of MarketConfig objects
        """
        all_markets = []
        crypto_series_prefixes = ["KXBTC", "KXETH", "KXSOL", "KXXRP", "KXDOGE", "KXADA", "KXAVAX"]
        
        for prefix in crypto_series_prefixes:
            series_ticker = f"{prefix}15M"
            path = f"/trade-api/v2/markets"
            params = {"series_ticker": series_ticker, "status": "open"}
            
            result = await self.get(path, params)
            if result and "markets" in result:
                markets = result["markets"]
                if markets:
                    # Extract asset from series ticker
                    asset_map = {
                        "KXBTC": "BTC-USD",
                        "KXETH": "ETH-USD",
                        "KXSOL": "SOL-USD",
                        "KXXRP": "XRP-USD",
                        "KXDOGE": "DOGE-USD",
                        "KXADA": "ADA-USD",
                        "KXAVAX": "AVAX-USD"
                    }
                    asset = asset_map.get(prefix, f"{prefix[2:]}-USD")
                    
                    config = MarketConfig(
                        series=series_ticker,
                        asset=asset,
                        granularity=900,  # 15 minutes
                        active_ticker=markets[0]["ticker"]
                    )
                    all_markets.append(config)
                    self.logger.info(f"Discovered market: {series_ticker} ({asset})")
        
        return all_markets
    
    async def get_market_price(self, ticker: str) -> Optional[float]:
        """Get current market price for a ticker"""
        path = f"/trade-api/v2/markets/{ticker}"
        result = await self.get(path)
        
        if result and "market" in result:
            yes_ask = result["market"].get("yes_ask", 0)
            return yes_ask / 100.0
        return None

# ============================================================================
# MULTI-MARKET ORCHESTRATOR
# ============================================================================

class AsyncMultiMarketBot:
    """
    Main orchestrator for asynchronous multi-market trading
    
    Architecture:
    - Dynamic market discovery on startup
    - WebSocket-driven price updates
    - Concurrent signal generation across all markets
    - Async trade execution
    - Non-blocking logging and persistence
    """
    
    def __init__(
        self, 
        model_path: str = "bot_v2_enhanced.json",
        features_path: str = "selected_features.json"
    ):
        # Initialize logger
        self.logger = AsyncLogger()
        
        # Load model
        self.model = XGBClassifier()
        try:
            self.model.load_model(model_path)
            print(f"✅ Model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
        
        # Load features
        try:
            with open(features_path, "r") as f:
                self.selected_features = json.load(f)
            print(f"✅ Loaded {len(self.selected_features)} features")
        except Exception as e:
            print(f"⚠️ Could not load features: {e}")
            self.selected_features = self.model.get_booster().feature_names
        
        # API credentials
        self.api_key = os.environ.get("KALSHI_API_KEY")
        key_file = "ergoKey.txt"
        with open(key_file, "r") as f:
            self.private_key_str = f.read()
        
        # Components
        self.kalshi_client: Optional[AsyncKalshiClient] = None
        self.ws_manager: Optional[CoinbaseWebSocketManager] = None
        self.order_book_model = OrderBookDepthModel()
        
        # Market registry
        self.markets: Dict[str, MarketConfig] = {}
        self.feature_engines: Dict[str, RollingFeatureEngine] = {}
        
        # Portfolio state
        self.portfolio = {
            "balance": 100.00,
            "pending_trades": [],
            "initial_balance": 100.00,
            "peak_balance": 100.00
        }
        
        # Risk parameters
        self.max_positions = 4
        self.max_allocation_per_trade = 0.15
        self.max_total_risk = 0.50
        self.min_edge_threshold = 0.10
        
        # Performance tracking
        self.stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
        
        self.running = False
    
    async def start(self):
        """Start the bot - main entry point"""
        await self.logger.start()
        self.logger.info("🚀 Starting Async Multi-Market Shadow Bot")
        
        # Initialize Kalshi client
        self.kalshi_client = AsyncKalshiClient(
            self.api_key, 
            self.private_key_str, 
            self.logger
        )
        
        async with self.kalshi_client:
            # Discover all 15m crypto markets
            discovered_markets = await self.kalshi_client.discover_crypto_15m_markets()
            
            if not discovered_markets:
                self.logger.error("No 15m crypto markets found. Exiting.")
                return
            
            # Register markets
            for market_config in discovered_markets:
                self.markets[market_config.series] = market_config
                self.feature_engines[market_config.series] = RollingFeatureEngine()
            
            self.logger.info(f"📊 Registered {len(self.markets)} markets")
            
            # Start WebSocket manager
            products = [m.asset for m in discovered_markets]
            self.ws_manager = CoinbaseWebSocketManager(self.logger)
            await self.ws_manager.start(products)
            
            # Start concurrent tasks
            self.running = True
            tasks = [
                asyncio.create_task(self._market_discovery_loop()),
                asyncio.create_task(self._trading_loop()),
                asyncio.create_task(self._dashboard_loop()),
            ]
            
            try:
                await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                self.logger.info("🛑 Shutting down...")
                self.running = False
                await self.ws_manager.stop()
                await self.logger.stop()
    
    async def _market_discovery_loop(self):
        """Periodic market discovery to find new 15m markets"""
        while self.running:
            await asyncio.sleep(300)  # Every 5 minutes
            
            try:
                discovered = await self.kalshi_client.discover_crypto_15m_markets()
                
                for market_config in discovered:
                    if market_config.series not in self.markets:
                        self.markets[market_config.series] = market_config
                        self.feature_engines[market_config.series] = RollingFeatureEngine()
                        self.logger.info(f"🆕 New market discovered: {market_config.series}")
            
            except Exception as e:
                self.logger.error(f"Market discovery error: {e}")
    
    async def _trading_loop(self):
        """Main trading loop - processes all markets concurrently"""
        while self.running:
            try:
                # Process all markets concurrently
                tasks = [
                    self._process_market(series, config) 
                    for series, config in self.markets.items()
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
                
            except Exception as e:
                self.logger.error(f"Trading loop error: {e}")
            
            await asyncio.sleep(5)  # 5-second cycle
    
    async def _process_market(self, series: str, config: MarketConfig):
        """Process a single market - generate signals and execute trades"""
        try:
            # Get latest price from WebSocket
            latest_candle = await self.ws_manager.get_latest_price(config.asset)
            if not latest_candle:
                return
            
            # Update feature engine
            feature_engine = self.feature_engines[series]
            feature_engine.update(latest_candle)
            
            # Calculate features
            features = feature_engine.calculate_features()
            if not features:
                return  # Insufficient data
            
            # Filter to selected features
            feature_vector = np.array([
                features.get(f, 0.0) for f in self.selected_features
            ]).reshape(1, -1)
            
            # Generate signal
            model_prob = self.model.predict_proba(feature_vector)[0][1]
            
            # Get Kalshi price
            kalshi_price = await self.kalshi_client.get_market_price(config.active_ticker)
            if not kalshi_price:
                return
            
            # Calculate edge
            edge_long = model_prob - kalshi_price
            edge_short = (1 - model_prob) - (1 - kalshi_price)
            
            # Generate signal
            signal = None
            edge = 0.0
            
            if edge_long >= self.min_edge_threshold and model_prob > 0.60:
                signal = SignalType.LONG
                edge = edge_long
            elif edge_short >= self.min_edge_threshold and model_prob < 0.40:
                signal = SignalType.SHORT
                edge = edge_short
            
            # Execute trade if signal generated
            if signal and signal != SignalType.HOLD:
                await self._execute_trade(series, signal, kalshi_price, edge, model_prob, features)
        
        except Exception as e:
            self.logger.error(f"Error processing {series}: {e}")
    
    async def _execute_trade(
        self, 
        series: str, 
        signal: SignalType, 
        kalshi_price: float,
        edge: float, 
        model_prob: float,
        features: Dict[str, float]
    ):
        """Execute a trade with risk management"""
        # Check risk limits
        if len(self.portfolio["pending_trades"]) >= self.max_positions:
            return
        
        total_deployed = sum(t["invested"] for t in self.portfolio["pending_trades"])
        total_capital = self.portfolio["balance"] + total_deployed
        
        if total_deployed / total_capital >= self.max_total_risk:
            return
        
        # Calculate position size using Kelly Criterion
        win_prob = 0.55 + (edge * 2)
        kelly_fraction = (win_prob - (1 - win_prob)) / 1.0
        kelly_fraction = max(0, min(kelly_fraction * 0.25, self.max_allocation_per_trade))
        
        investment = total_capital * kelly_fraction
        
        # Adjust for order book depth
        volatility = features.get("volatility_20", 0.0)
        execution_price, slippage = self.order_book_model.estimate_execution_price(
            kalshi_price, signal, int(investment / kalshi_price), volatility
        )
        
        contracts = int(investment / execution_price)
        if contracts < 1:
            return
        
        actual_cost = contracts * execution_price
        if actual_cost > self.portfolio["balance"]:
            return
        
        # Execute trade
        self.portfolio["balance"] -= actual_cost
        
        trade = {
            "market": series,
            "ticker": self.markets[series].active_ticker,
            "side": signal.value,
            "contracts": contracts,
            "entry_price": execution_price,
            "invested": actual_cost,
            "slippage": slippage,
            "edge": edge,
            "timestamp": time.time()
        }
        
        self.portfolio["pending_trades"].append(trade)
        
        self.logger.info(
            f"🚨 TRADE | {series} | {signal.value} | Edge: {edge:.2%} | "
            f"Contracts: {contracts} @ ${execution_price:.2f} | Slippage: {slippage:.2%}"
        )
    
    async def _dashboard_loop(self):
        """Display dashboard periodically"""
        while self.running:
            await asyncio.sleep(30)
            
            total_deployed = sum(t["invested"] for t in self.portfolio["pending_trades"])
            total_capital = self.portfolio["balance"] + total_deployed
            roi = ((total_capital - 100) / 100) * 100
            
            self.logger.info(
                f"💰 Balance: ${self.portfolio['balance']:.2f} | "
                f"Deployed: ${total_deployed:.2f} | ROI: {roi:.2f}% | "
                f"Positions: {len(self.portfolio['pending_trades'])}"
            )

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point"""
    bot = AsyncMultiMarketBot(
        model_path="bot_v2_enhanced.json",
        features_path="selected_features.json"
    )
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())
