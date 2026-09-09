"""
Debug Version - Extensive Logging to Diagnose Trading Issues
============================================================

This version adds detailed logging to show:
- Data fetch attempts and results
- Feature calculation status
- Model predictions
- Signal generation
- Why trades are/aren't executed
"""

import asyncio
import aiohttp
import json
import base64
import logging
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Optional
import time
import os
from dataclasses import dataclass

from xgboost import XGBClassifier
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ENHANCED LOGGING
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DebugBot")

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MarketConfig:
    series: str
    asset: str
    granularity: int
    active_ticker: Optional[str] = None

@dataclass
class OHLCV:
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float

# ============================================================================
# ROLLING FEATURE ENGINE
# ============================================================================

class RollingFeatureEngine:
    def __init__(self, max_window: int = 200):
        self.max_window = max_window
        self.close_buffer: deque = deque(maxlen=max_window)
        self.high_buffer: deque = deque(maxlen=max_window)
        self.low_buffer: deque = deque(maxlen=max_window)
        self.volume_buffer: deque = deque(maxlen=max_window)
        self.last_update = 0.0
    
    def update(self, candle: OHLCV):
        self.close_buffer.append(candle.close)
        self.high_buffer.append(candle.high)
        self.low_buffer.append(candle.low)
        self.volume_buffer.append(candle.volume)
        self.last_update = candle.timestamp
    
    def get_buffer_size(self) -> int:
        return len(self.close_buffer)
    
    def calculate_features(self) -> Optional[Dict[str, float]]:
        buffer_size = len(self.close_buffer)
        
        if buffer_size < 50:
            logger.debug(f"Insufficient data: {buffer_size}/50 candles")
            return None
        
        close = np.array(self.close_buffer)
        high = np.array(self.high_buffer)
        low = np.array(self.low_buffer)
        volume = np.array(self.volume_buffer)
        
        features = {}
        
        # Returns
        for period in [1, 2, 3, 5, 10, 15, 20]:
            if len(close) > period:
                features[f"return_{period}"] = (close[-1] / close[-period-1] - 1)
        
        # Moving averages
        if len(close) >= 50:
            features["sma_5"] = np.mean(close[-5:])
            features["sma_10"] = np.mean(close[-10:])
            features["sma_20"] = np.mean(close[-20:])
            features["sma_50"] = np.mean(close[-50:])
            features["ema_10"] = self._ema(close, 10)
            features["ema_20"] = self._ema(close, 20)
            features["ema_50"] = self._ema(close, 50)
            
            features["sma_dist_10"] = close[-1] / features["sma_10"] - 1
            features["sma_dist_20"] = close[-1] / features["sma_20"] - 1
            features["ema_dist_20"] = close[-1] / features["ema_20"] - 1
            features["ma_cross_5_10"] = 1.0 if features["sma_5"] > features["sma_10"] else 0.0
            features["ma_cross_10_20"] = 1.0 if features["sma_10"] > features["sma_20"] else 0.0
        
        # Volatility
        returns_1 = np.diff(close) / close[:-1]
        if len(returns_1) >= 30:
            features["volatility_10"] = np.std(returns_1[-10:])
            features["volatility_20"] = np.std(returns_1[-20:])
            features["volatility_30"] = np.std(returns_1[-30:])
            
            for period in [20, 50]:
                if len(close) >= period:
                    rolling_mean = np.mean(close[-period:])
                    rolling_std = np.std(close[-period:])
                    features[f"bb_position_{period}"] = (close[-1] - rolling_mean) / (2 * rolling_std + 1e-10)
                    features[f"bb_width_{period}"] = (2 * rolling_std) / (rolling_mean + 1e-10)
            
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
        
        # Momentum (RSI)
        for period in [7, 14, 21]:
            if len(close) >= period + 1:
                delta = np.diff(close[-period-1:])
                gains = np.where(delta > 0, delta, 0)
                losses = np.where(delta < 0, -delta, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                rs = avg_gain / (avg_loss + 1e-10)
                features[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        
        for period in [5, 10, 20]:
            if len(close) > period:
                features[f"momentum_{period}"] = close[-1] - close[-period-1]
                features[f"roc_{period}"] = (close[-1] - close[-period-1]) / (close[-period-1] + 1e-10)
        
        # Volume
        if len(volume) >= 20:
            features["volume_ma_10"] = np.mean(volume[-10:])
            features["volume_ma_20"] = np.mean(volume[-20:])
            features["volume_ratio_10"] = volume[-1] / (features["volume_ma_10"] + 1e-10)
            features["volume_ratio_20"] = volume[-1] / (features["volume_ma_20"] + 1e-10)
            
            try:
                features["volume_price_corr"] = np.corrcoef(volume[-20:], close[-20:])[0, 1]
                if np.isnan(features["volume_price_corr"]):
                    features["volume_price_corr"] = 0.0
            except:
                features["volume_price_corr"] = 0.0
        
        # Trend
        if "ema_20" in features and "sma_10" in features:
            features["trend_strength"] = abs(features["ema_20"] - features["sma_10"])
            features["trend_regime"] = 1.0 if features["ema_20"] > features["sma_10"] else 0.0
        
        if "momentum_10" in features and len(close) > 11:
            prev_momentum = close[-2] - close[-12]
            features["momentum_accel"] = features["momentum_10"] - prev_momentum
        
        if "return_1" in features and "volatility_20" in features:
            features["price_vol_ratio"] = features["return_1"] / (features["volatility_20"] + 1e-10)
        
        # Regimes
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
        
        # Patterns
        if len(high) >= 2:
            features["higher_high"] = 1.0 if high[-1] > high[-2] else 0.0
            features["lower_low"] = 1.0 if low[-1] < low[-2] else 0.0
        
        if len(close) >= 2:
            gap = (close[-1] - close[-2]) / (close[-2] + 1e-10)
            features["gap_up"] = 1.0 if gap > 0.005 else 0.0
            features["gap_down"] = 1.0 if gap < -0.005 else 0.0
        
        # Lagged
        for lag in [1, 2, 3]:
            if len(returns_1) > lag:
                features[f"return_1_lag{lag}"] = returns_1[-lag-1]
            if f"rsi_14" in features:
                features[f"rsi_14_lag{lag}"] = features["rsi_14"]
            if f"volume_ratio_10" in features and len(volume) > lag:
                features[f"volume_ratio_10_lag{lag}"] = volume[-lag-1] / (np.mean(volume[-lag-11:-lag-1]) + 1e-10)
        
        logger.debug(f"Calculated {len(features)} features from {buffer_size} candles")
        return features
    
    @staticmethod
    def _ema(data: np.ndarray, period: int) -> float:
        if len(data) < period:
            return np.mean(data)
        alpha = 2 / (period + 1)
        ema = data[-period]
        for price in data[-period+1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

# ============================================================================
# ASYNC KALSHI CLIENT
# ============================================================================

class AsyncKalshiClient:
    def __init__(self, api_key: str, private_key_str: str):
        self.api_key = api_key
        self.private_key = serialization.load_pem_private_key(
            private_key_str.encode(), 
            password=None
        )
        self.base_url = "https://api.elections.kalshi.com"
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiter = asyncio.Semaphore(10)
        self.last_request_time = 0.0
        self.min_request_interval = 0.1
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _sign_request(self, method: str, path: str, timestamp_str: str) -> str:
        message = f"{timestamp_str}{method}{path}"
        signature = self.private_key.sign(
            message.encode(), 
            padding.PKCS1v15(), 
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()
    
    async def _rate_limit(self):
        async with self.rate_limiter:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_request_interval:
                await asyncio.sleep(self.min_request_interval - elapsed)
            self.last_request_time = time.time()
    
    async def get(self, path: str, params: Optional[Dict] = None) -> Optional[dict]:
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
                    logger.warning(f"Kalshi API error: {response.status} for {path}")
                    return None
        except Exception as e:
            logger.error(f"Kalshi request failed for {path}: {e}")
            return None
    
    async def discover_crypto_15m_markets(self) -> List[MarketConfig]:
        all_markets = []
        crypto_series_prefixes = ["KXBTC", "KXETH", "KXSOL", "KXXRP"]
        
        for prefix in crypto_series_prefixes:
            series_ticker = f"{prefix}15M"
            path = f"/trade-api/v2/markets"
            params = {"series_ticker": series_ticker, "status": "open"}
            
            result = await self.get(path, params)
            if result and "markets" in result:
                markets = result["markets"]
                
                # Filter out markets with no bids/asks (inactive)
                active_markets = [
                    m for m in markets 
                    if m.get("yes_ask", 0) > 0 or m.get("no_ask", 0) > 0
                ]
                
                if active_markets:
                    # Sort by expiration time (most recent first) to get the freshest market
                    active_markets.sort(
                        key=lambda m: m.get("close_time", "9999-12-31T23:59:59Z"),
                        reverse=False  # Earliest expiration first (soonest to settle)
                    )
                    
                    asset_map = {
                        "KXBTC": "BTC-USD",
                        "KXETH": "ETH-USD",
                        "KXSOL": "SOL-USD",
                        "KXXRP": "XRP-USD",
                    }
                    asset = asset_map.get(prefix, f"{prefix[2:]}-USD")
                    
                    # Use the first active market (soonest to expire)
                    selected_market = active_markets[0]
                    
                    config = MarketConfig(
                        series=series_ticker,
                        asset=asset,
                        granularity=900,
                        active_ticker=selected_market["ticker"]
                    )
                    all_markets.append(config)
                    
                    close_time = selected_market.get("close_time", "unknown")
                    yes_ask = selected_market.get("yes_ask", 0) / 100
                    logger.info(
                        f"✅ Discovered: {series_ticker} → {asset} "
                        f"(ticker: {selected_market['ticker']}, "
                        f"expires: {close_time[:16]}, "
                        f"YES: ${yes_ask:.2f})"
                    )
                else:
                    logger.warning(f"⚠️ No active markets for {series_ticker}")
        
        return all_markets
    
    async def get_market_price(self, ticker: str) -> Optional[float]:
        path = f"/trade-api/v2/markets/{ticker}"
        result = await self.get(path)
        
        if result and "market" in result:
            market = result["market"]
            
            # Check if market is closed/settled
            status = market.get("status", "")
            if status not in ["open", "active"]:
                logger.warning(f"Market {ticker} is not active (status: {status})")
                return None
            
            yes_ask = market.get("yes_ask", 0)
            
            # Kalshi returns cents, so 0 means either no ask or market is closed
            if yes_ask == 0:
                logger.warning(f"Market {ticker} has no YES ask (likely expired/inactive)")
                return None
            
            price = yes_ask / 100.0
            logger.debug(f"Kalshi price for {ticker}: ${price:.4f}")
            return price
        else:
            logger.warning(f"Failed to get market data for {ticker}")
        return None

# ============================================================================
# HTTP DATA FETCHER
# ============================================================================

class HTTPDataFetcher:
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, OHLCV] = {}
        self.last_fetch: Dict[str, float] = {}
        self.fetch_count: Dict[str, int] = defaultdict(int)
    
    async def start(self):
        self.session = aiohttp.ClientSession()
        logger.info("✅ HTTP data fetcher started")
    
    async def stop(self):
        if self.session:
            await self.session.close()
    
    async def get_latest_price(self, asset: str, granularity: int = 900) -> Optional[OHLCV]:
        """Fetch latest candle from Coinbase"""
        
        # Rate limit: max 1 fetch per 10 seconds per asset
        now = time.time()
        if asset in self.last_fetch and (now - self.last_fetch[asset]) < 10:
            logger.debug(f"Cache hit for {asset} (last fetch {now - self.last_fetch[asset]:.1f}s ago)")
            return self.cache.get(asset)
        
        url = f"https://api.exchange.coinbase.com/products/{asset}/candles"
        params = {"granularity": granularity}
        headers = {"User-Agent": "Mozilla/5.0"}
        
        try:
            logger.debug(f"Fetching {asset} from Coinbase...")
            async with self.session.get(url, headers=headers, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        latest = data[0]
                        candle = OHLCV(
                            timestamp=latest[0],
                            open=float(latest[3]),
                            high=float(latest[2]),
                            low=float(latest[1]),
                            close=float(latest[4]),
                            volume=float(latest[5])
                        )
                        self.cache[asset] = candle
                        self.last_fetch[asset] = now
                        self.fetch_count[asset] += 1
                        
                        logger.info(
                            f"📊 Coinbase fetch #{self.fetch_count[asset]} | {asset} | "
                            f"Close: ${candle.close:,.2f} | Vol: {candle.volume:,.0f}"
                        )
                        return candle
                else:
                    logger.error(f"Coinbase API error {response.status} for {asset}")
        except Exception as e:
            logger.error(f"Coinbase fetch failed for {asset}: {e}")
        
        return None

# ============================================================================
# MAIN BOT
# ============================================================================

class DebugBot:
    def __init__(
        self, 
        model_path: str = "bot_v2_enhanced.json",
        features_path: str = "selected_features.json"
    ):
        logger.info("="*70)
        logger.info("DEBUG BOT INITIALIZATION")
        logger.info("="*70)
        
        # Load model
        self.model = XGBClassifier()
        try:
            self.model.load_model(model_path)
            logger.info(f"✅ Model loaded: {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
        
        # Load features
        try:
            with open(features_path, "r") as f:
                self.selected_features = json.load(f)
            logger.info(f"✅ Loaded {len(self.selected_features)} features")
            logger.debug(f"Features: {self.selected_features[:10]}...")  # Show first 10
        except Exception as e:
            logger.warning(f"⚠️ Could not load features: {e}")
            self.selected_features = self.model.get_booster().feature_names
        
        # API credentials
        self.api_key = os.environ.get("KALSHI_API_KEY")
        if not self.api_key:
            logger.error("❌ KALSHI_API_KEY not set!")
            raise ValueError("Missing API key")
        
        key_file = "ergoKey.txt"
        with open(key_file, "r") as f:
            self.private_key_str = f.read()
        logger.info("✅ API credentials loaded")
        
        # Components
        self.kalshi_client: Optional[AsyncKalshiClient] = None
        self.data_fetcher: Optional[HTTPDataFetcher] = None
        
        # Markets
        self.markets: Dict[str, MarketConfig] = {}
        self.feature_engines: Dict[str, RollingFeatureEngine] = {}
        
        # Portfolio
        self.portfolio = {
            "balance": 100.00,
            "pending_trades": [],
        }
        
        # Risk
        self.max_positions = 4
        self.max_allocation_per_trade = 0.15
        self.min_edge_threshold = 0.10
        
        self.running = False
        self.iteration = 0
        
        logger.info("="*70)
    
    async def start(self):
        logger.info("🚀 Starting Debug Bot")
        
        self.kalshi_client = AsyncKalshiClient(self.api_key, self.private_key_str)
        self.data_fetcher = HTTPDataFetcher()
        
        async with self.kalshi_client:
            await self.data_fetcher.start()
            
            # Discover markets
            discovered_markets = await self.kalshi_client.discover_crypto_15m_markets()
            
            if not discovered_markets:
                logger.error("❌ No markets found!")
                return
            
            # Register
            for market_config in discovered_markets:
                self.markets[market_config.series] = market_config
                self.feature_engines[market_config.series] = RollingFeatureEngine()
            
            logger.info(f"📊 Monitoring {len(self.markets)} markets")
            logger.info("⏳ Waiting for data accumulation (need 50+ candles, ~8-10 min)...")
            
            # Start loop
            self.running = True
            self.last_market_refresh = 0
            
            try:
                while self.running:
                    self.iteration += 1
                    logger.info(f"\n{'='*70}")
                    logger.info(f"ITERATION #{self.iteration}")
                    logger.info(f"{'='*70}")
                    
                    # Refresh markets every 5 minutes (20 iterations * 15s = 5min)
                    if self.iteration % 20 == 0 or self.iteration == 1:
                        logger.info("\n🔄 Refreshing active markets...")
                        discovered_markets = await self.kalshi_client.discover_crypto_15m_markets()
                        
                        for market_config in discovered_markets:
                            old_ticker = self.markets.get(market_config.series, {}).active_ticker if hasattr(self.markets.get(market_config.series, {}), 'active_ticker') else None
                            new_ticker = market_config.active_ticker
                            
                            if old_ticker != new_ticker:
                                logger.info(f"   📍 {market_config.series}: {old_ticker} → {new_ticker}")
                            
                            self.markets[market_config.series] = market_config
                        
                        logger.info(f"✅ Markets refreshed\n")
                    
                    # Process markets
                    for series, config in self.markets.items():
                        await self._process_market(series, config)
                    
                    # Dashboard
                    self._show_status()
                    
                    logger.info(f"{'='*70}\n")
                    await asyncio.sleep(15)
                    
            except KeyboardInterrupt:
                logger.info("🛑 Shutting down...")
                self.running = False
                await self.data_fetcher.stop()
    
    async def _process_market(self, series: str, config: MarketConfig):
        logger.info(f"\n--- Processing {series} ---")
        
        try:
            # 1. Fetch data
            logger.debug(f"Step 1: Fetching {config.asset} data...")
            latest_candle = await self.data_fetcher.get_latest_price(config.asset, config.granularity)
            if not latest_candle:
                logger.warning(f"❌ No data fetched for {series}")
                return
            
            # 2. Update buffer
            logger.debug(f"Step 2: Updating feature buffer...")
            feature_engine = self.feature_engines[series]
            feature_engine.update(latest_candle)
            buffer_size = feature_engine.get_buffer_size()
            logger.info(f"📈 Buffer size: {buffer_size}/50 candles")
            
            # 3. Calculate features
            logger.debug(f"Step 3: Calculating features...")
            features = feature_engine.calculate_features()
            if not features:
                logger.warning(f"⚠️ Insufficient data for {series} ({buffer_size}/50 candles)")
                return
            
            logger.info(f"✅ Features calculated: {len(features)} features")
            
            # 4. Prepare model input
            logger.debug(f"Step 4: Preparing model input...")
            feature_vector = np.array([
                features.get(f, 0.0) for f in self.selected_features
            ]).reshape(1, -1)
            
            # Check for missing features
            missing = [f for f in self.selected_features if f not in features]
            if missing:
                logger.warning(f"⚠️ Missing features: {missing[:5]}...")
            
            # 5. Model prediction
            logger.debug(f"Step 5: Running model prediction...")
            model_prob = self.model.predict_proba(feature_vector)[0][1]
            logger.info(f"🤖 Model probability: {model_prob:.4f} ({model_prob*100:.2f}%)")
            
            # 6. Get Kalshi price
            logger.debug(f"Step 6: Fetching Kalshi price...")
            kalshi_price = await self.kalshi_client.get_market_price(config.active_ticker)
            if not kalshi_price:
                logger.warning(f"❌ No Kalshi price for {series}")
                return
            
            logger.info(f"💰 Kalshi YES price: ${kalshi_price:.4f} ({kalshi_price*100:.2f}%)")
            
            # 7. Calculate edges
            edge_yes = model_prob - kalshi_price
            edge_no = (1 - model_prob) - (1 - kalshi_price)
            
            logger.info(f"📊 Edge (YES): {edge_yes:+.4f} ({edge_yes*100:+.2f}%)")
            logger.info(f"📊 Edge (NO):  {edge_no:+.4f} ({edge_no*100:+.2f}%)")
            
            # 8. Signal generation
            logger.debug(f"Step 7: Checking signal conditions...")
            
            if edge_yes >= self.min_edge_threshold and model_prob > 0.60:
                logger.info(f"🚨 SIGNAL GENERATED: LONG (YES)")
                logger.info(f"   Conditions met: edge={edge_yes:.2%} >= {self.min_edge_threshold:.2%}, prob={model_prob:.2%} > 60%")
                await self._execute_trade(series, "UP", kalshi_price, edge_yes, model_prob)
            elif edge_no >= self.min_edge_threshold and model_prob < 0.40:
                logger.info(f"🚨 SIGNAL GENERATED: SHORT (NO)")
                logger.info(f"   Conditions met: edge={edge_no:.2%} >= {self.min_edge_threshold:.2%}, prob={model_prob:.2%} < 40%")
                await self._execute_trade(series, "DOWN", 1 - kalshi_price, edge_no, model_prob)
            else:
                logger.info(f"⏸️  No signal")
                logger.debug(f"   LONG rejected: edge={edge_yes:.2%} < {self.min_edge_threshold:.2%} OR prob={model_prob:.2%} <= 60%")
                logger.debug(f"   SHORT rejected: edge={edge_no:.2%} < {self.min_edge_threshold:.2%} OR prob={model_prob:.2%} >= 40%")
                
        except Exception as e:
            logger.error(f"❌ Error processing {series}: {e}", exc_info=True)
    
    async def _execute_trade(self, series: str, side: str, contract_price: float, edge: float, model_prob: float):
        logger.info(f"\n🎯 ATTEMPTING TRADE EXECUTION")
        
        # Check position limits
        current_positions = len(self.portfolio["pending_trades"])
        if current_positions >= self.max_positions:
            logger.warning(f"❌ Trade blocked: Max positions ({self.max_positions}) reached")
            return
        
        # Position sizing
        contracts = int(self.portfolio["balance"] * self.max_allocation_per_trade / contract_price)
        if contracts < 1:
            logger.warning(f"❌ Trade blocked: Not enough capital (would buy {contracts} contracts)")
            return
        
        actual_cost = contracts * contract_price
        if actual_cost > self.portfolio["balance"]:
            logger.warning(f"❌ Trade blocked: Insufficient balance (need ${actual_cost:.2f}, have ${self.portfolio['balance']:.2f})")
            return
        
        # Execute
        self.portfolio["balance"] -= actual_cost
        
        trade = {
            "market": series,
            "ticker": self.markets[series].active_ticker,
            "side": side,
            "contracts": contracts,
            "entry_price": contract_price,
            "invested": actual_cost,
            "edge": edge,
            "model_prob": model_prob,
            "timestamp": time.time()
        }
        
        self.portfolio["pending_trades"].append(trade)
        
        logger.info(f"✅ TRADE EXECUTED!")
        logger.info(f"   Market: {series}")
        logger.info(f"   Side: {side}")
        logger.info(f"   Contracts: {contracts} @ ${contract_price:.4f}")
        logger.info(f"   Cost: ${actual_cost:.2f}")
        logger.info(f"   Edge: {edge:.2%}")
        logger.info(f"   Remaining balance: ${self.portfolio['balance']:.2f}")
    
    def _show_status(self):
        total_deployed = sum(t["invested"] for t in self.portfolio["pending_trades"])
        total_capital = self.portfolio["balance"] + total_deployed
        roi = ((total_capital - 100) / 100) * 100
        
        logger.info(f"\n💼 PORTFOLIO STATUS")
        logger.info(f"   Balance: ${self.portfolio['balance']:.2f}")
        logger.info(f"   Deployed: ${total_deployed:.2f}")
        logger.info(f"   Total: ${total_capital:.2f}")
        logger.info(f"   ROI: {roi:+.2f}%")
        logger.info(f"   Positions: {len(self.portfolio['pending_trades'])}/{self.max_positions}")

# ============================================================================
# MAIN
# ============================================================================

async def main():
    bot = DebugBot()
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())
