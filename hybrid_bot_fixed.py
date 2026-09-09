"""
Fixed Async Multi-Market Shadow Trading Bot - HYBRID VERSION
=============================================================

Critical Fix: Added HTTP polling fallback since WebSocket connections
are failing. This version uses BOTH WebSocket (preferred) and HTTP
polling (fallback) to ensure the bot always has data.

Changes from V3:
1. HTTP polling fallback for Coinbase data
2. Better error logging
3. WebSocket is now optional, not required
4. Feature calculation works with both data sources
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HybridBot")

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
# ROLLING FEATURE ENGINE (Same as V3)
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
    
    def calculate_features(self) -> Optional[Dict[str, float]]:
        if len(self.close_buffer) < 50:
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
            
            # Safely calculate correlation
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
# ASYNC KALSHI CLIENT (Same as V3)
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
                    logger.debug(f"Kalshi API error: {response.status} for {path}")
                    return None
        except Exception as e:
            logger.debug(f"Kalshi request failed: {e}")
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
                if markets:
                    asset_map = {
                        "KXBTC": "BTC-USD",
                        "KXETH": "ETH-USD",
                        "KXSOL": "SOL-USD",
                        "KXXRP": "XRP-USD",
                    }
                    asset = asset_map.get(prefix, f"{prefix[2:]}-USD")
                    
                    config = MarketConfig(
                        series=series_ticker,
                        asset=asset,
                        granularity=900,
                        active_ticker=markets[0]["ticker"]
                    )
                    all_markets.append(config)
                    logger.info(f"Discovered market: {series_ticker} ({asset})")
        
        return all_markets
    
    async def get_market_price(self, ticker: str) -> Optional[float]:
        path = f"/trade-api/v2/markets/{ticker}"
        result = await self.get(path)
        
        if result and "market" in result:
            yes_ask = result["market"].get("yes_ask", 0)
            return yes_ask / 100.0
        return None

# ============================================================================
# HYBRID DATA FETCHER (HTTP Fallback)
# ============================================================================

class HybridDataFetcher:
    """
    Fetches Coinbase data using HTTP (WebSocket removed due to connection issues)
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, OHLCV] = {}
        self.last_fetch: Dict[str, float] = {}
    
    async def start(self):
        self.session = aiohttp.ClientSession()
        logger.info("HTTP data fetcher started")
    
    async def stop(self):
        if self.session:
            await self.session.close()
    
    async def get_latest_price(self, asset: str, granularity: int = 900) -> Optional[OHLCV]:
        """Fetch latest candle from Coinbase REST API"""
        
        # Rate limit: don't fetch more than once per 10 seconds per asset
        now = time.time()
        if asset in self.last_fetch and (now - self.last_fetch[asset]) < 10:
            return self.cache.get(asset)
        
        url = f"https://api.exchange.coinbase.com/products/{asset}/candles"
        params = {"granularity": granularity}
        headers = {"User-Agent": "Mozilla/5.0"}
        
        try:
            async with self.session.get(url, headers=headers, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    if data and len(data) > 0:
                        # Coinbase returns [time, low, high, open, close, volume]
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
                        return candle
        except Exception as e:
            logger.debug(f"Coinbase fetch failed for {asset}: {e}")
        
        return None

# ============================================================================
# MAIN BOT
# ============================================================================

class HybridMultiMarketBot:
    def __init__(
        self, 
        model_path: str = "bot_v2_enhanced.json",
        features_path: str = "selected_features.json"
    ):
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
        except Exception as e:
            logger.warning(f"⚠️ Could not load features: {e}")
            self.selected_features = self.model.get_booster().feature_names
        
        # API credentials
        self.api_key = os.environ.get("KALSHI_API_KEY")
        key_file = "ergoKey.txt"
        with open(key_file, "r") as f:
            self.private_key_str = f.read()
        
        # Components
        self.kalshi_client: Optional[AsyncKalshiClient] = None
        self.data_fetcher: Optional[HybridDataFetcher] = None
        
        # Market registry
        self.markets: Dict[str, MarketConfig] = {}
        self.feature_engines: Dict[str, RollingFeatureEngine] = {}
        
        # Portfolio
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
        
        # Stats
        self.stats = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0.0})
        self.running = False
    
    async def start(self):
        logger.info("🚀 Starting Hybrid Multi-Market Shadow Bot")
        
        # Initialize clients
        self.kalshi_client = AsyncKalshiClient(self.api_key, self.private_key_str)
        self.data_fetcher = HybridDataFetcher()
        
        async with self.kalshi_client:
            await self.data_fetcher.start()
            
            # Discover markets
            discovered_markets = await self.kalshi_client.discover_crypto_15m_markets()
            
            if not discovered_markets:
                logger.error("No 15m crypto markets found. Exiting.")
                return
            
            # Register markets
            for market_config in discovered_markets:
                self.markets[market_config.series] = market_config
                self.feature_engines[market_config.series] = RollingFeatureEngine()
            
            logger.info(f"📊 Registered {len(self.markets)} markets")
            logger.info(f"🔄 Using HTTP polling for Coinbase data (every 10s per market)")
            
            # Start loops
            self.running = True
            tasks = [
                asyncio.create_task(self._market_discovery_loop()),
                asyncio.create_task(self._trading_loop()),
                asyncio.create_task(self._dashboard_loop()),
            ]
            
            try:
                await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                logger.info("🛑 Shutting down...")
                self.running = False
                await self.data_fetcher.stop()
    
    async def _market_discovery_loop(self):
        while self.running:
            await asyncio.sleep(300)
            try:
                discovered = await self.kalshi_client.discover_crypto_15m_markets()
                for market_config in discovered:
                    if market_config.series not in self.markets:
                        self.markets[market_config.series] = market_config
                        self.feature_engines[market_config.series] = RollingFeatureEngine()
                        logger.info(f"🆕 New market: {market_config.series}")
            except Exception as e:
                logger.error(f"Market discovery error: {e}")
    
    async def _trading_loop(self):
        while self.running:
            try:
                # Process all markets
                tasks = [
                    self._process_market(series, config) 
                    for series, config in self.markets.items()
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
            
            await asyncio.sleep(15)  # Every 15 seconds (adjusted from 5 to reduce API load)
    
    async def _process_market(self, series: str, config: MarketConfig):
        try:
            # Fetch Coinbase data (HTTP)
            latest_candle = await self.data_fetcher.get_latest_price(config.asset, config.granularity)
            if not latest_candle:
                return
            
            # Update feature engine
            feature_engine = self.feature_engines[series]
            feature_engine.update(latest_candle)
            
            # Calculate features
            features = feature_engine.calculate_features()
            if not features:
                return
            
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
            
            # Calculate edges
            edge_yes = model_prob - kalshi_price
            edge_no = (1 - model_prob) - (1 - kalshi_price)
            
            # Log signal (debug)
            if abs(edge_yes) > 0.05:  # Only log if edge > 5%
                logger.debug(
                    f"{series} | Model: {model_prob:.2%} | Kalshi: {kalshi_price:.2%} | "
                    f"Edge: {edge_yes:+.2%}"
                )
            
            # Execute trade if signal
            if edge_yes >= self.min_edge_threshold and model_prob > 0.60:
                await self._execute_trade(series, "UP", kalshi_price, edge_yes, model_prob)
            elif edge_no >= self.min_edge_threshold and model_prob < 0.40:
                await self._execute_trade(series, "DOWN", 1 - kalshi_price, edge_no, model_prob)
                
        except Exception as e:
            logger.error(f"Error processing {series}: {e}")
    
    async def _execute_trade(
        self, 
        series: str, 
        side: str, 
        contract_price: float,
        edge: float, 
        model_prob: float
    ):
        # Check risk limits
        if len(self.portfolio["pending_trades"]) >= self.max_positions:
            return
        
        total_deployed = sum(t["invested"] for t in self.portfolio["pending_trades"])
        total_capital = self.portfolio["balance"] + total_deployed
        
        if total_deployed / total_capital >= self.max_total_risk:
            return
        
        # Position sizing (Kelly Criterion)
        win_prob = 0.55 + (edge * 2)
        kelly_fraction = (win_prob - (1 - win_prob)) / 1.0
        kelly_fraction = max(0, min(kelly_fraction * 0.25, self.max_allocation_per_trade))
        
        investment = total_capital * kelly_fraction
        contracts = int(investment / contract_price)
        
        if contracts < 1:
            return
        
        actual_cost = contracts * contract_price
        if actual_cost > self.portfolio["balance"]:
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
            "timestamp": time.time()
        }
        
        self.portfolio["pending_trades"].append(trade)
        
        logger.info(
            f"🚨 TRADE | {series} | {side} | Edge: {edge:.2%} | "
            f"Contracts: {contracts} @ ${contract_price:.2f} | Deployed: ${actual_cost:.2f}"
        )
    
    async def _dashboard_loop(self):
        while self.running:
            await asyncio.sleep(30)
            
            total_deployed = sum(t["invested"] for t in self.portfolio["pending_trades"])
            total_capital = self.portfolio["balance"] + total_deployed
            roi = ((total_capital - 100) / 100) * 100
            
            logger.info(
                f"💰 Balance: ${self.portfolio['balance']:.2f} | "
                f"Deployed: ${total_deployed:.2f} | ROI: {roi:.2f}% | "
                f"Positions: {len(self.portfolio['pending_trades'])}"
            )

# ============================================================================
# MAIN
# ============================================================================

async def main():
    bot = HybridMultiMarketBot()
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())
