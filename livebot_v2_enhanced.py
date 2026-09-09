import os
import time
import json
import base64
import logging
import requests
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
from collections import defaultdict
import threading

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_live.log'),
        logging.StreamHandler()
    ]
)

class MultiMarketShadowBot:
    """
    Enhanced live trading bot with multi-market support
    Supports: BTC, ETH, XRP, SOL on 5m and 15m timeframes
    """
    
    def __init__(self, model_path="bot_v2_enhanced.json", features_path="selected_features.json"):
        # Market configuration
        self.markets = {
            "BTC_15M": {"series": "KXBTC15M", "asset": "BTC-USD", "granularity": 900},
            "BTC_5M": {"series": "KXBTC5M", "asset": "BTC-USD", "granularity": 300},
            "ETH_15M": {"series": "KXETH15M", "asset": "ETH-USD", "granularity": 900},
            "ETH_5M": {"series": "KXETH5M", "asset": "ETH-USD", "granularity": 300},
            # Add more markets as needed
            # "SOL_15M": {"series": "KXSOL15M", "asset": "SOL-USD", "granularity": 900},
            # "XRP_15M": {"series": "KXXRP15M", "asset": "XRP-USD", "granularity": 900},
        }
        
        # Active tickers for each market
        self.active_tickers = {market: None for market in self.markets}
        
        # File paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.portfolio_file = os.path.join(self.script_dir, "portfolio_v2.json")
        self.log_file = os.path.join(self.script_dir, "trade_log_v2.txt")
        self.key_file = os.path.join(self.script_dir, "ergoKey.txt")
        self.model_file = os.path.join(self.script_dir, model_path)
        self.features_file = os.path.join(self.script_dir, features_path)
        
        # Risk management parameters
        self.max_positions = 4  # Maximum concurrent positions across all markets
        self.max_allocation_per_trade = 0.15  # Max 15% per trade
        self.max_total_risk = 0.50  # Max 50% of bankroll at risk
        self.min_edge_threshold = 0.10  # Minimum edge to enter
        
        # Performance tracking
        self.performance_stats = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0,
            "by_market": defaultdict(lambda: {"trades": 0, "wins": 0, "pnl": 0})
        }
        
        # Load portfolio
        self.load_portfolio()
        
        # Load model
        logging.info("Loading trained model...")
        self.model = XGBClassifier()
        try:
            self.model.load_model(self.model_file)
            logging.info(f"✅ Model loaded: {self.model_file}")
        except Exception as e:
            logging.error(f"Failed to load model from {self.model_file}. Error: {e}")
            exit()
        
        # Load selected features
        try:
            with open(self.features_file, "r") as f:
                self.selected_features = json.load(f)
            logging.info(f"✅ Loaded {len(self.selected_features)} features")
        except Exception as e:
            logging.warning(f"Could not load features file: {e}. Using model's feature names.")
            self.selected_features = self.model.get_booster().feature_names

        # API credentials
        self.api_key = os.environ.get("KALSHI_API_KEY")
        try:
            with open(self.key_file, "r") as key_file:
                self.private_key_str = key_file.read()
        except FileNotFoundError:
            logging.error(f"Could not find {self.key_file}")
            exit()

        if not self.api_key:
            logging.error("Kalshi API Key missing! Export KALSHI_API_KEY")
            exit()

    def load_portfolio(self):
        """Load or initialize portfolio"""
        if os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, "r") as f:
                self.portfolio = json.load(f)
            logging.info(f"📂 Loaded portfolio: ${self.portfolio['balance']:.2f}")
        else:
            self.portfolio = {
                "balance": 100.00, 
                "pending_trades": [],
                "initial_balance": 100.00,
                "peak_balance": 100.00,
                "trades_count": 0
            }
            self.save_portfolio()
            self.write_log("--- NEW MULTI-MARKET SHADOW BOT INITIALIZED WITH $100.00 ---")

    def save_portfolio(self):
        """Save portfolio to disk"""
        with open(self.portfolio_file, "w") as f:
            json.dump(self.portfolio, f, indent=4)

    def write_log(self, message):
        """Write to trade log"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        with open(self.log_file, "a") as f:
            f.write(log_entry)
        logging.info(message)

    def display_dashboard(self):
        """Enhanced HUD with multi-market stats"""
        total_deployed = sum(t["invested"] for t in self.portfolio["pending_trades"])
        available = self.portfolio["balance"]
        total_capital = available + total_deployed
        
        # Calculate ROI
        initial = self.portfolio.get("initial_balance", 100.00)
        roi = ((total_capital - initial) / initial) * 100
        
        # Calculate max drawdown
        peak = self.portfolio.get("peak_balance", initial)
        if total_capital > peak:
            self.portfolio["peak_balance"] = total_capital
            peak = total_capital
            self.save_portfolio()
        drawdown = ((total_capital - peak) / peak) * 100 if peak > 0 else 0
        
        print("\n" + "="*70)
        print(f"{'MULTI-MARKET SHADOW BOT DASHBOARD':^70}")
        print("="*70)
        print(f"💰 Total Capital:    ${total_capital:>8.2f}  |  Available: ${available:>8.2f}")
        print(f"📊 Deployed:         ${total_deployed:>8.2f}  |  ROI: {roi:>7.2f}%")
        print(f"📈 Peak Balance:     ${peak:>8.2f}  |  Drawdown: {drawdown:>6.2f}%")
        print(f"🎯 Open Positions:   {len(self.portfolio['pending_trades']):>2}")
        print("-"*70)
        
        # Group trades by market
        trades_by_market = defaultdict(list)
        for trade in self.portfolio["pending_trades"]:
            market_key = trade.get("market", "UNKNOWN")
            trades_by_market[market_key].append(trade)
        
        if trades_by_market:
            print("ACTIVE POSITIONS:")
            for market, trades in trades_by_market.items():
                print(f"\n  {market}:")
                for i, trade in enumerate(trades, 1):
                    side_emoji = "🟢" if trade["side"] == "UP" else "🔴"
                    print(f"    {side_emoji} [{i}] {trade['contracts']} contracts @ ${trade['entry_price']:.2f}")
                    print(f"        Ticker: {trade['ticker']} | Invested: ${trade['invested']:.2f}")
        else:
            print("No active positions")
        
        print("="*70 + "\n")

    def sign_kalshi_request(self, method, path, timestamp_str):
        """Sign Kalshi API request"""
        key = serialization.load_pem_private_key(self.private_key_str.encode(), password=None)
        message = f"{timestamp_str}{method}{path}"
        signature = key.sign(message.encode(), padding.PKCS1v15(), hashes.SHA256())
        return base64.b64encode(signature).decode()

    def update_active_tickers(self):
        """Update active tickers for all markets"""
        updated = False
        for market_key, market_config in self.markets.items():
            series = market_config["series"]
            path = f"/trade-api/v2/markets?series_ticker={series}&status=open"
            url = f"https://api.elections.kalshi.com{path}"
            
            timestamp_str = str(int(time.time() * 1000))
            signature = self.sign_kalshi_request("GET", path, timestamp_str)
            headers = {
                "KALSHI-ACCESS-KEY": self.api_key,
                "KALSHI-ACCESS-SIGNATURE": signature,
                "KALSHI-ACCESS-TIMESTAMP": timestamp_str,
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    markets = response.json().get('markets', [])
                    if markets:
                        self.active_tickers[market_key] = markets[0]['ticker']
                        updated = True
                    else:
                        self.active_tickers[market_key] = None
            except Exception as e:
                logging.error(f"Failed to update ticker for {market_key}: {e}")
                
        return updated

    def fetch_kalshi_price(self, ticker):
        """Fetch current Kalshi price for a ticker"""
        path = f"/trade-api/v2/markets/{ticker}"
        url = f"https://api.elections.kalshi.com{path}"
        timestamp_str = str(int(time.time() * 1000))
        signature = self.sign_kalshi_request("GET", path, timestamp_str)
        
        headers = {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_str,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data['market']['yes_ask'] / 100
        except Exception as e:
            logging.debug(f"Price fetch failed for {ticker}: {e}")
        return None

    def fetch_coinbase_data(self, asset, granularity, periods=100):
        """Fetch recent Coinbase data for a specific asset"""
        url = f"https://api.exchange.coinbase.com/products/{asset}/candles"
        params = {"granularity": granularity}
        headers = {"User-Agent": "Mozilla/5.0"}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["time"], unit="s")
                df = df.sort_values("timestamp").reset_index(drop=True)
                return df.tail(periods)
        except Exception as e:
            logging.debug(f"Coinbase data fetch failed for {asset}: {e}")
        return None

    def engineer_current_features(self, df):
        """Engineer features from recent data (same as training)"""
        data = df.copy()
        
        # === PRICE-BASED FEATURES ===
        for period in [1, 2, 3, 5, 10, 15, 20]:
            data[f"return_{period}"] = data["close"].pct_change(period)
        
        data["sma_5"] = data["close"].rolling(5).mean()
        data["sma_10"] = data["close"].rolling(10).mean()
        data["sma_20"] = data["close"].rolling(20).mean()
        data["sma_50"] = data["close"].rolling(50).mean()
        data["ema_10"] = data["close"].ewm(span=10).mean()
        data["ema_20"] = data["close"].ewm(span=20).mean()
        data["ema_50"] = data["close"].ewm(span=50).mean()
        
        data["sma_dist_10"] = data["close"] / data["sma_10"] - 1
        data["sma_dist_20"] = data["close"] / data["sma_20"] - 1
        data["ema_dist_20"] = data["close"] / data["ema_20"] - 1
        data["ma_cross_5_10"] = (data["sma_5"] > data["sma_10"]).astype(int)
        data["ma_cross_10_20"] = (data["sma_10"] > data["sma_20"]).astype(int)
        
        # === VOLATILITY ===
        for period in [10, 20, 30]:
            data[f"volatility_{period}"] = data["return_1"].rolling(period).std()
        
        for period in [20, 50]:
            rolling_mean = data["close"].rolling(period).mean()
            rolling_std = data["close"].rolling(period).std()
            data[f"bb_position_{period}"] = (data["close"] - rolling_mean) / (2 * rolling_std)
            data[f"bb_width_{period}"] = (2 * rolling_std) / rolling_mean
        
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data["atr_14"] = true_range.rolling(14).mean()
        data["atr_20"] = true_range.rolling(20).mean()
        
        # === MOMENTUM ===
        for period in [7, 14, 21]:
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            data[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        
        for period in [5, 10, 20]:
            data[f"momentum_{period}"] = data["close"] - data["close"].shift(period)
            data[f"roc_{period}"] = (data["close"] - data["close"].shift(period)) / (data["close"].shift(period) + 1e-10)
        
        # === VOLUME ===
        data["volume_ma_10"] = data["volume"].rolling(10).mean()
        data["volume_ma_20"] = data["volume"].rolling(20).mean()
        data["volume_ratio_10"] = data["volume"] / (data["volume_ma_10"] + 1e-10)
        data["volume_ratio_20"] = data["volume"] / (data["volume_ma_20"] + 1e-10)
        data["volume_price_corr"] = data["volume"].rolling(20).corr(data["close"])
        
        # === TREND ===
        data["trend_strength"] = abs(data["ema_20"] - data["sma_10"])
        data["momentum_accel"] = data["momentum_10"].diff()
        data["price_vol_ratio"] = data["return_1"] / (data["volatility_20"] + 1e-10)
        
        # === REGIMES ===
        vol_mean_100 = data["volatility_20"].rolling(100).mean()
        data["high_vol_regime"] = (data["volatility_20"] > vol_mean_100).astype(int)
        data["trend_regime"] = (data["ema_20"] > data["sma_10"]).astype(int)
        data["consolidation_regime"] = (data["bb_width_20"] < data["bb_width_20"].rolling(50).quantile(0.3)).astype(int)
        
        # === PATTERNS ===
        data["higher_high"] = (data["high"] > data["high"].shift(1)).astype(int)
        data["lower_low"] = (data["low"] < data["low"].shift(1)).astype(int)
        data["gap_up"] = ((data["open"] - data["close"].shift(1)) / (data["close"].shift(1) + 1e-10) > 0.005).astype(int)
        data["gap_down"] = ((data["open"] - data["close"].shift(1)) / (data["close"].shift(1) + 1e-10) < -0.005).astype(int)
        
        # === LAGGED ===
        for lag in [1, 2, 3]:
            data[f"return_1_lag{lag}"] = data["return_1"].shift(lag)
            data[f"rsi_14_lag{lag}"] = data["rsi_14"].shift(lag)
            data[f"volume_ratio_10_lag{lag}"] = data["volume_ratio_10"].shift(lag)
        
        # Filter to only selected features
        available_features = [f for f in self.selected_features if f in data.columns]
        return data.iloc[-1:][available_features]

    def manage_open_positions(self, market_key):
        """Manage positions for a specific market"""
        ticker = self.active_tickers[market_key]
        if not ticker:
            return
            
        kalshi_price = self.fetch_kalshi_price(ticker)
        if kalshi_price is None:
            return
        
        live_no_price = 1.0 - kalshi_price
        
        for trade in self.portfolio["pending_trades"][:]:
            # Only manage trades for this market
            if trade.get("market") != market_key or trade["ticker"] != ticker:
                continue
                
            current_price = kalshi_price if trade["side"] == "UP" else live_no_price
            entry_price = trade["entry_price"]
            
            if entry_price == 0:
                continue
                
            net_profit_pct = (current_price - entry_price) / entry_price
            
            # Exit conditions
            take_profit = net_profit_pct >= 0.20
            stop_loss = net_profit_pct <= -0.15
            
            if take_profit or stop_loss:
                exit_value = trade["contracts"] * current_price
                profit = exit_value - trade["invested"]
                self.portfolio["balance"] += exit_value
                
                reason = "🎯 TAKE PROFIT" if take_profit else "🛑 STOP LOSS"
                self.write_log(
                    f"{reason} | {market_key} | {trade['side']} on {trade['ticker']} | "
                    f"Exit: ${current_price:.2f} (Net: {net_profit_pct:.2%}) | "
                    f"P&L: ${profit:.2f} | Balance: ${self.portfolio['balance']:.2f}"
                )
                
                # Update stats
                self.performance_stats["total_trades"] += 1
                self.performance_stats["by_market"][market_key]["trades"] += 1
                self.performance_stats["by_market"][market_key]["pnl"] += profit
                self.performance_stats["total_pnl"] += profit
                if profit > 0:
                    self.performance_stats["wins"] += 1
                    self.performance_stats["by_market"][market_key]["wins"] += 1
                else:
                    self.performance_stats["losses"] += 1
                
                self.portfolio["pending_trades"].remove(trade)
                self.save_portfolio()
                self.display_dashboard()

    def can_take_position(self):
        """Check if we can take a new position based on risk limits"""
        # Max positions limit
        if len(self.portfolio["pending_trades"]) >= self.max_positions:
            return False
        
        # Total risk limit
        total_deployed = sum(t["invested"] for t in self.portfolio["pending_trades"])
        total_capital = self.portfolio["balance"] + total_deployed
        if total_deployed / total_capital >= self.max_total_risk:
            return False
        
        return True

    def calculate_position_size(self, edge, kalshi_price):
        """Calculate optimal position size using Kelly Criterion (conservative)"""
        # Kelly fraction with 25% scaling for safety
        win_prob = 0.55 + (edge * 2)  # Estimate win probability from edge
        kelly_fraction = (win_prob - (1 - win_prob)) / 1.0  # Simplified Kelly
        kelly_fraction = max(0, min(kelly_fraction * 0.25, self.max_allocation_per_trade))
        
        # Calculate investment amount
        total_deployed = sum(t["invested"] for t in self.portfolio["pending_trades"])
        available = self.portfolio["balance"]
        max_investment = available * self.max_allocation_per_trade
        kelly_investment = (available + total_deployed) * kelly_fraction
        
        investment = min(max_investment, kelly_investment)
        contracts = int(investment / kalshi_price)
        
        return max(1, contracts)

    def execute_trade(self, market_key, side, contract_price, edge, model_prob):
        """Execute a shadow trade for a specific market"""
        if not self.can_take_position():
            logging.debug(f"Skipping {market_key} - risk limits reached")
            return
        
        ticker = self.active_tickers[market_key]
        if not ticker:
            return
        
        contracts = self.calculate_position_size(edge, contract_price)
        actual_cost = contracts * contract_price
        
        if actual_cost > self.portfolio["balance"]:
            logging.debug(f"Insufficient balance for {market_key} trade")
            return
        
        self.portfolio["balance"] -= actual_cost
        
        trade_record = {
            "market": market_key,
            "ticker": ticker,
            "side": side,
            "contracts": contracts,
            "entry_price": contract_price,
            "invested": actual_cost,
            "trade_time": time.time(),
            "model_prob": model_prob,
            "edge": edge
        }
        self.portfolio["pending_trades"].append(trade_record)
        self.portfolio["trades_count"] += 1
        self.save_portfolio()
        
        self.display_dashboard()
        self.write_log(
            f"🚨 NEW POSITION | {market_key} | {side} | Edge: {edge:.2%} | "
            f"Contracts: {contracts} @ ${contract_price:.2f} | "
            f"Deployed: ${actual_cost:.2f} | Balance: ${self.portfolio['balance']:.2f}"
        )

    def scan_market_for_opportunities(self, market_key):
        """Scan a specific market for trading opportunities"""
        ticker = self.active_tickers[market_key]
        if not ticker:
            return
        
        market_config = self.markets[market_key]
        
        # Fetch Kalshi price
        kalshi_price = self.fetch_kalshi_price(ticker)
        if kalshi_price is None:
            return
        
        # Fetch Coinbase data
        df = self.fetch_coinbase_data(
            market_config["asset"], 
            market_config["granularity"],
            periods=100
        )
        if df is None or len(df) < 50:
            return
        
        # Engineer features
        try:
            current_X = self.engineer_current_features(df)
            
            # Ensure all required features are present
            missing_features = set(self.selected_features) - set(current_X.columns)
            if missing_features:
                logging.debug(f"{market_key}: Missing features: {missing_features}")
                return
            
            # Predict
            model_prob = self.model.predict_proba(current_X[self.selected_features])[0][1]
            
            # Calculate edges
            edge_yes = model_prob - kalshi_price
            kalshi_no_price = 1.0 - kalshi_price
            edge_no = (1 - model_prob) - kalshi_no_price
            
            # Log signal
            logging.debug(
                f"{market_key} | Ticker: {ticker} | Model: {model_prob:.2%} | "
                f"Kalshi: {kalshi_price:.2%} | Edge: {edge_yes:.2%}"
            )
            
            # Entry logic with stricter filters
            if edge_yes >= self.min_edge_threshold and model_prob > 0.60:
                self.execute_trade(market_key, "UP", kalshi_price, edge_yes, model_prob)
            elif edge_no >= self.min_edge_threshold and model_prob < 0.40:
                self.execute_trade(market_key, "DOWN", kalshi_no_price, edge_no, model_prob)
                
        except Exception as e:
            logging.error(f"Error processing {market_key}: {e}")

    def resolve_pending_trades(self):
        """Resolve expired/settled trades"""
        still_pending = []
        
        for trade in self.portfolio["pending_trades"]:
            # Auto-sweeper for ghost trades
            if "trade_time" not in trade:
                logging.info(f"🧹 Purging ghost trade: {trade.get('ticker', 'UNKNOWN')}")
                self.portfolio["balance"] += trade["invested"]
                self.write_log(f"Auto-Sweeper: Refunded ${trade['invested']:.2f}")
                continue
            
            # 60-minute kill switch
            if time.time() - trade["trade_time"] > 3600:
                logging.info(f"🧹 Trade timeout: {trade['ticker']}")
                self.portfolio["balance"] += trade["invested"]
                self.write_log(f"Timeout: Refunded ${trade['invested']:.2f} from {trade['ticker']}")
                continue
            
            # Check settlement status
            ticker = trade["ticker"]
            path = f"/trade-api/v2/markets/{ticker}"
            url = f"https://api.elections.kalshi.com{path}"
            
            timestamp_str = str(int(time.time() * 1000))
            signature = self.sign_kalshi_request("GET", path, timestamp_str)
            headers = {
                "KALSHI-ACCESS-KEY": self.api_key,
                "KALSHI-ACCESS-SIGNATURE": signature,
                "KALSHI-ACCESS-TIMESTAMP": timestamp_str,
                "Content-Type": "application/json"
            }
            
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    market_data = response.json().get('market', {})
                    status = market_data.get('status')
                    
                    if status == "settled":
                        result = market_data.get('result')
                        won = (trade["side"] == "UP" and result == "yes") or \
                              (trade["side"] == "DOWN" and result == "no")
                        
                        if won:
                            payout = trade["contracts"] * 1.00
                            profit = payout - trade["invested"]
                            self.portfolio["balance"] += payout
                            self.write_log(
                                f"✅ SETTLED WIN | {trade.get('market', 'N/A')} | {trade['ticker']} | "
                                f"P&L: +${profit:.2f} | Balance: ${self.portfolio['balance']:.2f}"
                            )
                            self.performance_stats["wins"] += 1
                        else:
                            self.write_log(
                                f"❌ SETTLED LOSS | {trade.get('market', 'N/A')} | {trade['ticker']} | "
                                f"Loss: -${trade['invested']:.2f} | Balance: ${self.portfolio['balance']:.2f}"
                            )
                            self.performance_stats["losses"] += 1
                        
                        self.performance_stats["total_trades"] += 1
                        continue
                    
                elif response.status_code == 404:
                    logging.info(f"🧹 404 Error for {ticker}, refunding")
                    self.portfolio["balance"] += trade["invested"]
                    continue
                    
            except Exception as e:
                logging.debug(f"Settlement check failed for {ticker}: {e}")
            
            still_pending.append(trade)
        
        self.portfolio["pending_trades"] = still_pending
        self.save_portfolio()

    def run_live_loop(self):
        """Main live trading loop"""
        logging.info("🚀 Starting Multi-Market Shadow Bot")
        logging.info(f"📊 Monitoring {len(self.markets)} markets: {', '.join(self.markets.keys())}")
        
        # Initial cleanup
        self.resolve_pending_trades()
        self.display_dashboard()
        
        last_ticker_update = 0
        last_settlement_check = 0
        last_dashboard_update = 0
        iteration = 0
        
        while True:
            try:
                current_time = time.time()
                iteration += 1
                
                # Update tickers every 60 seconds
                if current_time - last_ticker_update > 60:
                    self.update_active_tickers()
                    last_ticker_update = current_time
                
                # Manage existing positions
                for market_key in self.markets:
                    if self.active_tickers[market_key]:
                        self.manage_open_positions(market_key)
                
                # Scan for new opportunities (only if we can take positions)
                if self.can_take_position():
                    for market_key in self.markets:
                        if self.active_tickers[market_key]:
                            self.scan_market_for_opportunities(market_key)
                
                # Settlement check every 60 seconds
                if current_time - last_settlement_check > 60:
                    self.resolve_pending_trades()
                    last_settlement_check = current_time
                
                # Dashboard update every 30 seconds
                if current_time - last_dashboard_update > 30:
                    self.display_dashboard()
                    last_dashboard_update = current_time
                
                # Heartbeat every 100 iterations (~8 minutes)
                if iteration % 100 == 0:
                    stats = self.performance_stats
                    win_rate = stats["wins"] / max(stats["total_trades"], 1)
                    logging.info(
                        f"💓 Heartbeat | Trades: {stats['total_trades']} | "
                        f"Win Rate: {win_rate:.1%} | Total P&L: ${stats['total_pnl']:.2f}"
                    )
                
                time.sleep(5)  # Main loop delay
                
            except KeyboardInterrupt:
                logging.info("\n🛑 Shutting down bot...")
                self.display_dashboard()
                break
            except Exception as e:
                logging.error(f"Main loop error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    bot = MultiMarketShadowBot()
    bot.run_live_loop()
