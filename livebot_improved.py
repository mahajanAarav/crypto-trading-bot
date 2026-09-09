import os
import time
import json
import base64
import logging
import requests
import warnings
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import serialization
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('shadow_bot.log'),
        logging.StreamHandler()
    ]
)

class MultiMarketShadowBot:
    """
    Advanced shadow trading bot supporting multiple markets simultaneously:
    - BTC/ETH/XRP/SOL
    - 5-minute and 15-minute intervals
    - Up to 8 concurrent markets
    """
    
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.portfolio_file = os.path.join(self.script_dir, "portfolio.json")
        self.log_file = os.path.join(self.script_dir, "trade_log.txt")
        self.key_file = os.path.join(self.script_dir, "ergoKey.txt")
        
        # Market configurations
        self.markets = {
            "BTC_15M": {"asset": "BTC", "product": "BTC-USD", "granularity": 900, "series": "KXBTC15M"},
            "BTC_5M": {"asset": "BTC", "product": "BTC-USD", "granularity": 300, "series": "KXBTC5M"},
            "ETH_15M": {"asset": "ETH", "product": "ETH-USD", "granularity": 900, "series": "KXETH15M"},
            "ETH_5M": {"asset": "ETH", "product": "ETH-USD", "granularity": 300, "series": "KXETH5M"},
            # Add more as needed
        }
        
        # Load models for each market
        self.models = {}
        self.feature_lists = {}
        self.load_models()
        
        # Trading state
        self.load_portfolio()
        self.active_tickers = {}  # {market_name: ticker}
        self.market_data_cache = {}  # {market_name: DataFrame}
        
        # API credentials
        self.api_key = os.environ.get("KALSHI_API_KEY")
        try:
            with open(self.key_file, "r") as key_file:
                self.private_key_str = key_file.read()
        except FileNotFoundError:
            logging.error(f"Could not find {self.key_file}")
            exit()

        if not self.api_key:
            logging.error("KALSHI_API_KEY environment variable not set")
            exit()
        
        # Risk management parameters
        self.max_total_exposure = 0.50  # Max 50% of capital across all positions
        self.max_per_trade = 0.15  # Max 15% per individual trade
        self.max_positions = 8  # Max concurrent positions
        self.max_per_market = 1  # Max 1 position per market type
        
        # Performance tracking
        self.performance_metrics = {market: {"trades": 0, "wins": 0, "total_pnl": 0.0} 
                                   for market in self.markets.keys()}

    def load_models(self):
        """Load trained models for each market"""
        logging.info("Loading trained models...")
        for market_name, config in self.markets.items():
            model_file = os.path.join(self.script_dir, 
                f"bot_{config['asset']}_{config['granularity']}s.pkl")
            
            if os.path.exists(model_file):
                try:
                    self.models[market_name] = joblib.load(model_file)
                    
                    # Load feature importance to get feature list
                    importance_file = model_file.replace('.pkl', '_feature_importance.csv')
                    if os.path.exists(importance_file):
                        features_df = pd.read_csv(importance_file)
                        self.feature_lists[market_name] = features_df['feature'].tolist()
                    else:
                        logging.warning(f"No feature list found for {market_name}")
                    
                    logging.info(f"✓ Loaded model for {market_name}")
                except Exception as e:
                    logging.error(f"Failed to load model for {market_name}: {e}")
            else:
                logging.warning(f"Model file not found for {market_name}: {model_file}")
        
        if not self.models:
            logging.error("No models loaded! Train models first using training_improved.py")
            exit()

    def load_portfolio(self):
        """Load or initialize portfolio"""
        if os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, "r") as f:
                self.portfolio = json.load(f)
        else:
            self.portfolio = {
                "balance": 100.00,
                "pending_trades": [],
                "total_deposited": 100.00,
                "total_withdrawn": 0.00,
                "inception_date": datetime.now().isoformat()
            }
            self.save_portfolio()
            self.write_log("=== NEW MULTI-MARKET SHADOW BOT INITIALIZED WITH $100.00 ===")

    def save_portfolio(self):
        """Save portfolio to disk"""
        with open(self.portfolio_file, "w") as f:
            json.dump(self.portfolio, f, indent=4)

    def write_log(self, message):
        """Write to log file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        with open(self.log_file, "a") as f:
            f.write(log_entry)
        logging.info(message)

    def display_hud(self):
        """Display current portfolio status"""
        total_exposure = sum(t["invested"] for t in self.portfolio["pending_trades"])
        exposure_pct = total_exposure / self.portfolio["balance"] * 100 if self.portfolio["balance"] > 0 else 0
        
        total_value = self.portfolio["balance"] + total_exposure
        roi = (total_value - self.portfolio["total_deposited"]) / self.portfolio["total_deposited"] * 100
        
        print("\n" + "="*70)
        print(f"💰 BALANCE: ${self.portfolio['balance']:.2f} | EXPOSURE: ${total_exposure:.2f} ({exposure_pct:.1f}%)")
        print(f"📊 TOTAL VALUE: ${total_value:.2f} | ROI: {roi:+.2f}%")
        print(f"🔥 OPEN POSITIONS: {len(self.portfolio['pending_trades'])}/{self.max_positions}")
        
        if self.portfolio["pending_trades"]:
            print("\nACTIVE TRADES:")
            for i, trade in enumerate(self.portfolio["pending_trades"], 1):
                market = trade.get('market', 'UNKNOWN')
                print(f"  [{i}] {market} | {trade['side']} | {trade['contracts']} @ ${trade['entry_price']:.2f} | ${trade['invested']:.2f}")
        
        print("\nMARKET PERFORMANCE:")
        for market, metrics in self.performance_metrics.items():
            if metrics['trades'] > 0:
                winrate = metrics['wins'] / metrics['trades'] * 100
                print(f"  {market}: {metrics['trades']} trades | {winrate:.1f}% win | ${metrics['total_pnl']:.2f} P&L")
        
        print("="*70 + "\n")

    def sign_kalshi_request(self, method, path, timestamp_str):
        """Sign Kalshi API request"""
        key = serialization.load_pem_private_key(self.private_key_str.encode(), password=None)
        message = f"{timestamp_str}{method}{path}"
        signature = key.sign(message.encode(), padding.PKCS1v15(), hashes.SHA256())
        return base64.b64encode(signature).decode()

    def update_active_tickers(self):
        """Update active tickers for all configured markets"""
        updated = {}
        
        for market_name, config in self.markets.items():
            path = f"/trade-api/v2/markets?series_ticker={config['series']}&status=open"
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
                        updated[market_name] = markets[0]['ticker']
                    else:
                        updated[market_name] = None
                else:
                    updated[market_name] = None
            except Exception as e:
                logging.debug(f"Error updating ticker for {market_name}: {e}")
                updated[market_name] = None
        
        self.active_tickers = updated
        active_count = sum(1 for t in updated.values() if t is not None)
        logging.info(f"Active markets: {active_count}/{len(self.markets)}")
        return active_count > 0

    def fetch_kalshi_price(self, ticker):
        """Fetch current Kalshi market price"""
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
            return None
        except:
            return None

    def fetch_coinbase_data(self, product, granularity):
        """Fetch recent Coinbase candles"""
        url = f"https://api.exchange.coinbase.com/products/{product}/candles?granularity={granularity}"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["time"], unit="s")
                return df.sort_values("timestamp").reset_index(drop=True)
            return None
        except:
            return None

    def engineer_features(self, df):
        """Engineer features matching training script"""
        data = df.copy()
        
        # Price returns
        data["return_1"] = data["close"].pct_change()
        data["return_3"] = data["close"].pct_change(3)
        data["return_5"] = data["close"].pct_change(5)
        data["return_10"] = data["close"].pct_change(10)
        data["return_15"] = data["close"].pct_change(15)
        data["return_30"] = data["close"].pct_change(30)
        
        # Volatility
        data["volatility_5"] = data["return_1"].rolling(5).std()
        data["volatility_10"] = data["return_1"].rolling(10).std()
        data["volatility_20"] = data["return_1"].rolling(20).std()
        data["volatility_50"] = data["return_1"].rolling(50).std()
        
        # Moving averages
        data["sma_5"] = data["close"].rolling(5).mean()
        data["sma_10"] = data["close"].rolling(10).mean()
        data["sma_20"] = data["close"].rolling(20).mean()
        data["sma_50"] = data["close"].rolling(50).mean()
        data["ema_10"] = data["close"].ewm(span=10).mean()
        data["ema_20"] = data["close"].ewm(span=20).mean()
        data["ema_50"] = data["close"].ewm(span=50).mean()
        
        # Distance from MAs
        data["sma_dist_5"] = (data["close"] / data["sma_5"]) - 1
        data["sma_dist_10"] = (data["close"] / data["sma_10"]) - 1
        data["sma_dist_20"] = (data["close"] / data["sma_20"]) - 1
        data["sma_dist_50"] = (data["close"] / data["sma_50"]) - 1
        
        # MA crossovers
        data["ma_cross_10_20"] = (data["sma_10"] > data["sma_20"]).astype(int)
        data["ma_cross_20_50"] = (data["sma_20"] > data["sma_50"]).astype(int)
        data["ema_cross_10_20"] = (data["ema_10"] > data["ema_20"]).astype(int)
        
        # Volume
        data["volume_sma_20"] = data["volume"].rolling(20).mean()
        data["volume_ratio"] = data["volume"] / data["volume_sma_20"]
        data["volume_trend"] = data["volume"].rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=True)
        
        # Momentum
        data["momentum_5"] = data["close"] - data["close"].shift(5)
        data["momentum_10"] = data["close"] - data["close"].shift(10)
        data["momentum_20"] = data["close"] - data["close"].shift(20)
        data["momentum_accel"] = data["momentum_10"].diff()
        
        # RSI variants
        for period in [7, 14, 21]:
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            data[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        for period in [10, 20]:
            rolling_mean = data["close"].rolling(period).mean()
            rolling_std = data["close"].rolling(period).std()
            data[f"bb_position_{period}"] = (data["close"] - rolling_mean) / (2 * rolling_std)
            data[f"bb_width_{period}"] = (rolling_std / rolling_mean) * 100
        
        # ATR
        data["tr"] = np.maximum(
            data["high"] - data["low"],
            np.maximum(
                abs(data["high"] - data["close"].shift(1)),
                abs(data["low"] - data["close"].shift(1))
            )
        )
        data["atr_14"] = data["tr"].rolling(14).mean()
        data["atr_ratio"] = data["atr_14"] / data["close"]
        
        # MACD
        ema_12 = data["close"].ewm(span=12).mean()
        ema_26 = data["close"].ewm(span=26).mean()
        data["macd"] = ema_12 - ema_26
        data["macd_signal"] = data["macd"].ewm(span=9).mean()
        data["macd_hist"] = data["macd"] - data["macd_signal"]
        
        # Stochastic
        low_14 = data["low"].rolling(14).min()
        high_14 = data["high"].rolling(14).max()
        data["stoch_k"] = 100 * (data["close"] - low_14) / (high_14 - low_14)
        data["stoch_d"] = data["stoch_k"].rolling(3).mean()
        
        # Price patterns
        data["higher_high"] = ((data["high"] > data["high"].shift(1)) & 
                               (data["high"].shift(1) > data["high"].shift(2))).astype(int)
        data["lower_low"] = ((data["low"] < data["low"].shift(1)) & 
                             (data["low"].shift(1) < data["low"].shift(2))).astype(int)
        
        # Candle patterns
        data["body"] = abs(data["close"] - data["open"])
        data["range"] = data["high"] - data["low"]
        data["body_to_range"] = data["body"] / data["range"].replace(0, 1)
        data["upper_shadow"] = data["high"] - np.maximum(data["open"], data["close"])
        data["lower_shadow"] = np.minimum(data["open"], data["close"]) - data["low"]
        
        # Regimes
        data["trend_regime"] = (data["ema_20"] > data["sma_50"]).astype(int)
        vol_mean = data["volatility_20"].rolling(100).mean()
        data["high_vol_regime"] = (data["volatility_20"] > vol_mean * 1.2).astype(int)
        data["low_vol_regime"] = (data["volatility_20"] < vol_mean * 0.8).astype(int)
        
        # Acceleration
        data["price_accel"] = data["return_1"].diff()
        data["vol_accel"] = data["volatility_20"].diff()
        
        # Relative strength
        data["rs_vs_sma10"] = data["close"] / data["sma_10"]
        data["rs_vs_sma20"] = data["close"] / data["sma_20"]
        data["rs_vs_sma50"] = data["close"] / data["sma_50"]
        
        # Multi-timeframe momentum
        data["mtf_momentum"] = (
            data["return_5"].rolling(3).mean() * 0.5 +
            data["return_10"].rolling(3).mean() * 0.3 +
            data["return_20"].rolling(3).mean() * 0.2
        )
        
        return data

    def get_prediction(self, market_name, df):
        """Get model prediction for a market"""
        if market_name not in self.models:
            return None
        
        try:
            # Engineer features
            data = self.engineer_features(df)
            
            # Get feature list for this model
            if market_name in self.feature_lists:
                features = self.feature_lists[market_name]
            else:
                # Fallback to model's feature names
                features = self.models[market_name].feature_names_in_
            
            # Extract last row with required features
            X = data.iloc[-1:][features]
            
            # Predict
            prob = self.models[market_name].predict_proba(X)[0][1]
            return prob
            
        except Exception as e:
            logging.error(f"Prediction error for {market_name}: {e}")
            return None

    def calculate_position_risk(self):
        """Calculate current risk exposure"""
        total_exposure = sum(t["invested"] for t in self.portfolio["pending_trades"])
        exposure_pct = total_exposure / (self.portfolio["balance"] + total_exposure)
        
        return {
            "total_exposure": total_exposure,
            "exposure_pct": exposure_pct,
            "available_capital": self.portfolio["balance"],
            "num_positions": len(self.portfolio["pending_trades"]),
            "can_trade": (exposure_pct < self.max_total_exposure and 
                         len(self.portfolio["pending_trades"]) < self.max_positions)
        }

    def execute_trade(self, market_name, ticker, side, contract_price, edge, model_prob):
        """Execute a shadow trade with risk management"""
        # Check if already in this market
        for trade in self.portfolio["pending_trades"]:
            if trade.get("market") == market_name:
                return  # Already have position in this market
        
        # Risk check
        risk = self.calculate_position_risk()
        if not risk["can_trade"]:
            logging.debug(f"Risk limit reached: {risk['exposure_pct']:.1%} exposure")
            return
        
        # Position sizing based on edge and volatility
        base_risk = 0.10  # 10% base risk
        edge_multiplier = min(abs(edge) / 0.15, 2.0)  # Scale with edge
        risk_pct = min(base_risk * edge_multiplier, self.max_per_trade)
        
        # Calculate position size
        available = risk["available_capital"]
        investment = available * risk_pct
        contracts = int(investment // contract_price)
        
        if contracts < 1:
            return
        
        actual_cost = contracts * contract_price
        self.portfolio["balance"] -= actual_cost
        
        trade_record = {
            "market": market_name,
            "ticker": ticker,
            "side": side,
            "contracts": contracts,
            "entry_price": contract_price,
            "invested": actual_cost,
            "model_prob": model_prob,
            "edge": edge,
            "trade_time": time.time(),
            "entry_timestamp": datetime.now().isoformat()
        }
        
        self.portfolio["pending_trades"].append(trade_record)
        self.save_portfolio()
        self.display_hud()
        
        self.write_log(
            f"🚨 TRADE EXECUTED: {market_name} {side} | "
            f"Edge: {abs(edge):.2%} | Prob: {model_prob:.2%} | "
            f"{contracts} @ ${contract_price:.2f} | ${actual_cost:.2f} invested"
        )

    def manage_positions(self):
        """Manage all open positions across markets"""
        for trade in self.portfolio["pending_trades"][:]:
            market_name = trade.get("market")
            ticker = trade["ticker"]
            
            # Get current price
            current_price = self.fetch_kalshi_price(ticker)
            if current_price is None:
                continue
            
            # Calculate P&L
            if trade["side"] == "UP":
                exit_price = current_price
            else:  # DOWN
                exit_price = 1.0 - current_price
            
            entry_price = trade["entry_price"]
            if entry_price == 0:
                continue
            
            net_pnl_pct = (exit_price - entry_price) / entry_price
            
            # Exit conditions
            take_profit = 0.25
            stop_loss = -0.12
            time_held = time.time() - trade.get("trade_time", time.time())
            max_hold_time = 3600  # 1 hour max hold
            
            should_exit = (
                net_pnl_pct >= take_profit or
                net_pnl_pct <= stop_loss or
                time_held > max_hold_time
            )
            
            if should_exit:
                exit_value = trade["contracts"] * exit_price
                profit = exit_value - trade["invested"]
                self.portfolio["balance"] += exit_value
                
                # Update performance metrics
                if market_name in self.performance_metrics:
                    self.performance_metrics[market_name]["trades"] += 1
                    if profit > 0:
                        self.performance_metrics[market_name]["wins"] += 1
                    self.performance_metrics[market_name]["total_pnl"] += profit
                
                reason = (
                    "🎯 PROFIT" if net_pnl_pct >= take_profit else
                    "🛑 STOP" if net_pnl_pct <= stop_loss else
                    "⏰ TIME"
                )
                
                self.write_log(
                    f"{reason} | {market_name} {trade['side']} | "
                    f"Exit: ${exit_price:.2f} | P&L: {net_pnl_pct:+.2%} (${profit:+.2f}) | "
                    f"Balance: ${self.portfolio['balance']:.2f}"
                )
                
                self.portfolio["pending_trades"].remove(trade)
                self.save_portfolio()
                self.display_hud()

    def resolve_settled_markets(self):
        """Check for and resolve settled markets"""
        for trade in self.portfolio["pending_trades"][:]:
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
                            
                            market_name = trade.get("market", "UNKNOWN")
                            if market_name in self.performance_metrics:
                                self.performance_metrics[market_name]["trades"] += 1
                                self.performance_metrics[market_name]["wins"] += 1
                                self.performance_metrics[market_name]["total_pnl"] += profit
                            
                            self.write_log(
                                f"✅ SETTLED WIN | {market_name} {trade['side']} | "
                                f"Payout: ${payout:.2f} | Profit: +${profit:.2f}"
                            )
                        else:
                            market_name = trade.get("market", "UNKNOWN")
                            if market_name in self.performance_metrics:
                                self.performance_metrics[market_name]["trades"] += 1
                                self.performance_metrics[market_name]["total_pnl"] -= trade["invested"]
                            
                            self.write_log(
                                f"❌ SETTLED LOSS | {market_name} {trade['side']} | "
                                f"Loss: -${trade['invested']:.2f}"
                            )
                        
                        self.portfolio["pending_trades"].remove(trade)
                        self.save_portfolio()
                        self.display_hud()
                        
                    elif status == "canceled":
                        self.portfolio["balance"] += trade["invested"]
                        self.portfolio["pending_trades"].remove(trade)
                        self.save_portfolio()
                        self.write_log(f"⚠️ MARKET CANCELED | Refunded ${trade['invested']:.2f}")
                        
                elif response.status_code == 404:
                    # Market doesn't exist - refund
                    self.portfolio["balance"] += trade["invested"]
                    self.portfolio["pending_trades"].remove(trade)
                    self.save_portfolio()
                    self.write_log(f"🧹 404 ERROR | Refunded ${trade['invested']:.2f}")
                    
            except Exception as e:
                logging.debug(f"Error checking settlement for {ticker}: {e}")

    def run_live_loop(self):
        """Main trading loop for multi-market bot"""
        logging.info("🚀 Starting Multi-Market Shadow Trading Bot")
        logging.info(f"Configured markets: {list(self.markets.keys())}")
        logging.info(f"Loaded models: {list(self.models.keys())}")
        
        # Initial cleanup
        self.resolve_settled_markets()
        self.display_hud()
        
        iteration = 0
        last_display = 0
        last_settlement_check = 0
        
        while True:
            iteration += 1
            current_time = time.time()
            
            try:
                # Update active tickers for all markets
                if iteration % 12 == 0:  # Every ~1 minute
                    self.update_active_tickers()
                
                # Manage existing positions
                self.manage_positions()
                
                # Check for settlement
                if current_time - last_settlement_check > 60:
                    self.resolve_settled_markets()
                    last_settlement_check = current_time
                
                # Scan markets for opportunities
                for market_name, ticker in self.active_tickers.items():
                    if ticker is None:
                        continue
                    
                    if market_name not in self.models:
                        continue
                    
                    # Fetch market data
                    config = self.markets[market_name]
                    df = self.fetch_coinbase_data(config["product"], config["granularity"])
                    
                    if df is None or len(df) < 60:
                        continue
                    
                    # Get prediction
                    model_prob = self.get_prediction(market_name, df)
                    if model_prob is None:
                        continue
                    
                    # Get Kalshi price
                    kalshi_yes_price = self.fetch_kalshi_price(ticker)
                    if kalshi_yes_price is None:
                        continue
                    
                    # Calculate edge
                    edge_yes = model_prob - kalshi_yes_price
                    edge_no = (1 - model_prob) - (1 - kalshi_yes_price)
                    
                    # Entry logic
                    min_edge = 0.12
                    
                    if model_prob > 0.62 and edge_yes >= min_edge:
                        self.execute_trade(
                            market_name, ticker, "UP", 
                            kalshi_yes_price, edge_yes, model_prob
                        )
                    elif model_prob < 0.38 and edge_no >= min_edge:
                        self.execute_trade(
                            market_name, ticker, "DOWN",
                            1.0 - kalshi_yes_price, edge_no, model_prob
                        )
                
                # Periodic status display
                if current_time - last_display > 120:  # Every 2 minutes
                    self.display_hud()
                    last_display = current_time
                
                time.sleep(5)
                
            except KeyboardInterrupt:
                logging.info("Shutting down gracefully...")
                self.display_hud()
                break
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(10)

if __name__ == "__main__":
    bot = MultiMarketShadowBot()
    bot.run_live_loop()
