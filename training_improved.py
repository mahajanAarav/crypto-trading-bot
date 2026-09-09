import pandas as pd
import numpy as np
import logging
import requests
import time
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class AdvancedShadowTradingBot:
    def __init__(self, asset="BTC", granularity=900):
        """
        asset: BTC, ETH, XRP, SOL
        granularity: 300 (5m) or 900 (15m)
        """
        self.asset = asset
        self.granularity = granularity
        self.model = None
        self.ensemble_model = None
        self.trade_log = []
        self.max_hold_bars = 12  # Increased for better exits
        self.trading_fee = 0.01 
        self.spread = 0.02
        self.feature_importance = None

    def fetch_market_data(self, periods=10000):
        """Fetch more historical data for better training"""
        logging.info(f"Fetching {self.asset} data (granularity={self.granularity}s)...")
        
        # Map assets to Coinbase product IDs
        product_map = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD", 
            "XRP": "XRP-USD",
            "SOL": "SOL-USD"
        }
        product = product_map.get(self.asset, "BTC-USD")
        
        all_candles = []
        end_time = datetime.utcnow()
        candles_per_request = 300
        requests_needed = (periods // candles_per_request) + 1
        
        for i in range(requests_needed):
            start_time = end_time - timedelta(seconds=self.granularity * candles_per_request)
            url = f"https://api.exchange.coinbase.com/products/{product}/candles"
            params = {
                "granularity": self.granularity,
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if not data:
                        break
                    all_candles.extend(data)
                    end_time = start_time
                    time.sleep(0.3)  # Rate limiting
                    if i % 10 == 0:
                        logging.info(f"Fetched {len(all_candles)} candles...")
                else:
                    logging.error(f"Coinbase API error: {response.status_code}")
                    break
            except Exception as e:
                logging.error(f"Request failed: {e}")
                break
                
        df = pd.DataFrame(all_candles, columns=["time", "low", "high", "open", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["time"], unit="s")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df.tail(periods).reset_index(drop=True)
        
        logging.info(f"Loaded {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df

    def engineer_advanced_features(self, df):
        """Enhanced feature engineering with more predictive signals"""
        data = df.copy()
        
        # Price-based features
        data["return_1"] = data["close"].pct_change()
        data["return_3"] = data["close"].pct_change(3)
        data["return_5"] = data["close"].pct_change(5)
        data["return_10"] = data["close"].pct_change(10)
        data["return_15"] = data["close"].pct_change(15)
        data["return_30"] = data["close"].pct_change(30)
        
        # Volatility features
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
        
        # Distance from moving averages
        data["sma_dist_5"] = (data["close"] / data["sma_5"]) - 1
        data["sma_dist_10"] = (data["close"] / data["sma_10"]) - 1
        data["sma_dist_20"] = (data["close"] / data["sma_20"]) - 1
        data["sma_dist_50"] = (data["close"] / data["sma_50"]) - 1
        
        # MA crossovers
        data["ma_cross_10_20"] = (data["sma_10"] > data["sma_20"]).astype(int)
        data["ma_cross_20_50"] = (data["sma_20"] > data["sma_50"]).astype(int)
        data["ema_cross_10_20"] = (data["ema_10"] > data["ema_20"]).astype(int)
        
        # Volume features
        data["volume_sma_20"] = data["volume"].rolling(20).mean()
        data["volume_ratio"] = data["volume"] / data["volume_sma_20"]
        data["volume_trend"] = data["volume"].rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=True)
        
        # Momentum indicators
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
        
        # Stochastic Oscillator
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
        
        # Regime detection
        data["trend_regime"] = (data["ema_20"] > data["sma_50"]).astype(int)
        vol_mean = data["volatility_20"].rolling(100).mean()
        data["high_vol_regime"] = (data["volatility_20"] > vol_mean * 1.2).astype(int)
        data["low_vol_regime"] = (data["volatility_20"] < vol_mean * 0.8).astype(int)
        
        # Price acceleration
        data["price_accel"] = data["return_1"].diff()
        data["vol_accel"] = data["volatility_20"].diff()
        
        # Relative strength to moving averages
        data["rs_vs_sma10"] = data["close"] / data["sma_10"]
        data["rs_vs_sma20"] = data["close"] / data["sma_20"]
        data["rs_vs_sma50"] = data["close"] / data["sma_50"]
        
        # Multi-timeframe momentum
        data["mtf_momentum"] = (
            data["return_5"].rolling(3).mean() * 0.5 +
            data["return_10"].rolling(3).mean() * 0.3 +
            data["return_20"].rolling(3).mean() * 0.2
        )
        
        # Target variable - multiple horizons for robustness
        # Using 6-8 periods ahead (1.5-2 hours for 15min, 30-40min for 5min)
        target_periods = 6 if self.granularity == 900 else 8
        data["target"] = (data["close"].shift(-target_periods) > data["close"]).astype(int)
        
        # Also create confidence target (strong moves)
        data["strong_target"] = (
            (data["close"].shift(-target_periods) / data["close"] > 1.01).astype(int)
        )
        
        data.dropna(inplace=True)
        logging.info(f"Engineered {len(data.columns)} features from {len(data)} samples")
        return data

    def select_important_features(self, data, n_features=40):
        """Select most important features using XGBoost feature importance"""
        logging.info("Running feature selection...")
        
        # Get all numeric features except target
        all_features = [col for col in data.columns if col not in 
                       ["timestamp", "time", "target", "strong_target"]]
        
        X = data[all_features]
        y = data["target"]
        
        # Quick XGBoost to get feature importance
        temp_model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        temp_model.fit(X, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            'feature': all_features,
            'importance': temp_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance
        selected_features = importance.head(n_features)['feature'].tolist()
        
        logging.info(f"Selected top {n_features} features:")
        for i, row in importance.head(20).iterrows():
            logging.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return selected_features

    def create_ensemble_model(self, best_params=None):
        """Create ensemble of XGBoost and LightGBM for better predictions"""
        if best_params is None:
            xgb_params = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.02,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': -1
            }
        else:
            xgb_params = {**best_params, 'random_state': 42, 'n_jobs': -1}
        
        lgbm_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.02,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        xgb_model = XGBClassifier(**xgb_params)
        lgbm_model = LGBMClassifier(**lgbm_params)
        
        # Voting ensemble with soft voting
        ensemble = VotingClassifier(
            estimators=[('xgb', xgb_model), ('lgbm', lgbm_model)],
            voting='soft',
            weights=[0.6, 0.4]  # Slightly favor XGBoost
        )
        
        return ensemble

    def walk_forward_training_advanced(self, data, features):
        """Enhanced walk-forward with adaptive windows and ensemble"""
        logging.info("Starting advanced walk-forward validation...")
        
        initial_window = 3000  # Larger initial training window
        test_window = 500
        step = 200  # Smaller step for more validation folds
        
        predictions = []
        fold_metrics = []
        
        for fold, start in enumerate(range(0, len(data) - initial_window - test_window, step)):
            # Adaptive window - grows over time
            train_size = min(initial_window + fold * 200, 5000)
            train_start = max(0, start + test_window - train_size)
            
            train = data.iloc[train_start:start + test_window]
            test = data.iloc[start + test_window:start + test_window + test_window]
            
            if len(test) < 100:  # Skip if test set too small
                break
            
            X_train, y_train = train[features], train["target"]
            X_test, y_test = test[features], test["target"]
            
            # Train ensemble
            self.ensemble_model = self.create_ensemble_model()
            self.ensemble_model.fit(X_train, y_train)
            
            # Predictions
            probs = self.ensemble_model.predict_proba(X_test)[:, 1]
            preds = (probs > 0.5).astype(int)
            
            # Calculate metrics
            auc = roc_auc_score(y_test, probs)
            acc = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds, zero_division=0)
            recall = recall_score(y_test, preds, zero_division=0)
            
            fold_metrics.append({
                'fold': fold,
                'auc': auc,
                'accuracy': acc,
                'precision': precision,
                'recall': recall,
                'train_size': len(train),
                'test_size': len(test)
            })
            
            # Store predictions
            test = test.copy()
            test["model_prob"] = probs
            predictions.append(test)
            
            if fold % 5 == 0:
                logging.info(f"Fold {fold}: AUC={auc:.4f}, Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        
        # Aggregate results
        test_data = pd.concat(predictions)
        
        # Print fold statistics
        metrics_df = pd.DataFrame(fold_metrics)
        logging.info("\n=== Walk-Forward Validation Results ===")
        logging.info(f"Mean AUC: {metrics_df['auc'].mean():.4f} ± {metrics_df['auc'].std():.4f}")
        logging.info(f"Mean Accuracy: {metrics_df['accuracy'].mean():.4f} ± {metrics_df['accuracy'].std():.4f}")
        logging.info(f"Mean Precision: {metrics_df['precision'].mean():.4f} ± {metrics_df['precision'].std():.4f}")
        logging.info(f"Mean Recall: {metrics_df['recall'].mean():.4f} ± {metrics_df['recall'].std():.4f}")
        logging.info("=" * 40)
        
        return test_data, metrics_df

    def simulate_trading_advanced(self, data):
        """Enhanced trading simulation with better risk management"""
        position = None
        entry_price = 0
        entry_prob = 0
        bars_held = 0
        equity_curve = []
        capital = 1.0

        for idx, row in data.iterrows():
            model_prob = row["model_prob"]
            
            # Simulated Kalshi prices with realistic bid-ask
            base_price = 0.5 + (model_prob - 0.5) * 0.6  # Less extreme pricing
            spread = self.spread
            
            yes_ask = min(base_price + spread/2, 0.99)
            yes_bid = max(base_price - spread/2, 0.01)
            no_ask = min((1 - base_price) + spread/2, 0.99)
            no_bid = max((1 - base_price) - spread/2, 0.01)

            prob_no = 1 - model_prob
            edge_yes = model_prob - yes_ask
            edge_no = prob_no - no_ask

            # Manage existing position
            if position is not None:
                bars_held += 1
                current_exit_price = yes_bid if position == "YES" else no_bid
                
                # Calculate P&L
                gross_pnl = (current_exit_price - entry_price) / entry_price
                net_pnl = gross_pnl - self.trading_fee
                
                # Dynamic exit conditions
                # 1. Take profit target
                take_profit = 0.25 if entry_prob > 0.70 else 0.20
                # 2. Stop loss
                stop_loss = -0.12 if entry_prob > 0.70 else -0.15
                # 3. Edge disappeared
                current_edge = (model_prob - yes_bid) if position == "YES" else (prob_no - no_bid)
                # 4. Max hold time
                
                if (net_pnl >= take_profit or 
                    net_pnl <= stop_loss or 
                    current_edge < 0.03 or 
                    bars_held >= self.max_hold_bars):
                    
                    self.trade_log[-1]["exit_price"] = current_exit_price
                    self.trade_log[-1]["profit_loss"] = net_pnl
                    self.trade_log[-1]["bars_held"] = bars_held
                    self.trade_log[-1]["exit_time"] = row["timestamp"]
                    self.trade_log[-1]["exit_reason"] = (
                        "take_profit" if net_pnl >= take_profit else
                        "stop_loss" if net_pnl <= stop_loss else
                        "edge_gone" if current_edge < 0.03 else
                        "max_hold"
                    )
                    
                    capital *= (1 + net_pnl)
                    position = None
                    bars_held = 0
                
                equity_curve.append(capital)
                continue

            # Skip high volatility regimes
            if row.get("high_vol_regime", 0) == 1:
                equity_curve.append(capital)
                continue

            # Entry logic with stricter filters
            min_edge = 0.12  # Increased minimum edge
            min_prob = 0.62  # Higher confidence threshold
            max_prob = 0.38  # For short side
            
            # Additional filters
            vol_ok = row.get("volatility_20", 0) < 0.03  # Not too volatile
            
            # Long entry
            if model_prob > min_prob and edge_yes >= min_edge and vol_ok:
                position = "YES"
                entry_price = yes_ask
                entry_prob = model_prob
                bars_held = 0
                self._record_entry_advanced(
                    row["timestamp"], model_prob, yes_ask, 
                    edge_yes, entry_price, position, row
                )
            # Short entry
            elif model_prob < max_prob and edge_no >= min_edge and vol_ok:
                position = "NO"
                entry_price = no_ask
                entry_prob = model_prob
                bars_held = 0
                self._record_entry_advanced(
                    row["timestamp"], prob_no, no_ask, 
                    edge_no, entry_price, position, row
                )
            
            equity_curve.append(capital)

        return equity_curve

    def _record_entry_advanced(self, timestamp, model_p, kalshi_p, edge, entry_price, pos_type, row):
        """Record trade with additional context"""
        trade = {
            "entry_time": timestamp,
            "side": pos_type,
            "model_probability": model_p,
            "kalshi_probability": kalshi_p,
            "edge": edge,
            "entry_price": entry_price,
            "volatility": row.get("volatility_20", 0),
            "rsi": row.get("rsi_14", 50),
            "regime": "trend" if row.get("trend_regime", 0) == 1 else "range",
            "exit_price": None,
            "profit_loss": None,
            "bars_held": None,
            "exit_time": None,
            "exit_reason": None
        }
        self.trade_log.append(trade)

    def evaluate_performance_advanced(self):
        """Enhanced performance evaluation with more metrics"""
        completed = [t for t in self.trade_log if t["profit_loss"] is not None]
        if not completed:
            logging.info("No closed trades")
            return None

        df = pd.DataFrame(completed)
        df["win"] = df["profit_loss"] > 0
        
        # Basic metrics
        win_rate = df["win"].mean()
        total_trades = len(df)
        avg_edge = df["edge"].mean()
        total_profit = df["profit_loss"].sum()
        
        # Win/Loss analysis
        wins = df[df["win"]]
        losses = df[~df["win"]]
        avg_win = wins["profit_loss"].mean() if len(wins) > 0 else 0
        avg_loss = losses["profit_loss"].mean() if len(losses) > 0 else 0
        
        # Risk metrics
        profit_factor = abs(wins["profit_loss"].sum() / losses["profit_loss"].sum()) if len(losses) > 0 else float('inf')
        sharpe_ratio = df["profit_loss"].mean() / df["profit_loss"].std() if df["profit_loss"].std() > 0 else 0
        max_drawdown = (df["profit_loss"].cumsum().cummax() - df["profit_loss"].cumsum()).max()
        
        # Exit reason analysis
        exit_reasons = df["exit_reason"].value_counts()
        
        print("\n" + "="*60)
        print("ADVANCED TRADING SIMULATION SUMMARY")
        print("="*60)
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Average Edge: {avg_edge:.2%}")
        print(f"Total Profit: {total_profit:.2%}")
        print(f"Average Win: {avg_win:.2%}")
        print(f"Average Loss: {avg_loss:.2%}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Average Bars Held: {df['bars_held'].mean():.1f}")
        print("\nExit Reasons:")
        for reason, count in exit_reasons.items():
            print(f"  {reason}: {count} ({count/total_trades:.1%})")
        print("="*60 + "\n")
        
        return df

    def save_model(self, filename):
        """Save the ensemble model"""
        import joblib
        logging.info(f"Saving ensemble model to {filename}")
        joblib.dump(self.ensemble_model, filename)
        
        # Also save feature importance
        if self.feature_importance is not None:
            self.feature_importance.to_csv(filename.replace('.pkl', '_feature_importance.csv'), index=False)

if __name__ == "__main__":
    # Example: Train on BTC 15-minute data
    bot = AdvancedShadowTradingBot(asset="BTC", granularity=900)
    
    # Fetch data
    raw_data = bot.fetch_market_data(periods=10000)
    
    # Engineer features
    processed_data = bot.engineer_advanced_features(raw_data)
    
    # Feature selection
    selected_features = bot.select_important_features(processed_data, n_features=40)
    
    # Walk-forward training
    predictions, metrics = bot.walk_forward_training_advanced(processed_data, selected_features)
    
    # Simulate trading
    equity_curve = bot.simulate_trading_advanced(predictions)
    
    # Evaluate
    results = bot.evaluate_performance_advanced()
    
    if results is not None:
        print("\nRecent Trades:")
        print(results[["entry_time", "side", "edge", "entry_price", "exit_price", 
                      "profit_loss", "bars_held", "exit_reason"]].tail(10))
        
        # Save model
        model_filename = f"bot_{bot.asset}_{bot.granularity}s.pkl"
        bot.save_model(model_filename)
        
        print(f"\n✅ Model saved to {model_filename}")
        print(f"✅ Feature importance saved to {model_filename.replace('.pkl', '_feature_importance.csv')}")
