import pandas as pd
import numpy as np
import logging
import requests
import time
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import optuna
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedShadowTradingBot:
    def __init__(self):
        # Enhanced model with better default parameters
        self.model = XGBClassifier(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.02,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.05,
            reg_lambda=1.0,
            scale_pos_weight=1.0,
            random_state=42,
            n_jobs=-1
        )
        self.trade_log = []
        self.max_hold_bars = 8
        self.trading_fee = 0.01 
        self.spread = 0.02
        self.feature_importance = None

    def fetch_market_data(self, periods=6000):
        """Fetch historical BTC data from Coinbase with error handling"""
        logging.info("Fetching real historical data from Coinbase...")
        all_candles = []
        end_time = datetime.utcnow()
        requests_needed = (periods // 300) + 1
        
        for i in range(requests_needed):
            start_time = end_time - timedelta(minutes=15 * 300)
            url = f"https://api.exchange.coinbase.com/products/BTC-USD/candles"
            params = {
                "granularity": 900,  # 15-minute candles
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
                    time.sleep(0.2)  # Rate limiting
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

        # Enhanced synthetic Kalshi prices with more realistic market dynamics
        np.random.seed(42)
        returns = df["close"].pct_change().fillna(0)
        
        # Add momentum and volatility factors
        momentum = returns.rolling(10).mean().fillna(0)
        volatility = returns.rolling(20).std().fillna(returns.std())
        
        kalshi_prices = 0.5 + (momentum * 100) + (returns * 30) + np.random.normal(0, 0.03, len(df))
        df["kalshi_price"] = np.clip(kalshi_prices, 0.01, 0.99)
        
        logging.info(f"Fetched {len(df)} candles")
        return df

    def engineer_features(self, df):
        """Enhanced feature engineering with 40+ technical indicators"""
        data = df.copy()
        
        # === PRICE-BASED FEATURES ===
        # Returns at multiple timeframes
        for period in [1, 2, 3, 5, 10, 15, 20]:
            data[f"return_{period}"] = data["close"].pct_change(period)
        
        # Moving averages
        data["sma_5"] = data["close"].rolling(5).mean()
        data["sma_10"] = data["close"].rolling(10).mean()
        data["sma_20"] = data["close"].rolling(20).mean()
        data["sma_50"] = data["close"].rolling(50).mean()
        data["ema_10"] = data["close"].ewm(span=10).mean()
        data["ema_20"] = data["close"].ewm(span=20).mean()
        data["ema_50"] = data["close"].ewm(span=50).mean()
        
        # MA crossovers and distances
        data["sma_dist_10"] = data["close"] / data["sma_10"] - 1
        data["sma_dist_20"] = data["close"] / data["sma_20"] - 1
        data["ema_dist_20"] = data["close"] / data["ema_20"] - 1
        data["ma_cross_5_10"] = (data["sma_5"] > data["sma_10"]).astype(int)
        data["ma_cross_10_20"] = (data["sma_10"] > data["sma_20"]).astype(int)
        
        # === VOLATILITY FEATURES ===
        for period in [10, 20, 30]:
            data[f"volatility_{period}"] = data["return_1"].rolling(period).std()
        
        # Bollinger Bands
        for period in [20, 50]:
            rolling_mean = data["close"].rolling(period).mean()
            rolling_std = data["close"].rolling(period).std()
            data[f"bb_position_{period}"] = (data["close"] - rolling_mean) / (2 * rolling_std)
            data[f"bb_width_{period}"] = (2 * rolling_std) / rolling_mean
        
        # ATR (Average True Range)
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        data["atr_14"] = true_range.rolling(14).mean()
        data["atr_20"] = true_range.rolling(20).mean()
        
        # === MOMENTUM FEATURES ===
        # RSI at multiple periods
        for period in [7, 14, 21]:
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss + 1e-10)
            data[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        
        # Momentum
        for period in [5, 10, 20]:
            data[f"momentum_{period}"] = data["close"] - data["close"].shift(period)
            
        # Rate of change
        for period in [5, 10, 20]:
            data[f"roc_{period}"] = (data["close"] - data["close"].shift(period)) / data["close"].shift(period)
        
        # === VOLUME FEATURES ===
        data["volume_ma_10"] = data["volume"].rolling(10).mean()
        data["volume_ma_20"] = data["volume"].rolling(20).mean()
        data["volume_ratio_10"] = data["volume"] / (data["volume_ma_10"] + 1e-10)
        data["volume_ratio_20"] = data["volume"] / (data["volume_ma_20"] + 1e-10)
        
        # Volume-price correlation
        data["volume_price_corr"] = data["volume"].rolling(20).corr(data["close"])
        
        # === TREND FEATURES ===
        data["trend_strength"] = abs(data["ema_20"] - data["sma_10"])
        data["momentum_accel"] = data["momentum_10"].diff()
        data["price_vol_ratio"] = data["return_1"] / (data["volatility_20"] + 1e-10)
        
        # === REGIME DETECTION ===
        vol_mean_100 = data["volatility_20"].rolling(100).mean()
        data["high_vol_regime"] = (data["volatility_20"] > vol_mean_100).astype(int)
        data["trend_regime"] = (data["ema_20"] > data["sma_10"]).astype(int)
        data["consolidation_regime"] = (data["bb_width_20"] < data["bb_width_20"].rolling(50).quantile(0.3)).astype(int)
        
        # === PRICE PATTERN FEATURES ===
        # Higher highs, lower lows
        data["higher_high"] = (data["high"] > data["high"].shift(1)).astype(int)
        data["lower_low"] = (data["low"] < data["low"].shift(1)).astype(int)
        
        # Gap detection
        data["gap_up"] = ((data["open"] - data["close"].shift(1)) / data["close"].shift(1) > 0.005).astype(int)
        data["gap_down"] = ((data["open"] - data["close"].shift(1)) / data["close"].shift(1) < -0.005).astype(int)
        
        # === KALSHI-SPECIFIC FEATURES ===
        data["kalshi_return"] = data["kalshi_price"].pct_change()
        data["kalshi_vol"] = data["kalshi_return"].rolling(20).std()
        data["kalshi_sma_10"] = data["kalshi_price"].rolling(10).mean()
        data["kalshi_dist"] = data["kalshi_price"] / data["kalshi_sma_10"] - 1
        
        # === LAGGED FEATURES ===
        for lag in [1, 2, 3]:
            data[f"return_1_lag{lag}"] = data["return_1"].shift(lag)
            data[f"rsi_14_lag{lag}"] = data["rsi_14"].shift(lag)
            data[f"volume_ratio_10_lag{lag}"] = data["volume_ratio_10"].shift(lag)
        
        # === TARGET VARIABLE ===
        # Predict if price will be higher in 4 bars (1 hour for 15-min candles)
        data["target"] = (data["close"].shift(-4) > data["close"]).astype(int)
        
        # Remove rows with NaN
        data.dropna(inplace=True)
        
        logging.info(f"Engineered {len(data.columns)} features")
        return data

    def select_best_features(self, data, n_features=30):
        """Feature selection using XGBoost importance"""
        features = [col for col in data.columns if col not in ["target", "timestamp", "time", "kalshi_price"]]
        
        X = data[features]
        y = data["target"]
        
        # Train a quick model to get feature importances
        temp_model = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        temp_model.fit(X, y)
        
        # Get feature importances
        importance_df = pd.DataFrame({
            'feature': features,
            'importance': temp_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        
        logging.info(f"\nTop 15 Most Important Features:")
        print(importance_df.head(15).to_string(index=False))
        
        # Select top N features
        selected_features = importance_df.head(n_features)['feature'].tolist()
        logging.info(f"\nSelected {len(selected_features)} features for training")
        
        return selected_features

    def optimize_hyperparameters_optuna(self, data, n_trials=50):
        """Hyperparameter optimization using Optuna (more efficient than GridSearch)"""
        logging.info(f"Starting Optuna optimization with {n_trials} trials...")
        
        # Select features first
        selected_features = self.select_best_features(data, n_features=30)
        
        X = data[selected_features]
        y = data["target"]
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 800, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95, step=0.05),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95, step=0.05),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'gamma': trial.suggest_float('gamma', 0, 0.3, step=0.05),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.1, step=0.01),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0, step=0.1),
                'random_state': 42,
                'n_jobs': -1
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = XGBClassifier(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred_proba)
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print("\n" + "="*50)
        print("OPTUNA OPTIMIZATION RESULTS")
        print("="*50)
        print(f"Best ROC AUC Score: {study.best_value:.4f}")
        print("\nBest Hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        print("="*50 + "\n")
        
        return study.best_params, selected_features

    def walk_forward_training(self, data, selected_features):
        """Walk-forward validation with detailed metrics"""
        window = 2000
        step = 500
        predictions = []
        
        logging.info("Starting walk-forward training...")

        for i, start in enumerate(range(0, len(data) - window - step, step)):
            train = data.iloc[start:start + window]
            test = data.iloc[start + window:start + window + step]

            X_train, y_train = train[selected_features], train["target"]
            X_test = test[selected_features]

            self.model.fit(X_train, y_train, verbose=False)
            probs = self.model.predict_proba(X_test)[:, 1]

            test = test.copy()
            test["model_prob"] = probs
            predictions.append(test)
            
            # Log progress
            if (i + 1) % 3 == 0:
                logging.info(f"Completed fold {i+1}")

        test_data = pd.concat(predictions)
        
        # Calculate comprehensive metrics
        y_true = test_data["target"]
        y_pred_proba = test_data["model_prob"]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        auc = roc_auc_score(y_true, y_pred_proba)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        print("\n" + "="*50)
        print("WALK-FORWARD VALIDATION RESULTS")
        print("="*50)
        print(f"ROC AUC Score:  {auc:.4f}")
        print(f"Accuracy:       {accuracy:.4f}")
        print(f"Precision:      {precision:.4f}")
        print(f"Recall:         {recall:.4f}")
        print("="*50 + "\n")
        
        return test_data

    def simulate_trading(self, data):
        """Enhanced trading simulation with better risk management"""
        position = None
        entry_price = 0
        bars_held = 0

        for _, row in data.iterrows():
            model_prob = row["model_prob"]
            kalshi_mid = row["kalshi_price"]
            
            # Market prices with spread
            yes_ask = kalshi_mid + (self.spread / 2)
            yes_bid = kalshi_mid - (self.spread / 2)
            no_ask = (1 - kalshi_mid) + (self.spread / 2)
            no_bid = (1 - kalshi_mid) - (self.spread / 2)

            prob_no = 1 - model_prob
            edge_yes = model_prob - yes_ask
            edge_no = prob_no - no_ask

            # === POSITION MANAGEMENT ===
            if position is not None:
                bars_held += 1
                current_exit_price = yes_bid if position == "YES" else no_bid
                current_edge = (model_prob - yes_bid) if position == "YES" else (prob_no - no_bid)

                gross_profit_pct = (current_exit_price - entry_price) / entry_price
                net_profit_pct = gross_profit_pct - self.trading_fee

                # Enhanced exit conditions
                take_profit = net_profit_pct >= 0.20
                stop_loss = net_profit_pct <= -0.15
                edge_gone = current_edge < 0.03
                max_time = bars_held >= self.max_hold_bars
                
                # Trailing stop for big winners
                trailing_stop = net_profit_pct >= 0.30 and current_edge < 0.05

                if take_profit or stop_loss or edge_gone or max_time or trailing_stop:
                    self.trade_log[-1]["exit_price"] = current_exit_price
                    self.trade_log[-1]["profit_loss"] = net_profit_pct
                    self.trade_log[-1]["exit_time"] = row["timestamp"]
                    self.trade_log[-1]["exit_reason"] = (
                        "take_profit" if take_profit else
                        "stop_loss" if stop_loss else
                        "edge_gone" if edge_gone else
                        "trailing_stop" if trailing_stop else
                        "max_time"
                    )
                    position = None
                    bars_held = 0
                continue

            # === ENTRY LOGIC ===
            # Skip high volatility regimes
            if row["high_vol_regime"] == 1:
                continue

            # Dynamic edge threshold based on volatility
            base_edge = 0.08
            vol_adjustment = row["volatility_20"] * 5
            edge_threshold = max(base_edge, min(0.15, base_edge + vol_adjustment))

            # Entry conditions with confidence filters
            strong_yes_signal = (
                model_prob > 0.62 and 
                edge_yes >= edge_threshold and
                row.get("trend_regime", 0) == 1  # Uptrend confirmation
            )
            
            strong_no_signal = (
                model_prob < 0.38 and 
                edge_no >= edge_threshold and
                row.get("trend_regime", 0) == 0  # Downtrend confirmation
            )

            if strong_yes_signal:
                position = "YES"
                entry_price = yes_ask
                bars_held = 0
                self._record_entry(row["timestamp"], model_prob, yes_ask, edge_yes, entry_price, position)
            elif strong_no_signal:
                position = "NO"
                entry_price = no_ask
                bars_held = 0
                self._record_entry(row["timestamp"], prob_no, no_ask, edge_no, entry_price, position)

    def _record_entry(self, timestamp, model_p, kalshi_p, edge, entry_price, pos_type):
        """Record trade entry"""
        trade = {
            "entry_time": timestamp, 
            "side": pos_type, 
            "model_probability": model_p,
            "kalshi_probability": kalshi_p, 
            "edge": edge, 
            "entry_price": entry_price,
            "exit_price": None, 
            "profit_loss": None, 
            "exit_time": None,
            "exit_reason": None
        }
        self.trade_log.append(trade)

    def evaluate_performance(self):
        """Comprehensive performance evaluation"""
        completed = [t for t in self.trade_log if t["profit_loss"] is not None]
        if not completed:
            logging.info("No closed trades")
            return None

        df = pd.DataFrame(completed)
        df["win"] = df["profit_loss"] > 0
        
        # Basic metrics
        win_rate = df["win"].mean()
        avg_edge = df["edge"].mean()
        total_profit = df["profit_loss"].sum()
        ev = df["profit_loss"].mean()
        
        # Winners vs losers
        winners = df[df["profit_loss"] > 0]
        losers = df[df["profit_loss"] < 0]
        avg_win = winners["profit_loss"].mean() if len(winners) > 0 else 0
        avg_loss = losers["profit_loss"].mean() if len(losers) > 0 else 0
        profit_factor = abs(winners["profit_loss"].sum() / losers["profit_loss"].sum()) if len(losers) > 0 else 0
        
        # Risk metrics
        returns = df["profit_loss"]
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
        
        # Exit reason analysis
        exit_reasons = df["exit_reason"].value_counts()

        print("\n" + "="*60)
        print("TRADING SIMULATION PERFORMANCE REPORT")
        print("="*60)
        print(f"Total Trades:              {len(df)}")
        print(f"Win Rate:                  {win_rate:.2%}")
        print(f"Average Edge:              {avg_edge:.2%}")
        print(f"Total Profit:              {total_profit:.2%}")
        print(f"Expected Value per Trade:  {ev:.2%}")
        print(f"\nAverage Win:               {avg_win:.2%}")
        print(f"Average Loss:              {avg_loss:.2%}")
        print(f"Profit Factor:             {profit_factor:.2f}")
        print(f"\nSharpe Ratio:              {sharpe_ratio:.2f}")
        print(f"Max Drawdown:              {max_drawdown:.2%}")
        print(f"\nExit Reason Breakdown:")
        for reason, count in exit_reasons.items():
            print(f"  {reason:20s} {count:3d} ({count/len(df):.1%})")
        print("="*60 + "\n")
        
        return df

if __name__ == "__main__":
    bot = EnhancedShadowTradingBot()
    
    # Fetch and engineer features
    raw_data = bot.fetch_market_data(periods=6000)
    processed_data = bot.engineer_features(raw_data)
    
    # === OPTIMIZATION BLOCK ===
    # Uncomment to run hyperparameter optimization (takes ~10-20 minutes)
    # best_params, selected_features = bot.optimize_hyperparameters_optuna(processed_data, n_trials=50)
    # bot.model = XGBClassifier(**best_params, random_state=42, n_jobs=-1)
    
    # OR use pre-selected features with default model
    selected_features = bot.select_best_features(processed_data, n_features=30)
    # ==========================

    # Walk-forward training
    predictions = bot.walk_forward_training(processed_data, selected_features)
    
    # Simulate trading
    bot.simulate_trading(predictions)
    results = bot.evaluate_performance()

    if results is not None:
        print("\nRecent Trades Sample:")
        print(results[["entry_time", "side", "edge", "entry_price", "exit_price", "profit_loss", "exit_reason"]].tail(10))
        
        # Save model
        model_path = "bot_v2_enhanced.json"
        print(f"\nSaving enhanced model to {model_path}...")
        bot.model.save_model(model_path)
        
        # Save feature list for live trading
        import json
        with open("selected_features.json", "w") as f:
            json.dump(selected_features, f)
        print("Saved selected features to selected_features.json")
        
        print("\n✅ Training complete! Use bot_v2_enhanced.json for live trading.")
