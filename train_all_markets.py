"""
Multi-Market Model Training Orchestrator

Trains optimized models for all configured markets:
- BTC, ETH, XRP, SOL
- 5-minute and 15-minute timeframes
- Saves models and feature importance for live trading
"""

import sys
import logging
from datetime import datetime
import pandas as pd

# Import the improved training class
sys.path.insert(0, '/home/claude')
from training_improved import AdvancedShadowTradingBot

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class MultiMarketTrainer:
    """Orchestrates training across multiple markets"""
    
    def __init__(self):
        # Define all markets to train
        self.markets = [
            {"asset": "BTC", "granularity": 900, "name": "BTC_15M"},
            {"asset": "BTC", "granularity": 300, "name": "BTC_5M"},
            {"asset": "ETH", "granularity": 900, "name": "ETH_15M"},
            {"asset": "ETH", "granularity": 300, "name": "ETH_5M"},
            {"asset": "XRP", "granularity": 900, "name": "XRP_15M"},
            {"asset": "XRP", "granularity": 300, "name": "XRP_5M"},
            {"asset": "SOL", "granularity": 900, "name": "SOL_15M"},
            {"asset": "SOL", "granularity": 300, "name": "SOL_5M"},
        ]
        
        self.results = []
    
    def train_market(self, asset, granularity, name):
        """Train model for a single market"""
        print("\n" + "="*80)
        print(f"TRAINING: {name} ({asset} @ {granularity}s)")
        print("="*80 + "\n")
        
        try:
            # Initialize bot
            bot = AdvancedShadowTradingBot(asset=asset, granularity=granularity)
            
            # Fetch data (more for 5m due to higher frequency)
            periods = 12000 if granularity == 300 else 10000
            logging.info(f"Fetching {periods} periods of data...")
            raw_data = bot.fetch_market_data(periods=periods)
            
            if len(raw_data) < 1000:
                logging.error(f"Insufficient data for {name}: {len(raw_data)} candles")
                return None
            
            # Engineer features
            logging.info("Engineering features...")
            processed_data = bot.engineer_advanced_features(raw_data)
            
            # Feature selection
            logging.info("Selecting important features...")
            n_features = 35 if granularity == 300 else 40  # Slightly fewer for 5m
            selected_features = bot.select_important_features(processed_data, n_features=n_features)
            
            # Walk-forward training
            logging.info("Starting walk-forward validation...")
            predictions, metrics = bot.walk_forward_training_advanced(processed_data, selected_features)
            
            # Simulate trading
            logging.info("Simulating trading strategy...")
            equity_curve = bot.simulate_trading_advanced(predictions)
            
            # Evaluate performance
            results = bot.evaluate_performance_advanced()
            
            if results is not None:
                # Calculate final metrics
                final_metrics = {
                    "market": name,
                    "asset": asset,
                    "granularity": granularity,
                    "total_trades": len(results),
                    "win_rate": results["win"].mean(),
                    "total_pnl": results["profit_loss"].sum(),
                    "avg_pnl_per_trade": results["profit_loss"].mean(),
                    "sharpe": results["profit_loss"].mean() / results["profit_loss"].std() if results["profit_loss"].std() > 0 else 0,
                    "model_auc": metrics["auc"].mean(),
                    "model_accuracy": metrics["accuracy"].mean(),
                    "training_date": datetime.now().isoformat()
                }
                
                # Save model
                model_filename = f"bot_{asset}_{granularity}s.pkl"
                bot.save_model(model_filename)
                
                print(f"\n✅ Successfully trained {name}")
                print(f"   Model saved: {model_filename}")
                print(f"   Win Rate: {final_metrics['win_rate']:.2%}")
                print(f"   Total P&L: {final_metrics['total_pnl']:.2%}")
                print(f"   Model AUC: {final_metrics['model_auc']:.4f}")
                
                return final_metrics
            else:
                logging.error(f"No trading results for {name}")
                return None
                
        except Exception as e:
            logging.error(f"Training failed for {name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train_all(self):
        """Train models for all markets"""
        print("\n" + "="*80)
        print("MULTI-MARKET MODEL TRAINING ORCHESTRATOR")
        print(f"Training {len(self.markets)} markets")
        print("="*80 + "\n")
        
        for market_config in self.markets:
            result = self.train_market(
                market_config["asset"],
                market_config["granularity"],
                market_config["name"]
            )
            
            if result:
                self.results.append(result)
        
        # Summary report
        self.print_summary()
    
    def print_summary(self):
        """Print summary of all training runs"""
        if not self.results:
            print("\n❌ No models were successfully trained")
            return
        
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80 + "\n")
        
        df = pd.DataFrame(self.results)
        
        print(f"Successfully trained: {len(df)}/{len(self.markets)} markets\n")
        
        print("Performance by Market:")
        print("-" * 80)
        for _, row in df.iterrows():
            print(f"{row['market']:12} | "
                  f"Win Rate: {row['win_rate']:>6.2%} | "
                  f"P&L: {row['total_pnl']:>+7.2%} | "
                  f"Sharpe: {row['sharpe']:>5.2f} | "
                  f"AUC: {row['model_auc']:>5.3f} | "
                  f"Trades: {row['total_trades']:>4.0f}")
        
        print("-" * 80)
        print(f"\nAverage Win Rate: {df['win_rate'].mean():.2%}")
        print(f"Average P&L: {df['total_pnl'].mean():+.2%}")
        print(f"Average Sharpe: {df['sharpe'].mean():.2f}")
        print(f"Average AUC: {df['model_auc'].mean():.3f}")
        
        # Save summary
        summary_file = f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(summary_file, index=False)
        print(f"\n📊 Summary saved to: {summary_file}")
        
        # Best performing markets
        print("\n🏆 Top 3 Markets by Win Rate:")
        top_winrate = df.nlargest(3, 'win_rate')[['market', 'win_rate', 'total_pnl', 'sharpe']]
        for idx, row in top_winrate.iterrows():
            print(f"   {row['market']}: {row['win_rate']:.2%} win rate, {row['total_pnl']:+.2%} P&L, {row['sharpe']:.2f} Sharpe")
        
        print("\n💰 Top 3 Markets by Total P&L:")
        top_pnl = df.nlargest(3, 'total_pnl')[['market', 'total_pnl', 'win_rate', 'sharpe']]
        for idx, row in top_pnl.iterrows():
            print(f"   {row['market']}: {row['total_pnl']:+.2%} P&L, {row['win_rate']:.2%} win rate, {row['sharpe']:.2f} Sharpe")
        
        print("\n" + "="*80)

if __name__ == "__main__":
    trainer = MultiMarketTrainer()
    trainer.train_all()
    
    print("\n✅ Training complete! Models are ready for live trading.")
    print("Run livebot_improved.py to start shadow trading.")
