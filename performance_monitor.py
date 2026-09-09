"""
Shadow Trading Bot Performance Monitor & Analyzer

Real-time and historical performance analysis:
- Portfolio tracking and metrics
- Per-market performance breakdown
- Trade analysis and patterns
- Risk metrics and alerts
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

class PerformanceMonitor:
    """Analyzes shadow trading bot performance"""
    
    def __init__(self, portfolio_file="portfolio.json", trade_log_file="trade_log.txt"):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.portfolio_file = os.path.join(self.script_dir, portfolio_file)
        self.trade_log_file = os.path.join(self.script_dir, trade_log_file)
        
        self.portfolio = self.load_portfolio()
        self.trades = self.parse_trade_log()
    
    def load_portfolio(self):
        """Load current portfolio state"""
        if os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, 'r') as f:
                return json.load(f)
        return None
    
    def parse_trade_log(self):
        """Parse trade log file into structured data"""
        if not os.path.exists(self.trade_log_file):
            return pd.DataFrame()
        
        trades = []
        with open(self.trade_log_file, 'r') as f:
            for line in f:
                if 'EXECUTED' in line or 'PROFIT' in line or 'STOP' in line or 'TIME' in line:
                    # Extract trade information
                    try:
                        timestamp_str = line.split('[')[1].split(']')[0]
                        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        
                        if 'EXECUTED' in line:
                            # Entry trade
                            parts = line.split('|')
                            market = parts[0].split('EXECUTED:')[1].strip().split()[0] if 'EXECUTED:' in parts[0] else 'UNKNOWN'
                            side = parts[0].split('EXECUTED:')[1].strip().split()[1] if 'EXECUTED:' in parts[0] else 'UNKNOWN'
                            
                            # Extract edge and prob
                            edge = None
                            prob = None
                            for part in parts:
                                if 'Edge:' in part:
                                    edge = float(part.split('Edge:')[1].strip().replace('%', '')) / 100
                                if 'Prob:' in part:
                                    prob = float(part.split('Prob:')[1].strip().replace('%', '')) / 100
                            
                            trades.append({
                                'timestamp': timestamp,
                                'type': 'entry',
                                'market': market,
                                'side': side,
                                'edge': edge,
                                'prob': prob
                            })
                        
                        elif any(x in line for x in ['PROFIT', 'STOP', 'TIME']):
                            # Exit trade
                            parts = line.split('|')
                            market = parts[0].split('|')[0].strip().split()[-2] if len(parts[0].split()) > 2 else 'UNKNOWN'
                            
                            # Extract P&L
                            pnl = None
                            for part in parts:
                                if 'P&L:' in part:
                                    pnl_str = part.split('P&L:')[1].strip().split()[0]
                                    pnl = float(pnl_str.replace('%', '').replace('+', '')) / 100
                            
                            reason = 'profit' if 'PROFIT' in line else 'stop' if 'STOP' in line else 'time'
                            
                            trades.append({
                                'timestamp': timestamp,
                                'type': 'exit',
                                'market': market,
                                'pnl': pnl,
                                'reason': reason
                            })
                    except:
                        continue
        
        return pd.DataFrame(trades) if trades else pd.DataFrame()
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if self.portfolio is None:
            return None
        
        # Basic portfolio metrics
        current_balance = self.portfolio.get('balance', 0)
        pending_value = sum(t['invested'] for t in self.portfolio.get('pending_trades', []))
        total_value = current_balance + pending_value
        
        initial_capital = self.portfolio.get('total_deposited', 100)
        total_pnl = total_value - initial_capital
        roi = (total_pnl / initial_capital) * 100
        
        # Trade analysis
        exits = self.trades[self.trades['type'] == 'exit'].copy()
        if len(exits) > 0:
            win_count = len(exits[exits['pnl'] > 0])
            loss_count = len(exits[exits['pnl'] <= 0])
            win_rate = win_count / len(exits) * 100 if len(exits) > 0 else 0
            
            avg_win = exits[exits['pnl'] > 0]['pnl'].mean() * 100 if win_count > 0 else 0
            avg_loss = exits[exits['pnl'] <= 0]['pnl'].mean() * 100 if loss_count > 0 else 0
            
            # Risk metrics
            returns = exits['pnl'].dropna()
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            # Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            # Exit reason analysis
            exit_reasons = exits['reason'].value_counts()
        else:
            win_rate = avg_win = avg_loss = sharpe = max_drawdown = 0
            exit_reasons = pd.Series()
        
        return {
            'portfolio': {
                'balance': current_balance,
                'pending_value': pending_value,
                'total_value': total_value,
                'initial_capital': initial_capital,
                'total_pnl': total_pnl,
                'roi': roi
            },
            'trades': {
                'total': len(exits),
                'wins': win_count if len(exits) > 0 else 0,
                'losses': loss_count if len(exits) > 0 else 0,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            },
            'risk': {
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'profit_factor': abs(avg_win * win_count / (avg_loss * loss_count)) if loss_count > 0 and avg_loss != 0 else 0
            },
            'exits': exit_reasons.to_dict() if not exit_reasons.empty else {}
        }
    
    def market_breakdown(self):
        """Break down performance by market"""
        if len(self.trades) == 0:
            return pd.DataFrame()
        
        exits = self.trades[self.trades['type'] == 'exit'].copy()
        if len(exits) == 0:
            return pd.DataFrame()
        
        market_stats = []
        for market in exits['market'].unique():
            market_trades = exits[exits['market'] == market]
            
            wins = len(market_trades[market_trades['pnl'] > 0])
            total = len(market_trades)
            win_rate = wins / total * 100 if total > 0 else 0
            
            total_pnl = market_trades['pnl'].sum() * 100
            avg_pnl = market_trades['pnl'].mean() * 100
            
            market_stats.append({
                'market': market,
                'trades': total,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl
            })
        
        return pd.DataFrame(market_stats).sort_values('total_pnl', ascending=False)
    
    def print_dashboard(self):
        """Print comprehensive performance dashboard"""
        metrics = self.calculate_metrics()
        
        if metrics is None:
            print("❌ No portfolio data found")
            return
        
        print("\n" + "="*80)
        print("SHADOW TRADING BOT PERFORMANCE DASHBOARD")
        print("="*80 + "\n")
        
        # Portfolio section
        print("📊 PORTFOLIO OVERVIEW")
        print("-" * 80)
        p = metrics['portfolio']
        print(f"Balance:          ${p['balance']:.2f}")
        print(f"Pending Trades:   ${p['pending_value']:.2f}")
        print(f"Total Value:      ${p['total_value']:.2f}")
        print(f"Initial Capital:  ${p['initial_capital']:.2f}")
        print(f"Total P&L:        ${p['total_pnl']:+.2f} ({p['roi']:+.2f}%)")
        print()
        
        # Open positions
        if self.portfolio.get('pending_trades'):
            print("🔥 OPEN POSITIONS")
            print("-" * 80)
            for i, trade in enumerate(self.portfolio['pending_trades'], 1):
                market = trade.get('market', 'UNKNOWN')
                print(f"[{i}] {market:12} | {trade['side']:4} | "
                      f"{trade['contracts']:3} contracts @ ${trade['entry_price']:.2f} | "
                      f"${trade['invested']:.2f} | Edge: {trade.get('edge', 0)*100:+.1f}%")
            print()
        
        # Trade statistics
        print("📈 TRADE STATISTICS")
        print("-" * 80)
        t = metrics['trades']
        if t['total'] > 0:
            print(f"Total Trades:     {t['total']}")
            print(f"Wins / Losses:    {t['wins']} / {t['losses']}")
            print(f"Win Rate:         {t['win_rate']:.2f}%")
            print(f"Average Win:      {t['avg_win']:+.2f}%")
            print(f"Average Loss:     {t['avg_loss']:+.2f}%")
        else:
            print("No completed trades yet")
        print()
        
        # Risk metrics
        print("⚠️  RISK METRICS")
        print("-" * 80)
        r = metrics['risk']
        if t['total'] > 0:
            print(f"Sharpe Ratio:     {r['sharpe_ratio']:.2f}")
            print(f"Max Drawdown:     {r['max_drawdown']:.2f}%")
            print(f"Profit Factor:    {r['profit_factor']:.2f}")
        else:
            print("Insufficient data for risk metrics")
        print()
        
        # Exit reasons
        if metrics['exits']:
            print("🚪 EXIT ANALYSIS")
            print("-" * 80)
            for reason, count in metrics['exits'].items():
                pct = count / t['total'] * 100 if t['total'] > 0 else 0
                print(f"{reason.capitalize():12} {count:3} ({pct:5.1f}%)")
            print()
        
        # Market breakdown
        market_df = self.market_breakdown()
        if not market_df.empty:
            print("🎯 PERFORMANCE BY MARKET")
            print("-" * 80)
            for _, row in market_df.iterrows():
                print(f"{row['market']:12} | "
                      f"Trades: {row['trades']:3.0f} | "
                      f"Win Rate: {row['win_rate']:5.1f}% | "
                      f"Total P&L: {row['total_pnl']:+6.2f}% | "
                      f"Avg P&L: {row['avg_pnl']:+6.2f}%")
            print()
        
        # Time analysis
        if len(self.trades) > 0:
            exits = self.trades[self.trades['type'] == 'exit']
            if len(exits) > 0:
                print("📅 ACTIVITY TIMELINE")
                print("-" * 80)
                first_trade = exits['timestamp'].min()
                last_trade = exits['timestamp'].max()
                days_active = (last_trade - first_trade).days + 1
                trades_per_day = t['total'] / days_active if days_active > 0 else 0
                
                print(f"First Trade:      {first_trade.strftime('%Y-%m-%d %H:%M')}")
                print(f"Last Trade:       {last_trade.strftime('%Y-%m-%d %H:%M')}")
                print(f"Days Active:      {days_active}")
                print(f"Trades per Day:   {trades_per_day:.2f}")
                print()
        
        print("="*80 + "\n")
    
    def generate_report(self, output_file="performance_report.html"):
        """Generate HTML performance report"""
        metrics = self.calculate_metrics()
        if metrics is None:
            return
        
        html = f"""
        <html>
        <head>
            <title>Shadow Trading Bot Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #e0e0e0; }}
                .header {{ background: #2d2d2d; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .section {{ background: #2d2d2d; padding: 15px; border-radius: 8px; margin-bottom: 15px; }}
                .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
                .metric-label {{ font-size: 0.9em; color: #888; }}
                .metric-value {{ font-size: 1.4em; font-weight: bold; }}
                .positive {{ color: #4CAF50; }}
                .negative {{ color: #f44336; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #444; }}
                th {{ background: #333; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Shadow Trading Bot Performance Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Portfolio Overview</h2>
                <div class="metric">
                    <div class="metric-label">Total Value</div>
                    <div class="metric-value">${metrics['portfolio']['total_value']:.2f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">P&L</div>
                    <div class="metric-value {'positive' if metrics['portfolio']['total_pnl'] > 0 else 'negative'}">
                        ${metrics['portfolio']['total_pnl']:+.2f}
                    </div>
                </div>
                <div class="metric">
                    <div class="metric-label">ROI</div>
                    <div class="metric-value {'positive' if metrics['portfolio']['roi'] > 0 else 'negative'}">
                        {metrics['portfolio']['roi']:+.2f}%
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Trade Statistics</h2>
                <div class="metric">
                    <div class="metric-label">Total Trades</div>
                    <div class="metric-value">{metrics['trades']['total']}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{metrics['trades']['win_rate']:.1f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg Win</div>
                    <div class="metric-value positive">{metrics['trades']['avg_win']:+.2f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg Loss</div>
                    <div class="metric-value negative">{metrics['trades']['avg_loss']:+.2f}%</div>
                </div>
            </div>
        """
        
        # Add market breakdown table
        market_df = self.market_breakdown()
        if not market_df.empty:
            html += """
            <div class="section">
                <h2>Performance by Market</h2>
                <table>
                    <tr>
                        <th>Market</th>
                        <th>Trades</th>
                        <th>Win Rate</th>
                        <th>Total P&L</th>
                        <th>Avg P&L</th>
                    </tr>
            """
            for _, row in market_df.iterrows():
                pnl_class = 'positive' if row['total_pnl'] > 0 else 'negative'
                html += f"""
                    <tr>
                        <td>{row['market']}</td>
                        <td>{row['trades']:.0f}</td>
                        <td>{row['win_rate']:.1f}%</td>
                        <td class="{pnl_class}">{row['total_pnl']:+.2f}%</td>
                        <td class="{pnl_class}">{row['avg_pnl']:+.2f}%</td>
                    </tr>
                """
            html += """
                </table>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"📄 Performance report saved to: {output_file}")

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.print_dashboard()
    monitor.generate_report()
