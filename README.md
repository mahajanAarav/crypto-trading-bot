# Asynchronous Multi-Market Trading Bot

An event-driven algorithmic trading system for short-horizon crypto prediction markets. It ingests live price data over WebSockets, computes technical features on rolling windows, generates directional signals from a gradient-boosted ensemble, and executes trades across multiple concurrent markets — all under an async architecture built for low latency.

Runs in **shadow mode**: signals and fills are simulated against live market data, with a simulated portfolio. No real capital is at risk.

---

## What it does

The system trades binary outcome contracts on 15-minute crypto markets (BTC, ETH, XRP, SOL). Every cycle it:

1. Receives live price ticks from Coinbase over persistent WebSocket connections
2. Updates rolling OHLCV windows and recomputes ~70 technical features incrementally
3. Runs the feature vector through a trained XGBoost/LightGBM ensemble to get a directional probability
4. Compares model probability against live market pricing to compute an edge
5. Sizes a position against portfolio risk limits and executes if the edge clears threshold

---

## Architecture

```
Coinbase WebSocket ──▶ RollingFeatureEngine ──▶ Ensemble Model
                            (deque + NumPy)          │
                                                     ▼
                                              TradeSignal
                                                     │
                       Risk Manager ◀────────────────┘
                            │
                            ▼
                   AsyncKalshiClient ──▶ RSA-signed REST orders
```

Everything runs on a single `asyncio` event loop. Design decisions worth calling out:

**Event-driven ingestion over polling.** Persistent WebSocket connections push price updates rather than the bot pulling on an interval, which removes polling latency and reduces API load.

**Rolling feature computation.** Features are maintained over `collections.deque` windows with NumPy operations rather than recomputed from a full DataFrame each tick. This is what keeps feature calculation inside the latency budget as market count grows.

**Non-blocking logging.** `AsyncLogger` pushes log records onto an `asyncio.Queue` drained by a background worker, so disk I/O never blocks the trading path.

**Dynamic market discovery.** Rather than hardcoding tickers, the bot queries available 15-minute series at runtime and registers whatever is active — 15m contracts expire constantly, so static tickers go stale fast.

**RSA request signing.** Kalshi authenticates via PKCS1v15-signed request headers; the client signs each request with a private key loaded from disk.

---

## Modeling

**Feature engineering** (`training_v2_enhanced.py`) builds 70+ features from raw OHLCV:

- Multi-timeframe returns and momentum (1, 3, 5, 10, 15, 30 periods)
- Multi-period RSI (7, 14, 21)
- Bollinger Bands across several windows
- MACD, Stochastic Oscillator
- Volatility measures over 5/10/20/50 period windows
- Candle pattern features
- Regime detection — trend vs. range, high vs. low volatility

**Feature selection** ranks all features by XGBoost gain importance and keeps the top 30. Cutting the input space this way was worth more to validation performance than any single added indicator.

**Ensemble** combines XGBoost and LightGBM via soft voting. The two disagree in different market regimes, so averaging their probabilities is more stable than either alone.

**Hyperparameter tuning** uses Optuna with a TPE sampler over `TimeSeriesSplit` folds.

**Walk-forward validation** is the important part. Random train/test splits leak future information into training on time-series data and produce results that look excellent and generalize terribly. Instead, the model trains on an expanding window and validates on the next unseen block, stepping forward through history — every evaluation uses only data that preceded it. Reported metrics come from this procedure, not from an in-sample fit.

---

## Risk management

Position sizing and exposure limits are enforced at portfolio level, not per trade:

| Control | Purpose |
|---|---|
| Max total exposure | Caps combined capital at risk across all open positions |
| Max per-trade risk | Bounds worst-case loss on any single position |
| Max concurrent positions | Limits correlated exposure across markets |
| One position per market | Prevents stacking the same directional bet |
| Edge-based sizing | Scales position with model conviction |
| Volatility adjustment | Shrinks size in high-volatility regimes |

Exits trigger on profit target, stop loss, time expiry, or edge decay — whichever comes first.

---

## Repository layout

| File | Purpose |
|---|---|
| `async_bot_v3.py` | Main async engine — WebSocket ingestion, rolling features, execution |
| `training_v2_enhanced.py` | Feature engineering, selection, Optuna tuning, walk-forward validation |
| `training_improved.py` | Earlier training pipeline |
| `livebot_v2_enhanced.py` | Synchronous multi-market bot (predecessor to v3) |
| `livebot_improved.py` | Earlier live bot |
| `hybrid_bot_fixed.py` | Hybrid execution variant |
| `market_discovery.py` | Runtime discovery of active market tickers |
| `train_all_markets.py` | Batch training across market/timeframe pairs |
| `benchmark.py` | Latency and throughput benchmarking |
| `performance_monitor.py` | Live portfolio and per-market performance tracking |
| `diagnostics.py` / `debug_bot.py` | Diagnostic harnesses for API and signal debugging |
| `check_api_env.py` | Credential and connectivity preflight check |
| `docs/` | Architecture notes, migration guides, version comparisons |

---

## Setup

```bash
git clone https://github.com/mahajanAarav/crypto-trading-bot.git
cd crypto-trading-bot
pip install -r requirements.txt
```

Credentials are read from the environment and from a key file that is **not** committed:

```bash
export KALSHI_API_KEY="your-api-key"
# place your RSA private key at ergoKey.txt (gitignored)
```

Verify connectivity, then train and run:

```bash
python check_api_env.py
python training_v2_enhanced.py
python async_bot_v3.py
```

---

## Known limitations

- **Shadow mode only.** Fills are simulated at observed prices; real execution would face slippage, partial fills, and queue position that this doesn't model.
- **No automated tests.** Correctness was verified through diagnostic scripts and manual iteration rather than a test suite.
- **Order book modeling is approximate.** `OrderBookDepthModel` estimates depth rather than consuming a true L2 feed.
- **Model retraining is manual.** There's no scheduled retraining as regimes shift.
- **Single-process.** All markets share one event loop; scaling past ~15 markets would want multiple processes.

---

## What I'd build next

Automated retraining on a schedule with drift detection, a proper backtesting harness with transaction cost modeling, a test suite around the feature pipeline, and a true L2 order book feed for realistic fill simulation.
