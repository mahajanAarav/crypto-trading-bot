"""
Kalshi Market Discovery Diagnostic
===================================

This script checks which KXBTC15M, KXETH15M, KXSOL15M, KXXRP15M markets
are actually active and tradeable right now.
"""

import asyncio
import aiohttp
import os
import base64
import time
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("MarketDiscovery")

async def discover_markets():
    logger.info("\n" + "="*70)
    logger.info("KALSHI MARKET DISCOVERY")
    logger.info("="*70)
    
    # Get credentials
    api_key = os.environ.get("KALSHI_API_KEY")
    if not api_key:
        logger.error("❌ KALSHI_API_KEY not set")
        return
    
    with open("ergoKey.txt", "r") as f:
        private_key_str = f.read()
    private_key = serialization.load_pem_private_key(private_key_str.encode(), password=None)
    
    series_list = ["KXBTC15M", "KXETH15M", "KXSOL15M", "KXXRP15M"]
    
    async with aiohttp.ClientSession() as session:
        for series in series_list:
            logger.info(f"\n{'='*70}")
            logger.info(f"📊 Searching for: {series}")
            logger.info(f"{'='*70}")
            
            # Try different status filters
            for status in ["open", "active", "all"]:
                path = "/trade-api/v2/markets"
                params = {"series_ticker": series, "status": status}
                
                timestamp_str = str(int(time.time() * 1000))
                message = f"{timestamp_str}GET{path}"
                signature = private_key.sign(message.encode(), padding.PKCS1v15(), hashes.SHA256())
                
                headers = {
                    "KALSHI-ACCESS-KEY": api_key,
                    "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
                    "KALSHI-ACCESS-TIMESTAMP": timestamp_str,
                    "Content-Type": "application/json"
                }
                
                try:
                    async with session.get(
                        f"https://api.elections.kalshi.com{path}",
                        headers=headers,
                        params=params,
                        timeout=10
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            
                            if "markets" in result and result["markets"]:
                                logger.info(f"\n✅ Found {len(result['markets'])} market(s) with status='{status}':")
                                
                                for i, market in enumerate(result["markets"][:5], 1):  # Show first 5
                                    ticker = market.get("ticker", "N/A")
                                    yes_ask = market.get("yes_ask", 0) / 100
                                    yes_bid = market.get("yes_bid", 0) / 100
                                    volume = market.get("volume", 0)
                                    status_val = market.get("status", "N/A")
                                    
                                    logger.info(f"\n   {i}. {ticker}")
                                    logger.info(f"      Status: {status_val}")
                                    logger.info(f"      YES Ask: ${yes_ask:.4f} ({yes_ask*100:.2f}%)")
                                    logger.info(f"      YES Bid: ${yes_bid:.4f} ({yes_bid*100:.2f}%)")
                                    logger.info(f"      Volume: {volume}")
                                    
                                    if yes_ask > 0:
                                        logger.info(f"      ✅ TRADEABLE (has liquidity)")
                                    else:
                                        logger.info(f"      ❌ NO LIQUIDITY (yes_ask = $0)")
                                
                                if len(result["markets"]) > 5:
                                    logger.info(f"\n   ... and {len(result['markets']) - 5} more")
                                
                                break  # Found markets, stop trying other statuses
                            else:
                                logger.info(f"   ⚠️  No markets found with status='{status}'")
                        else:
                            logger.warning(f"   ❌ API error {response.status} for status='{status}'")
                            text = await response.text()
                            logger.warning(f"      Response: {text[:200]}")
                except Exception as e:
                    logger.error(f"   ❌ Request failed for status='{status}': {e}")
                
                await asyncio.sleep(0.5)  # Rate limit
    
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    logger.info("\nIf you see markets with YES Ask > $0, those are tradeable!")
    logger.info("If all markets show YES Ask = $0, then:")
    logger.info("  1. Markets may be settled/expired")
    logger.info("  2. No active contracts for this timeframe")
    logger.info("  3. Need to wait for new markets to open")
    logger.info("\nNext steps:")
    logger.info("  • If tradeable markets found: Update bot's active_ticker")
    logger.info("  • If no markets found: Check Kalshi website for availability")
    logger.info(f"{'='*70}\n")

if __name__ == "__main__":
    asyncio.run(discover_markets())
