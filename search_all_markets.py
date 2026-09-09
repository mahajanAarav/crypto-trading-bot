"""
Check for Kalshi Crypto Markets (All Timeframes)
================================================

Searches for KXBTC, KXETH, KXSOL, KXXRP markets across different timeframes
to find ones with actual liquidity.
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
logger = logging.getLogger("MarketSearch")

async def search_all_crypto_markets():
    logger.info("\n" + "="*70)
    logger.info("SEARCHING ALL KALSHI CRYPTO MARKETS")
    logger.info("="*70)
    
    # Get credentials
    api_key = os.environ.get("KALSHI_API_KEY")
    if not api_key:
        logger.error("❌ KALSHI_API_KEY not set")
        return
    
    with open("ergoKey.txt", "r") as f:
        private_key_str = f.read()
    private_key = serialization.load_pem_private_key(private_key_str.encode(), password=None)
    
    # Search patterns for different crypto assets and timeframes
    search_patterns = [
        "KXBTC",  # All Bitcoin markets
        "KXETH",  # All Ethereum markets
        "KXSOL",  # All Solana markets
        "KXXRP",  # All XRP markets
    ]
    
    tradeable_markets = []
    
    async with aiohttp.ClientSession() as session:
        for pattern in search_patterns:
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 Searching: {pattern}* (all timeframes)")
            logger.info(f"{'='*70}")
            
            path = "/trade-api/v2/markets"
            # Use limit to get more results
            params = {"limit": 100, "status": "open"}
            
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
                            # Filter for markets matching our pattern
                            matching_markets = [
                                m for m in result["markets"] 
                                if m.get("ticker", "").startswith(pattern)
                            ]
                            
                            if matching_markets:
                                logger.info(f"\n✅ Found {len(matching_markets)} {pattern} market(s)")
                                
                                # Show markets with liquidity
                                liquid_markets = [
                                    m for m in matching_markets 
                                    if m.get("yes_ask", 0) > 0
                                ]
                                
                                if liquid_markets:
                                    logger.info(f"   💰 {len(liquid_markets)} have liquidity:\n")
                                    
                                    for market in liquid_markets[:10]:  # Show first 10
                                        ticker = market.get("ticker", "N/A")
                                        yes_ask = market.get("yes_ask", 0) / 100
                                        yes_bid = market.get("yes_bid", 0) / 100
                                        volume = market.get("volume", 0)
                                        
                                        logger.info(f"      📊 {ticker}")
                                        logger.info(f"         Ask: ${yes_ask:.4f} | Bid: ${yes_bid:.4f} | Vol: {volume}")
                                        
                                        tradeable_markets.append({
                                            "ticker": ticker,
                                            "yes_ask": yes_ask,
                                            "yes_bid": yes_bid,
                                            "volume": volume
                                        })
                                else:
                                    logger.info(f"   ⚠️  None have liquidity (all yes_ask = $0)")
                                    
                                    # Show a few examples anyway
                                    logger.info(f"\n   Examples (no liquidity):")
                                    for market in matching_markets[:3]:
                                        logger.info(f"      • {market.get('ticker', 'N/A')}")
                            else:
                                logger.info(f"   ⚠️  No {pattern} markets found")
                        else:
                            logger.info(f"   ⚠️  No markets returned from API")
                    else:
                        logger.warning(f"   ❌ API error {response.status}")
            except Exception as e:
                logger.error(f"   ❌ Request failed: {e}")
            
            await asyncio.sleep(0.5)  # Rate limit
    
    # Summary
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    
    if tradeable_markets:
        logger.info(f"\n✅ Found {len(tradeable_markets)} tradeable markets with liquidity!")
        logger.info("\nBest candidates for trading:")
        
        # Sort by volume
        tradeable_markets.sort(key=lambda x: x['volume'], reverse=True)
        
        for i, market in enumerate(tradeable_markets[:5], 1):
            logger.info(f"\n{i}. {market['ticker']}")
            logger.info(f"   Ask: ${market['yes_ask']:.4f} ({market['yes_ask']*100:.2f}%)")
            logger.info(f"   Bid: ${market['yes_bid']:.4f} ({market['yes_bid']*100:.2f}%)")
            logger.info(f"   Volume: {market['volume']}")
        
        logger.info("\n💡 Update your bot to use one of these tickers!")
    else:
        logger.info("\n❌ NO tradeable crypto markets found on Kalshi")
        logger.info("\nPossible reasons:")
        logger.info("  1. Kalshi discontinued these markets")
        logger.info("  2. No market makers providing liquidity")
        logger.info("  3. Markets only have liquidity at certain times")
        logger.info("\n💡 Recommendations:")
        logger.info("  • Check Kalshi website: https://kalshi.com/markets/crypto")
        logger.info("  • Contact Kalshi support about market availability")
        logger.info("  • Consider alternative prediction market platforms")
    
    logger.info(f"\n{'='*70}\n")

if __name__ == "__main__":
    asyncio.run(search_all_crypto_markets())
