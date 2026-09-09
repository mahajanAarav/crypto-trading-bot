"""
Kalshi API Environment Checker
==============================

This will show you EXACTLY which API endpoint you're using
and help verify if you're on production or demo.
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
logger = logging.getLogger("EnvCheck")

async def check_api_environment():
    logger.info("\n" + "="*70)
    logger.info("KALSHI API ENVIRONMENT DIAGNOSTIC")
    logger.info("="*70)
    
    # Check environment variables
    api_key = os.environ.get("KALSHI_API_KEY")
    
    logger.info("\n1️⃣ CHECKING ENVIRONMENT VARIABLES")
    logger.info("-" * 70)
    
    if api_key:
        logger.info(f"✅ KALSHI_API_KEY found: {api_key[:10]}...{api_key[-4:]}")
    else:
        logger.error("❌ KALSHI_API_KEY not set!")
        return
    
    # Check private key
    try:
        with open("ergoKey.txt", "r") as f:
            private_key_str = f.read()
        private_key = serialization.load_pem_private_key(private_key_str.encode(), password=None)
        logger.info("✅ Private key loaded from ergoKey.txt")
    except Exception as e:
        logger.error(f"❌ Failed to load private key: {e}")
        return
    
    # Test both endpoints
    endpoints = [
        ("PRODUCTION", "https://api.elections.kalshi.com"),
        ("DEMO/SANDBOX", "https://demo-api.elections.kalshi.com"),
    ]
    
    logger.info("\n2️⃣ TESTING API ENDPOINTS")
    logger.info("-" * 70)
    
    async with aiohttp.ClientSession() as session:
        for env_name, base_url in endpoints:
            logger.info(f"\n🔍 Testing {env_name}: {base_url}")
            
            # Test 1: Get exchange status
            path = "/trade-api/v2/exchange/status"
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
                    f"{base_url}{path}",
                    headers=headers,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"   ✅ Connected successfully!")
                        logger.info(f"   Exchange status: {result.get('exchange_active', 'unknown')}")
                        logger.info(f"   Trading active: {result.get('trading_active', 'unknown')}")
                    elif response.status == 401:
                        logger.info(f"   ❌ Authentication failed (wrong credentials for this environment)")
                    else:
                        logger.info(f"   ⚠️  HTTP {response.status}")
            except Exception as e:
                logger.info(f"   ❌ Connection failed: {e}")
            
            # Test 2: Try to get a specific market
            path = "/trade-api/v2/markets"
            params = {"series_ticker": "KXBTC15M", "status": "open"}
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
                    f"{base_url}{path}",
                    headers=headers,
                    params=params,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "markets" in result and result["markets"]:
                            market = result["markets"][0]
                            yes_ask = market.get("yes_ask", 0) / 100
                            volume = market.get("volume", 0)
                            
                            logger.info(f"   📊 KXBTC15M market found:")
                            logger.info(f"      Ticker: {market.get('ticker', 'N/A')}")
                            logger.info(f"      YES Ask: ${yes_ask:.4f}")
                            logger.info(f"      Volume: {volume}")
                            
                            if yes_ask > 0 and volume > 0:
                                logger.info(f"      🎯 THIS IS THE LIVE ENVIRONMENT!")
                            elif yes_ask == 0 and volume == 0:
                                logger.info(f"      ⚠️  No liquidity (demo or inactive market)")
                        else:
                            logger.info(f"   ⚠️  No KXBTC15M markets found")
            except Exception as e:
                logger.info(f"   ❌ Market check failed: {e}")
            
            await asyncio.sleep(1)
    
    # Check account balance
    logger.info("\n3️⃣ CHECKING ACCOUNT BALANCE")
    logger.info("-" * 70)
    
    for env_name, base_url in endpoints:
        logger.info(f"\n🔍 Checking balance on {env_name}")
        
        path = "/trade-api/v2/portfolio/balance"
        timestamp_str = str(int(time.time() * 1000))
        message = f"{timestamp_str}GET{path}"
        signature = private_key.sign(message.encode(), padding.PKCS1v15(), hashes.SHA256())
        
        headers = {
            "KALSHI-ACCESS-KEY": api_key,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
            "KALSHI-ACCESS-TIMESTAMP": timestamp_str,
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    f"{base_url}{path}",
                    headers=headers,
                    timeout=10
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        balance = result.get("balance", 0) / 100
                        logger.info(f"   💰 Account balance: ${balance:,.2f}")
                        
                        if balance > 0:
                            logger.info(f"   ✅ Funded account detected!")
                    elif response.status == 401:
                        logger.info(f"   ❌ Auth failed (wrong environment)")
                    else:
                        logger.info(f"   ⚠️  HTTP {response.status}")
            except Exception as e:
                logger.info(f"   ❌ Request failed: {e}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("DIAGNOSIS & NEXT STEPS")
    logger.info("="*70)
    
    logger.info("\n📋 If you see:")
    logger.info("   ✅ 'Connected successfully' on PRODUCTION")
    logger.info("   ✅ YES Ask > $0 on PRODUCTION")
    logger.info("   ✅ Volume > 0 on PRODUCTION")
    logger.info("   → You're on production! Markets might just be inactive right now.")
    
    logger.info("\n   ❌ Auth failed on PRODUCTION but works on DEMO")
    logger.info("   → Your credentials are still for DEMO. Get new production API key!")
    
    logger.info("\n   ⚠️  Both show $0 and 0 volume")
    logger.info("   → Markets genuinely have no liquidity. Check Kalshi website.")
    
    logger.info("\n🔗 Get PRODUCTION credentials here:")
    logger.info("   https://kalshi.com/profile/api")
    logger.info("   (Make sure you're logged into REAL account, not demo!)")
    
    logger.info("\n" + "="*70 + "\n")

if __name__ == "__main__":
    asyncio.run(check_api_environment())
