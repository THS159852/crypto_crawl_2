"""
================================================================================
CRYPTO TELEGRAM BOT - Professional Cryptocurrency Price & Chart Bot
================================================================================
Author: Your Name
Version: 2.0 (Optimized)
Description: A high-performance Telegram bot for real-time cryptocurrency 
             price tracking, candlestick charts, and market analysis.

Features:
- Real-time price from Binance, CCXT exchanges, DexScreener
- Candlestick charts with customizable themes
- Buy/Sell volume analysis
- Market cap, ATH, and supply information from CoinGecko
- Smart caching for fast response times
- Async architecture for high concurrency

================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import sys                              # System-specific parameters and functions
import asyncio                          # Async I/O, event loop, coroutines
import aiohttp                          # Async HTTP client (replaces requests for better performance)
import io                               # Core tools for working with streams (BytesIO for chart images)
import re                               # Regular expressions for pattern matching
import os                               # Operating system interface (environment variables)
import pandas as pd                     # Data manipulation and analysis library
import mplfinance as mpf                # Financial charts (candlestick) plotting
import ccxt.async_support as ccxt       # Cryptocurrency exchange trading library (async version)
from datetime import datetime           # Date and time handling
from functools import lru_cache         # Least Recently Used cache decorator
from cachetools import TTLCache         # Time-To-Live cache implementation
from telegram import Update             # Telegram bot update object
from telegram.ext import (              # Telegram bot extension utilities
    Application,                        # Main application class
    CommandHandler,                     # Handler for /commands
    MessageHandler,                     # Handler for text messages
    filters,                            # Message filters
    ContextTypes                        # Type hints for context
)
from telegram.constants import ParseMode  # Message formatting modes (MARKDOWN, HTML)
from dotenv import load_dotenv          # Load environment variables from .env file

# ==============================================================================
# CONFIGURATION & INITIALIZATION
# ==============================================================================

# --- UTF-8 Encoding Setup ---
# Ensures console output supports Vietnamese and special characters on Windows
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')  # Reconfigure stdout to UTF-8
        sys.stderr.reconfigure(encoding='utf-8')  # Reconfigure stderr to UTF-8
except: 
    pass  # Silently ignore if reconfiguration fails (e.g., on some platforms)

# --- Load Environment Variables ---
load_dotenv()  # Load variables from .env file into environment
BOT_TOKEN = os.environ.get('BOT_TOKEN')  # Get Telegram bot token

# Validate bot token exists
if not BOT_TOKEN:
    print("ERROR: Missing BOT_TOKEN.")
    sys.exit(1)  # Exit with error code if token not found

# ==============================================================================
# API ENDPOINTS
# ==============================================================================
BINANCE_API_URL = "https://api.binance.com/api/v3"           # Binance Spot API
BINANCE_F_API_URL = "https://fapi.binance.com/fapi/v1"       # Binance Futures API
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"       # CoinGecko API (free tier)
DEXSCREENER_API_URL = "https://api.dexscreener.com/latest/dex"  # DexScreener DEX API

# ==============================================================================
# CONSTANTS
# ==============================================================================
# Priority order for exchange lookups (fastest first)
PRIORITY_EXCHANGES = ['binance', 'bybit', 'okx', 'gateio', 'mexc', 'kucoin']

# Regex pattern to validate cryptocurrency symbols (1-10 alphanumeric characters)
VALID_SYMBOL_PATTERN = re.compile(r'^[a-zA-Z0-9]{1,10}$')

# Stablecoin quote currencies to try when looking up prices
STABLES = ['USDT', 'USDC', 'FDUSD', 'USD']

# ==============================================================================
# CACHING SYSTEM
# ==============================================================================
# TTLCache: Time-based expiring cache to reduce API calls and improve response time
# Parameters: maxsize = max items in cache, ttl = time-to-live in seconds

# Cache for CoinGecko coin IDs (rarely changes, 1 hour TTL)
coingecko_id_cache = TTLCache(maxsize=500, ttl=3600)

# Cache for price data (short TTL for real-time accuracy, 10 seconds)
price_cache = TTLCache(maxsize=200, ttl=10)

# Cache for market data like trending/gainers/losers (1 minute TTL)
market_cache = TTLCache(maxsize=100, ttl=60)

# Cache for CoinGecko detailed info like ATH, market cap (5 minutes TTL)
cg_info_cache = TTLCache(maxsize=100, ttl=300)

# ==============================================================================
# HTTP SESSION MANAGEMENT
# ==============================================================================
# Global HTTP session for connection pooling and reuse
_http_session: aiohttp.ClientSession | None = None


async def get_http_session() -> aiohttp.ClientSession:
    """
    Get or create a singleton HTTP session with connection pooling.
    
    Benefits of connection pooling:
    - Reuses TCP connections instead of creating new ones for each request
    - Reduces latency by avoiding TCP handshake overhead
    - DNS caching reduces DNS lookup time
    
    Returns:
        aiohttp.ClientSession: Configured HTTP session
    """
    global _http_session
    
    # Create new session if none exists or if current one is closed
    if _http_session is None or _http_session.closed:
        # Configure timeouts: 10s total, 5s for connection establishment
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        
        # Configure connection pool:
        # - limit=50: Max 50 simultaneous connections total
        # - limit_per_host=10: Max 10 connections per host
        # - ttl_dns_cache=300: Cache DNS results for 5 minutes
        connector = aiohttp.TCPConnector(
            limit=50, 
            limit_per_host=10, 
            ttl_dns_cache=300
        )
        
        _http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    
    return _http_session


async def close_http_session():
    """
    Close the global HTTP session gracefully.
    Should be called when shutting down the bot to release resources.
    """
    global _http_session
    if _http_session and not _http_session.closed:
        await _http_session.close()
        _http_session = None


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def is_contract_address(address: str) -> bool:
    """
    Check if a string is a valid Ethereum/EVM contract address.
    
    Valid contract addresses:
    - Start with '0x' prefix
    - Have exactly 42 characters (0x + 40 hex characters)
    
    Args:
        address: String to validate
        
    Returns:
        bool: True if valid contract address format
    """
    return address.startswith('0x') and len(address) == 42


def format_price(price: float) -> str:
    """
    Format a price value with appropriate decimal places based on magnitude.
    
    Formatting rules:
    - >= $10: 2 decimal places (e.g., $1,234.56)
    - >= $1: 3 decimal places (e.g., $1.234)
    - >= $0.001: 4 decimal places (e.g., $0.0012)
    - < $0.001: 8 decimal places (e.g., $0.00000012) - for small cap coins
    
    Args:
        price: Price value to format
        
    Returns:
        str: Formatted price string with $ prefix
    """
    if price is None: 
        return "N/A"
    if abs(price) >= 10: 
        return f"${price:,.2f}"      # $1,234.56
    if abs(price) >= 1: 
        return f"${price:,.3f}"      # $1.234
    if abs(price) >= 0.001: 
        return f"${price:,.4f}"      # $0.0012
    return f"${price:,.8f}"          # $0.00000012


def safe_eval(expression: str) -> float | None:
    """
    Safely evaluate a mathematical expression string.
    
    Security measures:
    - Only allows digits, operators (+, -, *, /), parentheses, decimal points, spaces
    - Runs eval() with empty builtins to prevent code injection
    
    Args:
        expression: Mathematical expression string (e.g., "100 * 2.5 + 50")
        
    Returns:
        float | None: Result of calculation, or None if invalid/unsafe expression
    """
    # Validate expression contains only safe characters
    if not re.match(r'^[0-9\s\+\-\*\/\(\)\.]+$', expression): 
        return None
    try: 
        # Evaluate with restricted environment (no access to builtins)
        return eval(expression, {"__builtins__": {}}, {})
    except: 
        return None


# ==============================================================================
# ASYNC DATA FETCHING FUNCTIONS
# ==============================================================================

async def fetch_json(url: str, timeout: float = 5) -> dict | list | None:
    """
    Generic async JSON fetcher with error handling.
    
    This is the base function for all API calls, providing:
    - Async non-blocking requests
    - Automatic JSON parsing
    - Error handling with graceful failure
    
    Args:
        url: API endpoint URL
        timeout: Request timeout in seconds (default: 5s)
        
    Returns:
        dict | list | None: Parsed JSON data, or None on error
    """
    try:
        session = await get_http_session()
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status == 200:
                return await resp.json()
    except: 
        pass  # Silently handle errors (timeout, connection, parsing)
    return None


async def get_coingecko_id(symbol: str) -> str | None:
    """
    Get CoinGecko coin ID from ticker symbol with caching.
    
    CoinGecko uses unique IDs (e.g., 'bitcoin', 'ethereum') rather than 
    ticker symbols (BTC, ETH). This function maps symbols to IDs.
    
    Caching: Results cached for 1 hour (symbols rarely change IDs)
    
    Args:
        symbol: Cryptocurrency ticker symbol (e.g., 'BTC', 'ETH')
        
    Returns:
        str | None: CoinGecko coin ID, or None if not found
    """
    symbol_lower = symbol.lower()
    
    # Check cache first
    if symbol_lower in coingecko_id_cache:
        return coingecko_id_cache[symbol_lower]
    
    # Search CoinGecko API
    data = await fetch_json(f"{COINGECKO_API_URL}/search?query={symbol}")
    if data:
        # Find exact symbol match (case-insensitive)
        for c in data.get('coins', []):
            if c.get('symbol', '').lower() == symbol_lower:
                cg_id = c.get('id')
                coingecko_id_cache[symbol_lower] = cg_id  # Cache result
                return cg_id
    return None


async def get_coingecko_info(symbol: str) -> dict | None:
    """
    Get detailed coin information from CoinGecko with caching.
    
    Retrieves:
    - All-Time High (ATH) price and percentage from ATH
    - All-Time Low (ATL) price and percentage from ATL
    - ATH/ATL dates
    - Market capitalization and rank
    - Circulating supply, Total supply, Max supply
    - 24h High/Low
    - Price changes (1h, 24h, 7d, 30d)
    - Trading volume
    - Fully Diluted Valuation (FDV)
    
    Caching: Results cached for 5 minutes
    
    Args:
        symbol: Cryptocurrency ticker symbol
        
    Returns:
        dict | None: Coin information dictionary, or None if not found
    """
    cache_key = f"cg_info_{symbol.lower()}"
    
    # Check cache first
    if cache_key in cg_info_cache:
        return cg_info_cache[cache_key]
    
    # Get CoinGecko ID first
    cid = await get_coingecko_id(symbol)
    if not cid: 
        return None
    
    # Fetch detailed coin data
    data = await fetch_json(f"{COINGECKO_API_URL}/coins/{cid}", timeout=10)
    if not data: 
        return None
    
    # Extract market data
    md = data.get('market_data', {})
    result = {
        # Basic info
        'name': data.get('name', ''),
        'symbol': data.get('symbol', '').upper(),
        'rank': data.get('market_cap_rank', 0),
        
        # Current price data
        'current_price': md.get('current_price', {}).get('usd', 0),
        'high_24h': md.get('high_24h', {}).get('usd', 0),
        'low_24h': md.get('low_24h', {}).get('usd', 0),
        
        # Price changes
        'change_1h': md.get('price_change_percentage_1h_in_currency', {}).get('usd', 0),
        'change_24h': md.get('price_change_percentage_24h', 0),
        'change_7d': md.get('price_change_percentage_7d', 0),
        'change_30d': md.get('price_change_percentage_30d', 0),
        
        # ATH data
        'ath': md.get('ath', {}).get('usd', 0),
        'ath_change': md.get('ath_change_percentage', {}).get('usd', 0),
        'ath_date': md.get('ath_date', {}).get('usd', '').split('T')[0],
        
        # ATL data
        'atl': md.get('atl', {}).get('usd', 0),
        'atl_change': md.get('atl_change_percentage', {}).get('usd', 0),
        'atl_date': md.get('atl_date', {}).get('usd', '').split('T')[0],
        
        # Market cap & Volume
        'cap': md.get('market_cap', {}).get('usd', 0),
        'volume_24h': md.get('total_volume', {}).get('usd', 0),
        'fdv': md.get('fully_diluted_valuation', {}).get('usd', 0),
        
        # Supply
        'circulating': md.get('circulating_supply', 0),
        'total_supply': md.get('total_supply', 0),
        'max_supply': md.get('max_supply', 0),
    }
    
    cg_info_cache[cache_key] = result  # Cache result
    return result


def calculate_price_prediction(price_data: dict, cg_info: dict | None) -> dict:
    """
    Calculate price prediction based on technical indicators and market data.
    
    This uses multiple factors to estimate potential price movements:
    1. Recent momentum (5m, 1h, 24h changes)
    2. Distance from ATH/ATL
    3. Volume analysis
    4. Support/Resistance levels (24h high/low)
    
    DISCLAIMER: This is for educational purposes only and NOT financial advice.
    
    Args:
        price_data: Current price and change data
        cg_info: CoinGecko market data (optional)
        
    Returns:
        dict: Prediction data with bullish/bearish scenarios
    """
    current_price = price_data.get('price', 0)
    if current_price <= 0:
        return None
    
    # === COLLECT SIGNALS ===
    signals = {
        'bullish': 0,
        'bearish': 0,
        'neutral': 0
    }
    
    # Signal 1: Short-term momentum (5m, 1h)
    p5m = price_data.get('p5m', 0) or 0
    p1h = price_data.get('p1h', 0) or 0
    p24h = price_data.get('p24h', 0) or 0
    
    # 5-minute trend
    if p5m > 0.5:
        signals['bullish'] += 1
    elif p5m < -0.5:
        signals['bearish'] += 1
    else:
        signals['neutral'] += 1
    
    # 1-hour trend
    if p1h > 1:
        signals['bullish'] += 2
    elif p1h < -1:
        signals['bearish'] += 2
    else:
        signals['neutral'] += 1
    
    # 24-hour trend (stronger weight)
    if p24h > 3:
        signals['bullish'] += 3
    elif p24h < -3:
        signals['bearish'] += 3
    elif p24h > 0:
        signals['bullish'] += 1
    elif p24h < 0:
        signals['bearish'] += 1
    
    # Signal 2: ATH/ATL analysis (if available)
    if cg_info:
        ath_change = cg_info.get('ath_change', 0)
        atl_change = cg_info.get('atl_change', 0)
        
        # Far from ATH = potential upside
        if ath_change < -50:
            signals['bullish'] += 2
        elif ath_change < -80:
            signals['bullish'] += 3
        
        # Close to ATL = potential bounce
        if atl_change < 100:
            signals['bullish'] += 2
        
        # Close to ATH = potential resistance
        if ath_change > -10:
            signals['bearish'] += 2
        
        # 7d and 30d trends
        change_7d = cg_info.get('change_7d', 0) or 0
        change_30d = cg_info.get('change_30d', 0) or 0
        
        if change_7d > 10:
            signals['bullish'] += 2
        elif change_7d < -10:
            signals['bearish'] += 2
            
        if change_30d > 20:
            signals['bullish'] += 1
        elif change_30d < -20:
            signals['bearish'] += 1
    
    # === CALCULATE PREDICTION ===
    total_signals = signals['bullish'] + signals['bearish'] + signals['neutral']
    if total_signals == 0:
        total_signals = 1
    
    bullish_score = signals['bullish'] / total_signals * 100
    bearish_score = signals['bearish'] / total_signals * 100
    
    # Determine trend
    if bullish_score > bearish_score + 15:
        trend = 'BULLISH'
        trend_icon = '🟢'
        confidence = min(bullish_score, 85)
    elif bearish_score > bullish_score + 15:
        trend = 'BEARISH'
        trend_icon = '🔴'
        confidence = min(bearish_score, 85)
    else:
        trend = 'NEUTRAL'
        trend_icon = '🟡'
        confidence = 50
    
    # Calculate price targets based on volatility
    # Use recent price swings to estimate potential movement
    volatility = abs(p24h) if p24h else 5  # Default 5% if no data
    volatility = max(volatility, 2)  # Minimum 2%
    volatility = min(volatility, 30)  # Cap at 30%
    
    # Target calculations
    if trend == 'BULLISH':
        target_pct_high = volatility * 1.5
        target_pct_low = volatility * 0.3
        target_price_high = current_price * (1 + target_pct_high / 100)
        target_price_low = current_price * (1 - target_pct_low / 100)
        
        potential_gain = target_price_high - current_price
        potential_loss = current_price - target_price_low
        
    elif trend == 'BEARISH':
        target_pct_high = volatility * 0.3
        target_pct_low = volatility * 1.5
        target_price_high = current_price * (1 + target_pct_high / 100)
        target_price_low = current_price * (1 - target_pct_low / 100)
        
        potential_gain = target_price_high - current_price
        potential_loss = current_price - target_price_low
        
    else:  # NEUTRAL
        target_pct_high = volatility * 0.8
        target_pct_low = volatility * 0.8
        target_price_high = current_price * (1 + target_pct_high / 100)
        target_price_low = current_price * (1 - target_pct_low / 100)
        
        potential_gain = target_price_high - current_price
        potential_loss = current_price - target_price_low
    
    # Recommendation
    if trend == 'BULLISH' and confidence > 60:
        recommendation = '📈 Xu hướng LONG'
    elif trend == 'BEARISH' and confidence > 60:
        recommendation = '📉 Xu hướng SHORT'
    else:
        recommendation = '⏳ Chờ tín hiệu rõ hơn'
    
    return {
        'trend': trend,
        'trend_icon': trend_icon,
        'confidence': confidence,
        'bullish_score': bullish_score,
        'bearish_score': bearish_score,
        'target_high': target_price_high,
        'target_low': target_price_low,
        'target_pct_high': target_pct_high,
        'target_pct_low': target_pct_low,
        'potential_gain': potential_gain,
        'potential_loss': potential_loss,
        'recommendation': recommendation,
        'signals': signals
    }


async def get_binance_kline_change(trading_pair: str, interval: str, is_future: bool = False) -> float | None:
    """
    Calculate price change percentage from Binance kline (candlestick) data.
    
    Fetches the last 2 candles and calculates the percentage change
    between them. Used for 5-minute and 1-hour price changes.
    
    Args:
        trading_pair: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Kline interval (e.g., '5m', '1h', '1d')
        is_future: Use Futures API if True, Spot API if False
        
    Returns:
        float | None: Percentage change, or None on error
    """
    # Select appropriate API endpoint
    api_url = BINANCE_F_API_URL if is_future else BINANCE_API_URL
    url = f"{api_url}/klines?symbol={trading_pair}&interval={interval}&limit=2"
    
    data = await fetch_json(url)
    if not data or len(data) < 2: 
        return None
    
    try:
        prev_close = float(data[0][4])   # Previous candle close price (index 4)
        curr_price = float(data[1][4])   # Current candle close price
        
        if prev_close == 0: 
            return 0.0
        
        # Calculate percentage change: ((new - old) / old) * 100
        return ((curr_price - prev_close) / prev_close) * 100
    except: 
        return None


async def get_binance_price_data(symbol: str) -> tuple[dict | None, str, str]:
    """
    Fetch price data from Binance Spot and Futures APIs in parallel.
    
    Strategy:
    1. Try all combinations of Spot/Futures and stablecoins simultaneously
    2. Return the first successful result (prioritizes Spot over Futures)
    3. Fetch 5m and 1h kline changes in parallel for efficiency
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC')
        
    Returns:
        tuple: (price_data dict, source name, pair name) or (None, "", "")
    """
    symbol_upper = symbol.upper()
    
    async def fetch_binance(is_future: bool, stable: str):
        """Inner function to fetch from a specific Binance endpoint"""
        api_url = BINANCE_F_API_URL if is_future else BINANCE_API_URL
        pair = f"{symbol_upper}{stable}"
        url = f"{api_url}/ticker/24hr?symbol={pair}"
        
        data = await fetch_json(url)
        if not data: 
            return None
        
        try:
            price = float(data['lastPrice'])           # Current price
            p24h = float(data['priceChangePercent'])   # 24h change percentage
            
            # Fetch 1h and 5m kline changes in parallel for speed
            kline_tasks = [
                get_binance_kline_change(pair, '1h', is_future),
                get_binance_kline_change(pair, '5m', is_future)
            ]
            p1h, p5m = await asyncio.gather(*kline_tasks)
            
            # Calculate historical prices from current price and change percentages
            # Formula: old_price = current_price / (1 + change_percent/100)
            price_24h = price / (1 + p24h/100) if p24h != -100 else 0
            price_1h = price / (1 + p1h/100) if p1h else 0
            price_5m = price / (1 + p5m/100) if p5m else 0
            
            return {
                'price': price,
                'p5m': p5m, 'price_5m': price_5m,       # 5-minute data
                'p1h': p1h, 'price_1h': price_1h,       # 1-hour data
                'p24h': p24h, 'price_24h': price_24h,   # 24-hour data
                'vol': float(data['quoteVolume']),      # 24h volume in quote currency
                'vol_unit': stable,                      # Quote currency (USDT, USDC, etc.)
                'source': "Binance Futures" if is_future else "Sàn Binance",
                'pair': f"{symbol_upper}/{stable}"
            }
        except: 
            return None
    
    # Create tasks for all Spot/Futures + stablecoin combinations
    tasks = []
    for is_future in [False, True]:        # Spot first, then Futures
        for stable in ['USDT', 'USDC', 'FDUSD']:
            tasks.append(fetch_binance(is_future, stable))
    
    # Execute all requests in parallel
    results = await asyncio.gather(*tasks)
    
    # Return first successful result
    for r in results:
        if r:
            return r, r['source'], r['pair']
    
    return None, "", ""


async def get_ccxt_price(symbol: str) -> tuple[dict | None, str, str]:
    """
    Fetch price from CCXT-supported exchanges as fallback.
    
    Uses early return pattern: stops at first exchange that has the pair.
    Tries exchanges in order: gateio, mexc, kucoin, okx, bybit
    
    Args:
        symbol: Cryptocurrency symbol
        
    Returns:
        tuple: (price_data dict, source name, pair name) or (None, "", "")
    """
    symbol_upper = symbol.upper()
    
    # Try each exchange in sequence
    for ex_id in ['gateio', 'mexc', 'kucoin', 'okx', 'bybit']:
        ex = None
        try:
            # Initialize exchange with 5 second timeout
            ex = getattr(ccxt, ex_id)({'timeout': 5000})
            
            # Try each stablecoin pair
            for stable in STABLES:
                try:
                    t = await ex.fetch_ticker(f"{symbol_upper}/{stable}")
                    if t and t.get('last'):
                        price = t['last']
                        p24h = t.get('percentage', 0)
                        return {
                            'price': price,
                            'p24h': p24h,
                            'price_24h': price / (1 + p24h/100) if p24h else 0,
                            'vol': t.get('quoteVolume', 0),
                            'vol_unit': 'USDT'
                        }, f"Sàn {ex.name}", t['symbol']
                except: 
                    continue
        except: 
            pass
        finally:
            # Always close exchange connection to free resources
            if ex: 
                await ex.close()
    
    return None, "", ""


async def get_dexscreener_price(symbol: str) -> tuple[dict | None, str, str]:
    """
    Fetch price from DexScreener for DEX-traded tokens.
    
    Useful for:
    - New tokens not yet listed on CEX
    - DeFi tokens primarily traded on DEXs
    - Meme coins on Uniswap, PancakeSwap, etc.
    
    Selection strategy: Choose pair with highest USD liquidity
    
    Args:
        symbol: Token symbol to search
        
    Returns:
        tuple: (price_data dict, source name, pair name) or (None, "", "")
    """
    url = f"{DEXSCREENER_API_URL}/search?q={symbol}"
    data = await fetch_json(url)
    
    if not data or not data.get('pairs'): 
        return None, "", ""
    
    # Select pair with highest liquidity for most accurate price
    p = max(data['pairs'], key=lambda x: x.get('liquidity', {}).get('usd', 0))
    
    try:
        price = float(p['priceUsd'])
        price_data = {
            'price': price,
            'p5m': p.get('priceChange', {}).get('m5'),    # 5-minute change
            'p1h': p.get('priceChange', {}).get('h1'),    # 1-hour change
            'p24h': p.get('priceChange', {}).get('h24'),  # 24-hour change
            'vol': p.get('volume', {}).get('h24', 0),     # 24h volume
            'vol_unit': 'USD'
        }
        
        # Calculate historical prices from percentages
        if price_data['p5m']: 
            price_data['price_5m'] = price / (1 + price_data['p5m']/100)
        if price_data['p1h']: 
            price_data['price_1h'] = price / (1 + price_data['p1h']/100)
        if price_data['p24h']: 
            price_data['price_24h'] = price / (1 + price_data['p24h']/100)
        
        source = f"DexScreener ({p.get('dexId')})"
        pair_name = f"{p['baseToken']['symbol']}/{p['quoteToken']['symbol']}"
        return price_data, source, pair_name
    except:
        return None, "", ""


# ==============================================================================
# MAIN REPORT GENERATION
# ==============================================================================

async def get_token_report(symbol: str) -> str:
    """
    Generate a comprehensive price report for a cryptocurrency.
    
    Data flow:
    1. Check cache for recent report
    2. Try Binance (fastest, most liquid)
    3. Try CCXT exchanges (fallback for non-Binance coins)
    4. Try DexScreener (for DEX-only tokens)
    5. Fetch CoinGecko info in parallel with formatting
    6. Calculate price prediction
    7. Format and cache the result
    
    Report includes:
    - Current price with source
    - 5m, 1h, 24h price changes with historical prices
    - 24h High/Low
    - 24h trading volume
    - ATH/ATL and market cap from CoinGecko
    - Market rank and FDV
    - Supply information
    - Price prediction with targets
    - Disclaimer warning
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        
    Returns:
        str: Formatted report message in Markdown
    """
    symbol_upper = symbol.upper()
    
    # Check cache for recent report (10 second TTL)
    cache_key = f"report_{symbol_upper}"
    if cache_key in price_cache:
        return price_cache[cache_key]
    
    # === DATA FETCHING (waterfall strategy) ===
    
    # 1. Try Binance first (fastest and most reliable)
    price_data, source, pair_name = await get_binance_price_data(symbol)
    
    # 2. If not on Binance, try other CCXT exchanges
    if not price_data:
        price_data, source, pair_name = await get_ccxt_price(symbol)
    
    # 3. Last resort: DexScreener for DEX-only tokens
    if not price_data:
        price_data, source, pair_name = await get_dexscreener_price(symbol)
    
    # Return error if no data found anywhere
    if not price_data:
        return f"😕 Không tìm thấy thông tin cho **{symbol_upper}**."
    
    # Start fetching CoinGecko info in background (don't wait)
    cg_task = asyncio.create_task(get_coingecko_info(symbol))
    
    # === FORMAT OUTPUT ===
    msg = f"🪙 **{pair_name}**\n"
    msg += f"📍 Nguồn: `{source}`\n"
    msg += "══════════════════════\n"
    
    # Current price (large emphasis)
    msg += f"💰 **Giá hiện tại:** `{format_price(price_data['price'])}`\n"
    msg += "══════════════════════\n"
    
    # Price changes section
    msg += "📊 **BIẾN ĐỘNG GIÁ**\n"
    
    # 5-minute price change
    if price_data.get('p5m') is not None:
        icon = '🟢' if price_data['p5m'] >= 0 else '🔴'
        val_5m = format_price(price_data.get('price_5m', 0)) if price_data.get('price_5m') else 'N/A'
        msg += f"├ 5 phút:  {icon} `{price_data['p5m']:+.2f}%` ({val_5m})\n"
    
    # 1-hour price change
    if price_data.get('p1h') is not None:
        icon = '🟢' if price_data['p1h'] >= 0 else '🔴'
        val_1h = format_price(price_data.get('price_1h', 0)) if price_data.get('price_1h') else 'N/A'
        msg += f"├ 1 giờ:   {icon} `{price_data['p1h']:+.2f}%` ({val_1h})\n"
        
    # 24-hour price change
    if price_data.get('p24h') is not None:
        icon = '🟢' if price_data['p24h'] >= 0 else '🔴'
        val_24h = format_price(price_data.get('price_24h', 0)) if price_data.get('price_24h') else 'N/A'
        msg += f"└ 24 giờ:  {icon} `{price_data['p24h']:+.2f}%` ({val_24h})\n"

    msg += "──────────────────────\n"
    msg += f"📈 KL giao dịch 24h: `${price_data['vol']:,.0f}`\n"

    # Wait for CoinGecko info and add if available
    cg_info = await cg_task
    
    if cg_info:
        msg += "══════════════════════\n"
        msg += "📋 **THÔNG TIN THỊ TRƯỜNG**\n"
        
        # Market rank
        if cg_info.get('rank'):
            msg += f"├ Xếp hạng: `#{cg_info['rank']}`\n"
        
        # 24h High/Low
        if cg_info.get('high_24h') and cg_info.get('low_24h'):
            msg += f"├ Cao 24h: `{format_price(cg_info['high_24h'])}`\n"
            msg += f"├ Thấp 24h: `{format_price(cg_info['low_24h'])}`\n"
        
        # Extended price changes
        if cg_info.get('change_7d'):
            icon = '🟢' if cg_info['change_7d'] >= 0 else '🔴'
            msg += f"├ 7 ngày: {icon} `{cg_info['change_7d']:+.2f}%`\n"
        if cg_info.get('change_30d'):
            icon = '🟢' if cg_info['change_30d'] >= 0 else '🔴'
            msg += f"└ 30 ngày: {icon} `{cg_info['change_30d']:+.2f}%`\n"
        
        msg += "──────────────────────\n"
        
        # ATH Section
        msg += "🏆 **ALL-TIME HIGH (ATH)**\n"
        msg += f"├ Giá ATH: `{format_price(cg_info['ath'])}`\n"
        msg += f"├ Từ ATH: `{cg_info['ath_change']:+.2f}%`\n"
        msg += f"└ Ngày: `{cg_info['ath_date']}`\n"
        
        # ATL Section
        if cg_info.get('atl'):
            msg += "──────────────────────\n"
            msg += "📉 **ALL-TIME LOW (ATL)**\n"
            msg += f"├ Giá ATL: `{format_price(cg_info['atl'])}`\n"
            msg += f"├ Từ ATL: `{cg_info['atl_change']:+.2f}%`\n"
            msg += f"└ Ngày: `{cg_info['atl_date']}`\n"
        
        msg += "──────────────────────\n"
        
        # Market Cap & Volume
        msg += "💎 **VỐN HÓA & CUNG**\n"
        msg += f"├ Vốn hóa: `${cg_info['cap']:,.0f}`\n"
        
        if cg_info.get('fdv'):
            msg += f"├ FDV: `${cg_info['fdv']:,.0f}`\n"
        
        msg += f"├ Lưu thông: `{cg_info['circulating']:,.0f}`\n"
        
        total_supply = f"{cg_info['total_supply']:,.0f}" if cg_info.get('total_supply') else "∞"
        msg += f"├ Tổng cung: `{total_supply}`\n"
        
        max_supply = f"{cg_info['max_supply']:,.0f}" if cg_info.get('max_supply') else "∞"
        msg += f"└ Cung tối đa: `{max_supply}`\n"
    
    # === PRICE PREDICTION SECTION ===
    prediction = calculate_price_prediction(price_data, cg_info)
    
    if prediction:
        msg += "══════════════════════\n"
        msg += f"🔮 **DỰ ĐOÁN XU HƯỚNG**\n"
        msg += f"──────────────────────\n"
        
        # Trend indicator
        msg += f"📊 Xu hướng: {prediction['trend_icon']} **{prediction['trend']}**\n"
        msg += f"📈 Độ tin cậy: `{prediction['confidence']:.1f}%`\n"
        msg += f"──────────────────────\n"
        
        # Score breakdown
        msg += f"├ Tín hiệu Tăng: `{prediction['bullish_score']:.1f}%`\n"
        msg += f"└ Tín hiệu Giảm: `{prediction['bearish_score']:.1f}%`\n"
        msg += f"──────────────────────\n"
        
        # Price targets
        msg += "🎯 **MỤC TIÊU GIÁ (24h)**\n"
        
        # Bullish scenario
        msg += f"🟢 **Nếu TĂNG:**\n"
        msg += f"├ Mục tiêu: `{format_price(prediction['target_high'])}`\n"
        msg += f"├ Tăng: `+{format_price(prediction['potential_gain'])}`\n"
        msg += f"└ % Tăng: `+{prediction['target_pct_high']:.2f}%`\n"
        
        # Bearish scenario
        msg += f"🔴 **Nếu GIẢM:**\n"
        msg += f"├ Mục tiêu: `{format_price(prediction['target_low'])}`\n"
        msg += f"├ Giảm: `-{format_price(prediction['potential_loss'])}`\n"
        msg += f"└ % Giảm: `-{prediction['target_pct_low']:.2f}%`\n"
        
        msg += f"──────────────────────\n"
        msg += f"💡 **Gợi ý:** {prediction['recommendation']}\n"
    
    # === DISCLAIMER WARNING ===
    msg += "══════════════════════\n"
    msg += "⚠️ **CẢNH BÁO QUAN TRỌNG** ⚠️\n"
    msg += "```\n"
    msg += "🚨 ĐÂY CHỈ LÀ DỰ ĐOÁN!\n"
    msg += "Thị trường crypto biến động mạnh.\n"
    msg += "MỌI SỰ MẤT MÁT TÀI CHÍNH\n"
    msg += "CHÚNG TÔI KHÔNG CHỊU TRÁCH NHIỆM!\n"
    msg += "Hãy DYOR - Do Your Own Research!\n"
    msg += "```\n"
    msg += "══════════════════════\n"

    # Add timestamp
    msg += f"🕐 `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
    
    # Cache the result for future requests
    price_cache[cache_key] = msg
    return msg


# ==============================================================================
# CHART GENERATION
# ==============================================================================

# --- Chart Image URLs (Fast - No rendering required) ---
# These services provide pre-rendered chart images

def get_tradingview_chart_url(symbol: str, interval: str = '1h', theme: str = 'light') -> str:
    """
    Generate TradingView mini chart URL.
    
    This returns a URL to TradingView's chart widget that can be
    screenshotted or embedded. Very fast as charts are pre-rendered.
    
    Args:
        symbol: Trading pair (e.g., 'BTC', 'ETH')
        interval: Timeframe - maps to TradingView intervals
        theme: 'light' or 'dark'
        
    Returns:
        str: TradingView chart URL
    """
    # Map intervals to TradingView format
    tv_intervals = {
        '1m': '1', '5m': '5', '15m': '15', '30m': '30',
        '1h': '60', '2h': '120', '4h': '240',
        '1d': 'D', '1w': 'W', '1M': 'M'
    }
    tv_interval = tv_intervals.get(interval, '60')
    tv_theme = 'dark' if theme == 'dark' else 'light'
    
    return f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol.upper()}USDT&interval={tv_interval}&theme={tv_theme}"


async def get_chart_image_from_api(symbol: str, interval: str = '1h', theme: str = 'light') -> tuple[io.BytesIO | None, str | None]:
    """
    Get chart image from external chart image APIs.
    
    Method 1: Use chart-img.com (free TradingView screenshot service)
    Method 2: Use quickchart.io for custom charts
    
    This is MUCH faster than generating charts locally because:
    - No matplotlib rendering
    - No OHLCV data fetching
    - Pre-cached images on CDN
    
    Args:
        symbol: Cryptocurrency symbol
        interval: Chart timeframe
        theme: 'light' or 'dark'
        
    Returns:
        tuple: (BytesIO image, chart name) or (None, None)
    """
    symbol_upper = symbol.upper()
    
    # Map intervals to TradingView format
    tv_intervals = {
        '1m': '1', '5m': '5', '15m': '15', '30m': '30',
        '1h': '60', '2h': '120', '4h': '240',
        '1d': 'D', '1w': 'W', '1M': 'M'
    }
    tv_interval = tv_intervals.get(interval, '60')
    tv_theme = 'dark' if theme == 'dark' else 'light'
    
    # --- Method 1: chart-img.com (TradingView screenshots) ---
    # This service takes screenshots of TradingView charts
    # Free tier: Limited requests, but very fast
    chart_img_url = (
        f"https://api.chart-img.com/v1/tradingview/advanced-chart?"
        f"symbol=BINANCE:{symbol_upper}USDT"
        f"&interval={tv_interval}"
        f"&theme={tv_theme}"
        f"&studies=Volume"
        f"&width=800"
        f"&height=450"
    )
    
    try:
        session = await get_http_session()
        async with session.get(chart_img_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                image_data = await resp.read()
                buf = io.BytesIO(image_data)
                buf.seek(0)
                return buf, f"BINANCE:{symbol_upper}USDT"
    except:
        pass
    
    # --- Method 2: Try different exchanges on chart-img ---
    for exchange in ['BYBIT', 'OKX', 'COINBASE']:
        try:
            alt_url = (
                f"https://api.chart-img.com/v1/tradingview/advanced-chart?"
                f"symbol={exchange}:{symbol_upper}USDT"
                f"&interval={tv_interval}"
                f"&theme={tv_theme}"
                f"&width=800"
                f"&height=450"
            )
            async with session.get(alt_url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
                if resp.status == 200:
                    image_data = await resp.read()
                    buf = io.BytesIO(image_data)
                    buf.seek(0)
                    return buf, f"{exchange}:{symbol_upper}USDT"
        except:
            continue
    
    return None, None


async def generate_ccxt_chart(symbol: str, interval: str, theme: str = 'light') -> tuple[io.BytesIO | None, str | None]:
    """
    Generate a TradingView-style candlestick chart using CCXT exchange data.
    
    Features:
    - TradingView-style dark/light theme
    - OHLC price display in header
    - Current price line (dashed)
    - Price change percentage
    - Standard green/red candle colors
    - Y-axis (price) on right side
    - Volume bars at bottom with color matching candles
    - 100 candles of history
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC')
        interval: Timeframe (e.g., '1h', '4h', '1d')
        theme: 'light' (default) or 'dark'
        
    Returns:
        tuple: (BytesIO image buffer, chart title) or (None, None)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import numpy as np
    
    print(f"Drawing chart {symbol} {interval}...")
    
    # Try each exchange in priority order
    for ex_id in PRIORITY_EXCHANGES:
        ex = None
        try:
            ex = getattr(ccxt, ex_id)({'timeout': 10000})
            await ex.load_markets()  # Load available trading pairs
            
            # Try each stablecoin pair
            for s in STABLES:
                pair = f"{symbol.upper()}/{s}"
                if pair in ex.markets:
                    # Fetch OHLCV (Open, High, Low, Close, Volume) data
                    ohlcv = await ex.fetch_ohlcv(pair, timeframe=interval, limit=100)
                    if not ohlcv: 
                        continue
                    
                    # Convert to pandas DataFrame
                    df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    df['Date'] = pd.to_datetime(df['Date'], unit='ms')  # Convert timestamp to datetime
                    df = df.set_index('Date')
                    
                    # Get current candle data for OHLC display
                    current_open = df['Open'].iloc[-1]
                    current_high = df['High'].iloc[-1]
                    current_low = df['Low'].iloc[-1]
                    current_close = df['Close'].iloc[-1]
                    
                    # Calculate price change
                    if len(df) > 1:
                        prev_close = df['Close'].iloc[-2]
                        price_change = current_close - prev_close
                        price_change_pct = ((current_close - prev_close) / prev_close) * 100
                    else:
                        price_change = 0
                        price_change_pct = 0
                    
                    # Determine if current candle is bullish or bearish
                    is_bullish = current_close >= current_open
                    
                    # TradingView-style colors
                    if theme == 'dark':
                        bg_color = '#131722'      # Dark background
                        text_color = '#D1D4DC'    # Light gray text
                        grid_color = '#2A2E39'    # Dark grid
                        border_color = '#2A2E39'
                    else:
                        bg_color = '#FFFFFF'      # White background
                        text_color = '#131722'    # Dark text
                        grid_color = '#E0E3EB'    # Light grid
                        border_color = '#E0E3EB'
                    
                    up_color = '#26A69A'       # TradingView green
                    down_color = '#EF5350'     # TradingView red
                    
                    # Configure chart colors (TradingView-style)
                    mc = mpf.make_marketcolors(
                        up=up_color,
                        down=down_color,
                        edge={'up': up_color, 'down': down_color},
                        wick={'up': up_color, 'down': down_color},
                        volume={'up': up_color, 'down': down_color},
                    )
                    
                    # Create chart style
                    s_style = mpf.make_mpf_style(
                        base_mpf_style='nightclouds' if theme == 'dark' else 'yahoo',
                        marketcolors=mc,
                        facecolor=bg_color,
                        edgecolor=border_color,
                        figcolor=bg_color,
                        gridcolor=grid_color,
                        gridstyle='-',
                        gridaxis='both',
                        y_on_right=True,
                        rc={
                            'axes.labelcolor': text_color,
                            'axes.edgecolor': border_color,
                            'xtick.color': text_color,
                            'ytick.color': text_color,
                            'font.size': 10,
                        }
                    )
                    
                    # Create figure with extra space for header
                    fig, axes = mpf.plot(
                        df, 
                        type='candle',
                        style=s_style,
                        volume=True,
                        figsize=(12, 7),
                        returnfig=True,
                        panel_ratios=(3, 1),  # Main chart 3x height of volume
                        tight_layout=False,
                    )
                    
                    # Get the main price axis
                    ax_main = axes[0]
                    ax_vol = axes[2] if len(axes) > 2 else None
                    
                    # Add current price horizontal dashed line
                    price_line_color = up_color if is_bullish else down_color
                    ax_main.axhline(
                        y=current_close, 
                        color=price_line_color, 
                        linestyle='--', 
                        linewidth=1,
                        alpha=0.8
                    )
                    
                    # Add price label box on right side
                    bbox_props = dict(
                        boxstyle="round,pad=0.3", 
                        facecolor=price_line_color, 
                        edgecolor=price_line_color,
                        alpha=0.9
                    )
                    ax_main.annotate(
                        f'{current_close:.8f}'.rstrip('0').rstrip('.'),
                        xy=(1.0, current_close),
                        xycoords=('axes fraction', 'data'),
                        fontsize=9,
                        color='white',
                        fontweight='bold',
                        ha='left',
                        va='center',
                        bbox=bbox_props
                    )
                    
                    # Format exchange name for display
                    exchange_display = ex.name.replace('Binance', 'Binance')
                    pair_display = pair.replace('/', '')
                    
                    # Add TradingView-style header
                    header_y = 0.97
                    
                    # Symbol and timeframe (main title)
                    title_text = f"{exchange_display}:{pair_display} • {interval}"
                    fig.text(0.02, header_y, title_text, 
                             fontsize=14, fontweight='bold', color=text_color,
                             transform=fig.transFigure, va='top')
                    
                    # OHLC values with colors
                    ohlc_y = 0.93
                    ohlc_spacing = 0.12
                    
                    # Format price for display
                    def fmt_price(p):
                        if p >= 1000:
                            return f"{p:,.2f}"
                        elif p >= 1:
                            return f"{p:.2f}"
                        elif p >= 0.01:
                            return f"{p:.4f}"
                        else:
                            return f"{p:.8f}"
                    
                    # O H L C labels with values
                    change_color = up_color if price_change_pct >= 0 else down_color
                    
                    fig.text(0.02, ohlc_y, f"O ", fontsize=10, color='#787B86',
                             transform=fig.transFigure, va='top')
                    fig.text(0.035, ohlc_y, f"{fmt_price(current_open)}", fontsize=10, 
                             color=change_color, transform=fig.transFigure, va='top')
                    
                    fig.text(0.02 + ohlc_spacing, ohlc_y, f"H ", fontsize=10, color='#787B86',
                             transform=fig.transFigure, va='top')
                    fig.text(0.035 + ohlc_spacing, ohlc_y, f"{fmt_price(current_high)}", fontsize=10, 
                             color=change_color, transform=fig.transFigure, va='top')
                    
                    fig.text(0.02 + ohlc_spacing*2, ohlc_y, f"L ", fontsize=10, color='#787B86',
                             transform=fig.transFigure, va='top')
                    fig.text(0.035 + ohlc_spacing*2, ohlc_y, f"{fmt_price(current_low)}", fontsize=10, 
                             color=change_color, transform=fig.transFigure, va='top')
                    
                    fig.text(0.02 + ohlc_spacing*3, ohlc_y, f"C ", fontsize=10, color='#787B86',
                             transform=fig.transFigure, va='top')
                    fig.text(0.035 + ohlc_spacing*3, ohlc_y, f"{fmt_price(current_close)}", fontsize=10, 
                             color=change_color, transform=fig.transFigure, va='top')
                    
                    # Price change percentage
                    change_sign = '+' if price_change_pct >= 0 else ''
                    fig.text(0.02 + ohlc_spacing*4.2, ohlc_y, 
                             f"({change_sign}{price_change_pct:.2f}%)", 
                             fontsize=10, color=change_color,
                             transform=fig.transFigure, va='top')
                    
                    # Adjust subplot positioning to make room for header
                    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.92)
                    
                    # Remove default title
                    ax_main.set_title('')
                    
                    # Style the axes
                    ax_main.spines['top'].set_visible(False)
                    ax_main.spines['left'].set_visible(False)
                    if ax_vol:
                        ax_vol.spines['top'].set_visible(False)
                        ax_vol.spines['left'].set_visible(False)
                    
                    # Render chart to BytesIO buffer
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=120, 
                                facecolor=bg_color, edgecolor='none',
                                bbox_inches='tight', pad_inches=0.1)
                    plt.close(fig)
                    buf.seek(0)
                    return buf, f"{exchange_display}:{pair_display}"
        except Exception as e:
            print(f"Chart error for {ex_id}: {e}")
            pass
        finally:
            if ex: 
                await ex.close()
    
    return None, None


async def generate_coingecko_chart(coin_id: str, interval: str, theme: str = 'light') -> tuple[io.BytesIO | None, str | None]:
    """
    Generate a TradingView-style candlestick chart using CoinGecko OHLC data.
    
    Used as fallback when exchange data is not available.
    Note: CoinGecko has limited timeframe options.
    
    Args:
        coin_id: CoinGecko coin ID (e.g., 'bitcoin')
        interval: Timeframe (e.g., '1h', '1d')
        theme: 'light' (default) or 'dark'
        
    Returns:
        tuple: (BytesIO image buffer, chart title) or (None, None)
    """
    import matplotlib.pyplot as plt
    
    # Determine data range based on interval
    days = '1' if interval in ['5m', '15m', '1h', '4h'] else '90'
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
    
    data = await fetch_json(url, timeout=10)
    if not data: 
        return None, None
    
    try:
        # Create DataFrame from OHLC data
        df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df = df.set_index('Date')
        
        # Resample to desired interval
        rule = interval.replace('m', 'T') if interval in ['5m', '15m'] else '1h' if interval in ['1h', '4h'] else '1D'
        if interval != '1d': 
            df = df.resample(rule).agg({
                'Open': 'first', 
                'High': 'max', 
                'Low': 'min', 
                'Close': 'last'
            }).dropna()
        
        # Get current candle data for OHLC display
        current_open = df['Open'].iloc[-1]
        current_high = df['High'].iloc[-1]
        current_low = df['Low'].iloc[-1]
        current_close = df['Close'].iloc[-1]
        
        # Calculate price change
        if len(df) > 1:
            prev_close = df['Close'].iloc[-2]
            price_change_pct = ((current_close - prev_close) / prev_close) * 100
        else:
            price_change_pct = 0
        
        is_bullish = current_close >= current_open
        
        # TradingView-style colors
        if theme == 'dark':
            bg_color = '#131722'
            text_color = '#D1D4DC'
            grid_color = '#2A2E39'
            border_color = '#2A2E39'
        else:
            bg_color = '#FFFFFF'
            text_color = '#131722'
            grid_color = '#E0E3EB'
            border_color = '#E0E3EB'
        
        up_color = '#26A69A'
        down_color = '#EF5350'
        
        # Configure chart colors
        mc = mpf.make_marketcolors(
            up=up_color,
            down=down_color,
            edge={'up': up_color, 'down': down_color},
            wick={'up': up_color, 'down': down_color},
        )
        
        s_style = mpf.make_mpf_style(
            base_mpf_style='nightclouds' if theme == 'dark' else 'yahoo',
            marketcolors=mc,
            facecolor=bg_color,
            edgecolor=border_color,
            figcolor=bg_color,
            gridcolor=grid_color,
            gridstyle='-',
            y_on_right=True,
            rc={
                'axes.labelcolor': text_color,
                'axes.edgecolor': border_color,
                'xtick.color': text_color,
                'ytick.color': text_color,
            }
        )
        
        # Create figure
        fig, axes = mpf.plot(
            df, 
            type='candle', 
            style=s_style, 
            volume=False,
            figsize=(12, 6),
            returnfig=True,
            tight_layout=False,
        )
        
        ax_main = axes[0]
        
        # Add current price line
        price_line_color = up_color if is_bullish else down_color
        ax_main.axhline(
            y=current_close, 
            color=price_line_color, 
            linestyle='--', 
            linewidth=1,
            alpha=0.8
        )
        
        # Add price label
        bbox_props = dict(
            boxstyle="round,pad=0.3", 
            facecolor=price_line_color, 
            edgecolor=price_line_color,
            alpha=0.9
        )
        ax_main.annotate(
            f'{current_close:.8f}'.rstrip('0').rstrip('.'),
            xy=(1.0, current_close),
            xycoords=('axes fraction', 'data'),
            fontsize=9,
            color='white',
            fontweight='bold',
            ha='left',
            va='center',
            bbox=bbox_props
        )
        
        # Add header
        symbol_display = coin_id.upper()
        header_y = 0.97
        title_text = f"CoinGecko:{symbol_display}/USD • {interval}"
        fig.text(0.02, header_y, title_text, 
                 fontsize=14, fontweight='bold', color=text_color,
                 transform=fig.transFigure, va='top')
        
        # OHLC display
        def fmt_price(p):
            if p >= 1000:
                return f"{p:,.2f}"
            elif p >= 1:
                return f"{p:.2f}"
            elif p >= 0.01:
                return f"{p:.4f}"
            else:
                return f"{p:.8f}"
        
        ohlc_y = 0.93
        ohlc_spacing = 0.12
        change_color = up_color if price_change_pct >= 0 else down_color
        
        fig.text(0.02, ohlc_y, f"O ", fontsize=10, color='#787B86',
                 transform=fig.transFigure, va='top')
        fig.text(0.035, ohlc_y, f"{fmt_price(current_open)}", fontsize=10, 
                 color=change_color, transform=fig.transFigure, va='top')
        
        fig.text(0.02 + ohlc_spacing, ohlc_y, f"H ", fontsize=10, color='#787B86',
                 transform=fig.transFigure, va='top')
        fig.text(0.035 + ohlc_spacing, ohlc_y, f"{fmt_price(current_high)}", fontsize=10, 
                 color=change_color, transform=fig.transFigure, va='top')
        
        fig.text(0.02 + ohlc_spacing*2, ohlc_y, f"L ", fontsize=10, color='#787B86',
                 transform=fig.transFigure, va='top')
        fig.text(0.035 + ohlc_spacing*2, ohlc_y, f"{fmt_price(current_low)}", fontsize=10, 
                 color=change_color, transform=fig.transFigure, va='top')
        
        fig.text(0.02 + ohlc_spacing*3, ohlc_y, f"C ", fontsize=10, color='#787B86',
                 transform=fig.transFigure, va='top')
        fig.text(0.035 + ohlc_spacing*3, ohlc_y, f"{fmt_price(current_close)}", fontsize=10, 
                 color=change_color, transform=fig.transFigure, va='top')
        
        change_sign = '+' if price_change_pct >= 0 else ''
        fig.text(0.02 + ohlc_spacing*4.2, ohlc_y, 
                 f"({change_sign}{price_change_pct:.2f}%)", 
                 fontsize=10, color=change_color,
                 transform=fig.transFigure, va='top')
        
        plt.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.92)
        ax_main.set_title('')
        ax_main.spines['top'].set_visible(False)
        ax_main.spines['left'].set_visible(False)
        
        # Render chart
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, 
                    facecolor=bg_color, edgecolor='none',
                    bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        buf.seek(0)
        return buf, f"CoinGecko:{symbol_display}"
    except Exception as e:
        print(f"CoinGecko chart error: {e}")
        return None, None


# ==============================================================================
# VOLUME ANALYSIS
# ==============================================================================

async def get_buy_sell_vol(symbol: str, interval: str) -> str:
    """
    Analyze buy vs sell volume from Binance kline data.
    
    Binance provides "taker buy volume" in kline data, which represents
    aggressive buying. Total volume minus buy volume = sell volume.
    
    Args:
        symbol: Cryptocurrency symbol
        interval: Time interval ('15m', '1h', '3h', '4h', '24h')
        
    Returns:
        str: Formatted volume analysis message
    """
    # Map user intervals to Binance kline intervals
    tf_map = {'15m': '15m', '1h': '1h', '3h': '1h', '24h': '1d', '4h': '4h', '1d': '1d'}
    tf = tf_map.get(interval)
    limit = 3 if interval == '3h' else 1  # 3h = sum of 3x 1h candles
    
    pair = f"{symbol.upper()}USDT"
    url = f"{BINANCE_API_URL}/klines?symbol={pair}&interval={tf}&limit={limit}"
    
    klines = await fetch_json(url, timeout=10)
    if not klines:
        return f"😕 Không tìm thấy volume Binance cho {symbol.upper()}."
    
    try:
        # Sum up volume from all candles
        # Index 5: Total volume, Index 9: Taker buy volume
        total = sum(float(k[5]) for k in klines)
        buy = sum(float(k[9]) for k in klines)
        sell = total - buy
        
        if total == 0: 
            return "Volume 0."
        
        pct = (buy / total) * 100
        net = buy - sell
        state = "🟢 MUA > BÁN" if pct > 50 else "🔴 BÁN > MUA"
        
        return (
            f"📊 **Vol {symbol.upper()} ({interval})**\n"
            f"Nguồn: `Binance Spot`\n"
            f"----------------\n"
            f"**{state}**\n"
            f"🟢 Mua: `{format_price(buy)}` ({pct:.1f}%)\n"
            f"🔴 Bán: `{format_price(sell)}` ({100-pct:.1f}%)\n"
            f"⚖️ Ròng: `{format_price(net)}` {symbol.upper()}"
        )
    except:
        return "Lỗi phân tích volume."


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

async def get_current_price(symbol: str) -> tuple[float | None, str | None]:
    """
    Get current price quickly (optimized for /cal command).
    
    Uses lightweight price endpoint instead of full ticker for speed.
    
    Args:
        symbol: Cryptocurrency symbol
        
    Returns:
        tuple: (price, source) or (None, error message)
    """
    # Check cache first
    cache_key = f"price_{symbol.lower()}"
    if cache_key in price_cache:
        return price_cache[cache_key]
    
    # Try Binance simple price endpoint (fastest)
    for stable in ['USDT', 'USDC']:
        url = f"{BINANCE_API_URL}/ticker/price?symbol={symbol.upper()}{stable}"
        data = await fetch_json(url, timeout=3)
        if data and 'price' in data:
            price = float(data['price'])
            price_cache[cache_key] = (price, "Binance")
            return price, "Binance"
    
    # CCXT fallback
    for ex_id in ['gateio', 'mexc']:
        ex = None
        try:
            ex = getattr(ccxt, ex_id)({'timeout': 5000})
            for s in STABLES:
                try:
                    t = await ex.fetch_ticker(f"{symbol.upper()}/{s}")
                    if t and t.get('last'):
                        result = (t['last'], ex.name)
                        price_cache[cache_key] = result
                        return result
                except: 
                    continue
        except: 
            pass
        finally:
            if ex: 
                await ex.close()
    
    # CoinGecko fallback (slowest but most comprehensive)
    cid = await get_coingecko_id(symbol)
    if cid:
        data = await fetch_json(f"{COINGECKO_API_URL}/simple/price?ids={cid}&vs_currencies=usd")
        if data and data.get(cid, {}).get('usd'):
            result = (float(data[cid]['usd']), "CoinGecko")
            price_cache[cache_key] = result
            return result
    
    return None, "Không tìm thấy giá."


async def get_dex_data(address: str) -> str:
    """
    Get token data from DexScreener by contract address.
    
    Args:
        address: Token contract address (0x...)
        
    Returns:
        str: Formatted token info or "NOT_FOUND"
    """
    data = await fetch_json(f"{DEXSCREENER_API_URL}/search?q={address}", timeout=10)
    
    if not data or not data.get('pairs'):
        return "NOT_FOUND"
    
    p = data['pairs'][0]
    return (
        f"🪙 **{p['baseToken']['name']}** (DexScreener)\n"
        f"Price: {format_price(float(p['priceUsd']))}\n"
        f"Vol 24h: `${p.get('volume', {}).get('h24', 0):,.0f}`"
    )


async def get_volume_data(symbol: str, date_str: str = None) -> str:
    """
    Get volume data from CoinGecko.
    
    Two modes:
    1. With date: Get volume for specific date
    2. Without date: Get cumulative all-time volume
    
    Args:
        symbol: Cryptocurrency symbol
        date_str: Optional date in YYYYMMDD format
        
    Returns:
        str: Formatted volume message
    """
    cid = await get_coingecko_id(symbol)
    if not cid: 
        return "❌ Không tìm thấy coin."
    
    try:
        if date_str:
            # Get volume for specific date
            dt = datetime.strptime(date_str, '%Y%m%d').strftime('%d-%m-%Y')
            data = await fetch_json(f"{COINGECKO_API_URL}/coins/{cid}/history?date={dt}", timeout=10)
            if data:
                vol = data.get('market_data', {}).get('total_volume', {}).get('usd')
                return f"📊 **Vol {symbol.upper()} ngày {date_str}:** `${int(vol):,}`" if vol else "Không có dữ liệu."
        else:
            # Get all-time cumulative volume
            data = await fetch_json(f"{COINGECKO_API_URL}/coins/{cid}/market_chart?vs_currency=usd&days=max&interval=daily", timeout=10)
            if data:
                vols = data.get('total_volumes', [])
                total = sum(x[1] for x in vols)
                return f"📊 **Tổng Vol Tích Lũy {symbol.upper()}:** `${int(total):,}`"
    except: 
        pass
    return "Lỗi dữ liệu."


async def get_trending() -> str:
    """
    Get trending cryptocurrencies from CoinGecko.
    
    Returns:
        str: Formatted trending list
    """
    cache_key = "trending"
    if cache_key in market_cache:
        return market_cache[cache_key]
    
    data = await fetch_json(f"{COINGECKO_API_URL}/search/trending", timeout=10)
    if not data:
        return "Lỗi trending."
    
    msg = "🔥 **Trending:**\n"
    for i, c in enumerate(data.get('coins', [])[:7]):
        msg += f"{i+1}. {c['item']['symbol']}\n"
    
    market_cache[cache_key] = msg
    return msg


async def get_market(order: str, limit: int = 10) -> str:
    """
    Get top gainers or losers from CoinGecko.
    
    Args:
        order: 'gainers' or 'losers'
        limit: Number of results (default: 10)
        
    Returns:
        str: Formatted market data
    """
    cache_key = f"market_{order}"
    if cache_key in market_cache:
        return market_cache[cache_key]
    
    sort = 'price_change_percentage_24h_desc' if order == 'gainers' else 'price_change_percentage_24h_asc'
    url = f"{COINGECKO_API_URL}/coins/markets?vs_currency=usd&order={sort}&per_page={limit}"
    
    data = await fetch_json(url, timeout=10)
    if not data:
        return "Lỗi market data."
    
    msg = f"📊 **Top {order.title()}**\n"
    for c in data:
        msg += f"{c['symbol'].upper()}: {format_price(c['current_price'])} ({c['price_change_percentage_24h']:+.2f}%)\n"
    
    market_cache[cache_key] = msg
    return msg


# ==============================================================================
# TELEGRAM COMMAND HANDLERS
# ==============================================================================

async def start_cmd(update, context):
    """
    Handle /start command - Display help message.
    Requires '/' prefix: /start
    """
    msg = update.message or update.channel_post
    if not msg:
        return
    
    help_text = """
🤖 **CRYPTO BOT PRO**
══════════════════════

📌 **Lưu ý:** Tất cả lệnh phải bắt đầu bằng dấu `/`

📊 **XEM GIÁ & THÔNG TIN**
├ `/p btc` - Xem giá chi tiết
├ `/ath eth` - Xem ATH & thông tin
└ `$sol` - Xem giá nhanh

📈 **XEM CHART**
├ `/ch btc` - Chart BTC 1h (mặc định light)
├ `/ch btc 4h` - Chart 4 giờ
└ `/ch btc 1d dark` - Chart ngày, nền tối

📉 **PHÂN TÍCH VOLUME**
├ `/btc 1h` - Vol 1 giờ
├ `/eth 15m` - Vol 15 phút
└ `/sol 24h` - Vol 24 giờ

🧮 **TÍNH TOÁN**
├ `/cal btc 0.5` - Tính giá trị 0.5 BTC
└ `/val 100 * 2.5` - Máy tính

📋 **THỊ TRƯỜNG**
├ `/trending` - Coin trending
├ `/buy` - Top tăng giá
├ `/sell` - Top giảm giá
└ `/vol btc` - Volume tích lũy

🔍 **KHÁC**
└ `0x...` - Tra cứu contract

══════════════════════
⏱️ Timeframes: `15m, 1h, 3h, 4h, 24h`
🎨 Themes: `light` (mặc định), `dark`
"""
    await msg.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def chart_cmd(update, context):
    """
    Handle /ch command - Generate and send candlestick chart.
    
    Requires '/' prefix:
    - /ch btc           -> BTC 1h chart (light theme)
    - /ch btc 4h        -> BTC 4h chart (light theme)
    - /ch btc 1d dark   -> BTC 1d chart (dark theme)
    
    Strategy (fastest to slowest):
    1. Try chart-img.com API (TradingView screenshots) - ~1-2 seconds
    2. Try CCXT local generation - ~3-5 seconds
    3. Try CoinGecko local generation - ~3-5 seconds
    """
    msg = update.message or update.channel_post
    if not msg: 
        return
    
    txt = msg.text.strip()
    # Remove leading '/'
    if txt.startswith('/'):
        txt = txt[1:]
    
    parts = txt.split()
    if len(parts) < 2:
        return await msg.reply_text("⚠️ Thiếu symbol. Ví dụ: `/ch btc 4h`", parse_mode=ParseMode.MARKDOWN)
    
    sym = parts[1]                                              # Symbol to chart
    tf = parts[2] if len(parts) > 2 else '1h'                  # Timeframe (default: 1h)
    
    # Theme selection: default is light, use dark only if explicitly specified
    theme = 'dark' if 'dark' in [p.lower() for p in parts] else 'light'
    
    # Send "loading" message
    wait = await msg.reply_text(f"⏳ Đang tải chart {sym}...")
    
    buf, name = None, None
    
    # --- Strategy 1: Fast API-based chart (TradingView screenshot) ---
    # This is the fastest method (~1-2 seconds)
    buf, name = await get_chart_image_from_api(sym, tf, theme)
    
    # --- Strategy 2: Fallback to local CCXT generation ---
    if not buf:
        await context.bot.edit_message_text(
            chat_id=msg.chat.id, 
            message_id=wait.message_id, 
            text=f"⏳ Đang vẽ chart {sym}..."
        )
        buf, name = await generate_ccxt_chart(sym, tf, theme)
    
    # --- Strategy 3: Fallback to CoinGecko ---
    if not buf:
        cid = await get_coingecko_id(sym)
        if cid:
            buf, name = await generate_coingecko_chart(cid, tf, theme)
    
    # Delete loading message
    await context.bot.delete_message(msg.chat.id, wait.message_id)
    
    # Send chart or error message
    if buf:
        await msg.reply_photo(buf, caption=f"📊 {name} | {tf}")
    else:
        await msg.reply_text("❌ Không tìm thấy chart. Thử symbol khác.")


async def vol_analysis_handler(update, context):
    """
    Handle volume analysis command.
    
    Requires '/' prefix:
    - /btc 1h -> Analyze BTC buy/sell volume for past hour
    """
    msg = update.message or update.channel_post
    if not msg: 
        return
    
    txt = msg.text.strip()
    # Remove leading '/'
    if txt.startswith('/'):
        txt = txt[1:]
    
    # Parse command: <symbol> <interval>
    match = re.match(r'^([a-zA-Z0-9]{2,10})\s+(15m|1h|3h|4h|24h)$', txt, re.IGNORECASE)
    if not match: 
        return
    
    sym = match.group(1)
    tf = match.group(2).lower()
    
    wait = await msg.reply_text(f"⏳ Phân tích vol {sym} {tf}...")
    res = await get_buy_sell_vol(sym, tf)
    await context.bot.edit_message_text(
        chat_id=msg.chat.id, 
        message_id=wait.message_id, 
        text=res, 
        parse_mode=ParseMode.MARKDOWN
    )


async def ath_cmd(update, context):
    """
    Handle /ath command - Show all-time high info.
    """
    msg = update.message or update.channel_post
    if not msg: 
        return
    
    parts = msg.text.split()
    if len(parts) < 2: 
        return
    
    res = await get_token_report(parts[1])
    await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)


async def calculate_cmd(update, context):
    """
    Handle /cal command - Calculate token value.
    
    Requires '/' prefix:
    - /cal btc 0.5 -> Calculate value of 0.5 BTC in USD
    """
    msg = update.message or update.channel_post
    if not msg: 
        return
    
    txt = msg.text.strip()
    # Remove leading '/'
    if txt.startswith('/'):
        txt = txt[1:]
    
    parts = txt.split()
    if len(parts) < 3:
        await msg.reply_text("⚠️ Dùng: `/cal <ký hiệu> <số lượng>`\nVí dụ: `/cal btc 0.5`", parse_mode=ParseMode.MARKDOWN)
        return
    
    symbol = parts[1]
    try:
        amount = float(parts[2])
    except:
        await msg.reply_text("⚠️ Số lượng không hợp lệ!", parse_mode=ParseMode.MARKDOWN)
        return
    
    wait = await msg.reply_text(f"🔍 Tính toán {symbol}...")
    p, src = await get_current_price(symbol)
    
    if p:
        total = p * amount
        res = (
            f"💰 **Kết quả tính toán**\n"
            f"--------------------\n"
            f"💵 **Giá:** `{format_price(p)}` / {symbol.upper()}\n"
            f"🔢 **SL:** `{amount:g}`\n"
            f"--------------------\n"
            f"💎 **Tổng:** `{format_price(total)}`"
        )
    else:
        res = f"😕 Không tìm thấy giá {symbol.upper()}."
    
    await context.bot.edit_message_text(
        chat_id=msg.chat.id, 
        message_id=wait.message_id, 
        text=res, 
        parse_mode=ParseMode.MARKDOWN
    )


async def vol_cmd(update, context):
    """
    Handle /vol command - Get volume data.
    
    Requires '/' prefix:
    - /vol btc            -> Get all-time cumulative volume
    - /vol btc 20231225   -> Get volume for specific date
    """
    msg = update.message or update.channel_post
    if not msg:
        return
    
    txt = msg.text.strip()
    # Remove leading '/'
    if txt.startswith('/'):
        txt = txt[1:]
    
    parts = txt.split()
    if len(parts) < 2:
        await msg.reply_text("⚠️ Dùng: `/vol <symbol>` hoặc `/vol <symbol> <YYYYMMDD>`", parse_mode=ParseMode.MARKDOWN)
        return
    
    sym = parts[1]
    date_str = parts[2] if len(parts) > 2 else None
    
    wait = await msg.reply_text("⏳ Check vol...")
    res = await get_volume_data(sym, date_str)
    await context.bot.edit_message_text(
        chat_id=msg.chat.id, 
        message_id=wait.message_id, 
        text=res, 
        parse_mode=ParseMode.MARKDOWN
    )


async def trending_cmd(update, context):
    """Handle /trending command - Show trending coins."""
    res = await get_trending()
    await update.message.reply_text(res, parse_mode=ParseMode.MARKDOWN)


async def buy_cmd(update, context):
    """Handle /buy command - Show top gainers."""
    res = await get_market('gainers')
    await update.message.reply_text(res, parse_mode=ParseMode.MARKDOWN)


async def sell_cmd(update, context):
    """Handle /sell command - Show top losers."""
    res = await get_market('losers')
    await update.message.reply_text(res, parse_mode=ParseMode.MARKDOWN)


async def value_cmd(update, context):
    """
    Handle /val command - Evaluate math expression.
    
    Requires '/' prefix:
    - /val 100 * 2.5 + 50 -> 300
    """
    msg = update.message or update.channel_post
    if not msg:
        return
    
    txt = msg.text.strip()
    # Remove leading '/'
    if txt.startswith('/'):
        txt = txt[1:]
    
    expr = txt.split(maxsplit=1)[1] if len(txt.split()) > 1 else ""
    
    if not expr:
        await msg.reply_text("⚠️ Dùng: `/val <biểu thức>`\nVí dụ: `/val 100 * 2.5 + 50`", parse_mode=ParseMode.MARKDOWN)
        return
    
    res = safe_eval(expr)
    if res is not None:
        await msg.reply_text(f"🧮 Kết quả: `{res}`", parse_mode=ParseMode.MARKDOWN)
    else:
        await msg.reply_text("⚠️ Biểu thức không hợp lệ!", parse_mode=ParseMode.MARKDOWN)


# ==============================================================================
# MAIN MESSAGE HANDLER
# ==============================================================================

async def handle_msg(update, context):
    """
    Main message handler - Routes incoming messages to appropriate handlers.
    
    ALL COMMANDS REQUIRE '/' PREFIX:
    
    Supported formats:
    - /<coin> <timeframe>    -> Volume analysis (/btc 1h)
    - /ch <symbol>           -> Chart (/ch btc)
    - /start                 -> Help
    - /p <symbol>            -> Price report (/p btc)
    - /ath <symbol>          -> ATH info
    - /cal <coin> <amount>   -> Calculate value
    - /val <expression>      -> Math calculator
    - /trending              -> Trending coins
    - /buy                   -> Top gainers
    - /sell                  -> Top losers
    - /vol <coin>            -> Volume data
    - $symbol                -> Quick price lookup
    - 0x...                  -> Contract address lookup
    """
    msg = update.message or update.channel_post
    if not msg or not msg.text: 
        return
    
    txt = msg.text.strip()
    txt_lower = txt.lower()
    
    # Commands MUST start with '/' (except $ and 0x)
    if txt.startswith('/'):
        # Remove the leading '/' for parsing
        cmd_text = txt[1:]
        cmd_lower = cmd_text.lower()
        
        # Volume analysis: /<coin> <timeframe> (e.g., /btc 1h)
        if re.match(r'^[a-zA-Z0-9]{2,10}\s+(15m|1h|3h|4h|24h)$', cmd_text, re.IGNORECASE):
            return await vol_analysis_handler(update, context)

        # Chart command: /ch <symbol>
        if cmd_lower.startswith('ch '):
            return await chart_cmd(update, context)
        
        # Start/Help command: /start
        if cmd_lower == 'start' or cmd_lower.startswith('start '):
            return await start_cmd(update, context)
        
        # Price report command: /p <symbol>
        if cmd_lower.startswith('p '):
            parts = cmd_text.split()
            if len(parts) > 1:
                res = await get_token_report(parts[1])
                await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)
            return
        
        # ATH command: /ath <symbol>
        if cmd_lower.startswith('ath '):
            parts = cmd_text.split()
            if len(parts) > 1:
                res = await get_token_report(parts[1])
                await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)
            return
        
        # Calculator command: /cal <symbol> <amount>
        if cmd_lower.startswith('cal '): 
            return await calculate_cmd(update, context)
        
        # Value/Math expression command: /val <expression>
        if cmd_lower.startswith('val '): 
            return await value_cmd(update, context)
        
        # Trending command: /trending
        if cmd_lower == 'trending' or cmd_lower.startswith('trending '):
            return await trending_cmd(update, context)
        
        # Buy (top gainers) command: /buy
        if cmd_lower == 'buy' or cmd_lower.startswith('buy '):
            return await buy_cmd(update, context)
        
        # Sell (top losers) command: /sell
        if cmd_lower == 'sell' or cmd_lower.startswith('sell '):
            return await sell_cmd(update, context)
        
        # Volume command: /vol <symbol>
        if cmd_lower.startswith('vol '):
            return await vol_cmd(update, context)
    
    # Quick price lookup with $ prefix (e.g., $btc) - no / required
    if txt.startswith('$') and len(txt) > 1:
        sym = txt[1:]
        if VALID_SYMBOL_PATTERN.match(sym):
            res = await get_token_report(sym)
            await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)
        return

    # Contract address lookup (0x...) - no / required
    if is_contract_address(txt):
        res = await get_dex_data(txt)
        if res == "NOT_FOUND":
            res = "❌ Không tìm thấy contract."
        await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)


async def on_shutdown(app):
    """
    Cleanup function called when bot shuts down.
    Closes HTTP session to release resources.
    """
    await close_http_session()


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """
    Initialize and run the Telegram bot.
    
    Setup:
    1. Create Application with bot token
    2. Configure timeouts for reliability
    3. Register message handler
    4. Start polling for updates
    """
    print("Bot started...")
    
    # Build application with custom timeouts
    app = Application.builder().token(BOT_TOKEN).connect_timeout(30).read_timeout(60).build()
    
    # Register single handler for all text messages and commands
    # This centralized handler routes to specific functions based on message content
    app.add_handler(MessageHandler(filters.TEXT | filters.COMMAND, handle_msg))
    
    # Start the bot (blocking call)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
