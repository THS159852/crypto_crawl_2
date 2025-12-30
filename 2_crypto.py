"""
================================================================================
CRYPTO TELEGRAM BOT - Professional Cryptocurrency Price & Chart Bot
================================================================================
Version: 3.0 (Merged & Optimized)
Description: A high-performance Telegram bot for real-time cryptocurrency 
             price tracking, candlestick charts, and market analysis.
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import sys
import asyncio
import aiohttp
import io
import re
import os
import pandas as pd
import mplfinance as mpf
import ccxt.async_support as ccxt
from datetime import datetime, timedelta, timezone
from telegram import Update
from telegram.ext import Application, MessageHandler, filters
from telegram.constants import ParseMode
from dotenv import load_dotenv

# ==============================================================================
# CONFIGURATION & INITIALIZATION
# ==============================================================================

# UTF-8 Encoding Setup
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Load Environment Variables
load_dotenv()
BOT_TOKEN = os.environ.get('BOT_TOKEN')

if not BOT_TOKEN:
    print("ERROR: Missing BOT_TOKEN.")
    sys.exit(1)

# ==============================================================================
# API ENDPOINTS
# ==============================================================================
BINANCE_API_URL = "https://api.binance.com/api/v3"
BINANCE_F_API_URL = "https://fapi.binance.com/fapi/v1"
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
DEXSCREENER_API_URL = "https://api.dexscreener.com/latest/dex"

# ==============================================================================
# CONSTANTS
# ==============================================================================
PRIORITY_EXCHANGES = ['binance', 'bybit', 'okx', 'gateio', 'mexc', 'kucoin']
VALID_SYMBOL_PATTERN = re.compile(r'^[a-zA-Z0-9]{1,10}$')
STABLES = ['USDT', 'USDC', 'FDUSD', 'USD']

# ==============================================================================
# SIMPLE CACHING (without cachetools dependency)
# ==============================================================================
_cache = {}
_cache_times = {}


def get_cached(key: str, ttl: int = 60):
    """Get cached value if not expired."""
    if key in _cache and key in _cache_times:
        if (datetime.now() - _cache_times[key]).total_seconds() < ttl:
            return _cache[key]
    return None


def set_cached(key: str, value):
    """Set cached value with current timestamp."""
    _cache[key] = value
    _cache_times[key] = datetime.now()


# ==============================================================================
# HTTP SESSION MANAGEMENT
# ==============================================================================
_http_session: aiohttp.ClientSession | None = None


async def get_http_session() -> aiohttp.ClientSession:
    """Get or create a singleton HTTP session with connection pooling."""
    global _http_session
    
    if _http_session is None or _http_session.closed:
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        connector = aiohttp.TCPConnector(
            limit=50,
            limit_per_host=10,
            ttl_dns_cache=300
        )
        _http_session = aiohttp.ClientSession(timeout=timeout, connector=connector)
    
    return _http_session


async def close_http_session():
    """Close the global HTTP session gracefully."""
    global _http_session
    if _http_session and not _http_session.closed:
        await _http_session.close()
        _http_session = None


async def safe_close_exchange(exchange):
    """Safely close a CCXT exchange connection."""
    if exchange is not None:
        try:
            await exchange.close()
        except Exception:
            pass


class CCXTExchange:
    """Context manager for safe CCXT exchange usage."""
    
    def __init__(self, exchange_id: str, timeout: int = 5000):
        self.exchange_id = exchange_id
        self.timeout = timeout
        self.exchange = None
    
    async def __aenter__(self):
        try:
            self.exchange = getattr(ccxt, self.exchange_id)({'timeout': self.timeout})
            return self.exchange
        except Exception:
            return None
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await safe_close_exchange(self.exchange)
        return False  # Don't suppress exceptions


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def is_contract_address(address: str) -> bool:
    """Check if a string is a valid EVM contract address."""
    return address.startswith('0x') and len(address) == 42


def format_price(price: float) -> str:
    """Format a price value with appropriate decimal places."""
    if price is None:
        return "N/A"
    if abs(price) >= 10:
        return f"${price:,.2f}"
    if abs(price) >= 1:
        return f"${price:,.3f}"
    if abs(price) >= 0.001:
        return f"${price:,.4f}"
    return f"${price:,.8f}"


def safe_eval(expression: str) -> float | None:
    """Safely evaluate a mathematical expression string."""
    if not re.match(r'^[0-9\s\+\-\*\/\(\)\.]+$', expression):
        return None
    try:
        return eval(expression, {"__builtins__": {}}, {})
    except Exception:
        return None


# ==============================================================================
# ASYNC DATA FETCHING FUNCTIONS
# ==============================================================================

async def fetch_json(url: str, timeout: float = 5) -> dict | list | None:
    """Generic async JSON fetcher with error handling."""
    try:
        session = await get_http_session()
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception:
        pass
    return None


async def get_coingecko_id(symbol: str) -> str | None:
    """Get CoinGecko coin ID from ticker symbol with caching."""
    symbol_lower = symbol.lower()
    
    cached = get_cached(f"cgid_{symbol_lower}", ttl=3600)
    if cached:
        return cached
    
    data = await fetch_json(f"{COINGECKO_API_URL}/search?query={symbol}")
    if data:
        for c in data.get('coins', []):
            if c.get('symbol', '').lower() == symbol_lower:
                cg_id = c.get('id')
                set_cached(f"cgid_{symbol_lower}", cg_id)
                return cg_id
    return None


async def get_coingecko_info(symbol: str) -> dict | None:
    """Get detailed coin information from CoinGecko with caching."""
    cache_key = f"cg_info_{symbol.lower()}"
    
    cached = get_cached(cache_key, ttl=300)
    if cached:
        return cached
    
    cid = await get_coingecko_id(symbol)
    if not cid:
        return None
    
    data = await fetch_json(f"{COINGECKO_API_URL}/coins/{cid}", timeout=10)
    if not data:
        return None
    
    md = data.get('market_data', {})
    result = {
        'name': data.get('name', ''),
        'symbol': data.get('symbol', '').upper(),
        'rank': data.get('market_cap_rank', 0),
        'current_price': md.get('current_price', {}).get('usd', 0),
        'high_24h': md.get('high_24h', {}).get('usd', 0),
        'low_24h': md.get('low_24h', {}).get('usd', 0),
        'change_1h': md.get('price_change_percentage_1h_in_currency', {}).get('usd', 0),
        'change_24h': md.get('price_change_percentage_24h', 0),
        'change_7d': md.get('price_change_percentage_7d', 0),
        'change_30d': md.get('price_change_percentage_30d', 0),
        'ath': md.get('ath', {}).get('usd', 0),
        'ath_change': md.get('ath_change_percentage', {}).get('usd', 0),
        'ath_date': md.get('ath_date', {}).get('usd', '').split('T')[0] if md.get('ath_date', {}).get('usd') else '',
        'atl': md.get('atl', {}).get('usd', 0),
        'atl_change': md.get('atl_change_percentage', {}).get('usd', 0),
        'atl_date': md.get('atl_date', {}).get('usd', '').split('T')[0] if md.get('atl_date', {}).get('usd') else '',
        'cap': md.get('market_cap', {}).get('usd', 0),
        'volume_24h': md.get('total_volume', {}).get('usd', 0),
        'fdv': md.get('fully_diluted_valuation', {}).get('usd', 0),
        'circulating': md.get('circulating_supply', 0),
        'total_supply': md.get('total_supply', 0),
        'max_supply': md.get('max_supply', 0),
    }
    
    set_cached(cache_key, result)
    return result


async def get_binance_kline_change(trading_pair: str, interval: str, is_future: bool = False) -> float | None:
    """Calculate price change percentage from Binance kline data."""
    api_url = BINANCE_F_API_URL if is_future else BINANCE_API_URL
    url = f"{api_url}/klines?symbol={trading_pair}&interval={interval}&limit=2"
    
    data = await fetch_json(url)
    if not data or len(data) < 2:
        return None
    
    try:
        prev_close = float(data[0][4])
        curr_price = float(data[1][4])
        if prev_close == 0:
            return 0.0
        return ((curr_price - prev_close) / prev_close) * 100
    except Exception:
        return None


async def get_binance_price_data(symbol: str) -> tuple[dict | None, str, str]:
    """Fetch price data from Binance Spot and Futures APIs in parallel."""
    symbol_upper = symbol.upper()
    
    async def fetch_binance(is_future: bool, stable: str):
        api_url = BINANCE_F_API_URL if is_future else BINANCE_API_URL
        pair = f"{symbol_upper}{stable}"
        url = f"{api_url}/ticker/24hr?symbol={pair}"
        
        data = await fetch_json(url)
        if not data:
            return None
        
        try:
            price = float(data['lastPrice'])
            p24h = float(data['priceChangePercent'])
            
            kline_tasks = [
                get_binance_kline_change(pair, '1h', is_future),
                get_binance_kline_change(pair, '5m', is_future)
            ]
            p1h, p5m = await asyncio.gather(*kline_tasks)
            
            price_24h = price / (1 + p24h/100) if p24h != -100 else 0
            price_1h = price / (1 + p1h/100) if p1h else 0
            price_5m = price / (1 + p5m/100) if p5m else 0
            
            return {
                'price': price,
                'p5m': p5m, 'price_5m': price_5m,
                'p1h': p1h, 'price_1h': price_1h,
                'p24h': p24h, 'price_24h': price_24h,
                'vol': float(data['quoteVolume']),
                'vol_unit': stable,
                'source': "Binance Futures" if is_future else "Sàn Binance",
                'pair': f"{symbol_upper}/{stable}"
            }
        except Exception:
            return None
    
    tasks = []
    for is_future in [False, True]:
        for stable in ['USDT', 'USDC', 'FDUSD']:
            tasks.append(fetch_binance(is_future, stable))
    
    results = await asyncio.gather(*tasks)
    
    for r in results:
        if r:
            return r, r['source'], r['pair']
    
    return None, "", ""


async def get_ccxt_price(symbol: str) -> tuple[dict | None, str, str]:
    """Fetch price from CCXT-supported exchanges as fallback."""
    symbol_upper = symbol.upper()
    
    for ex_id in ['gateio', 'mexc', 'kucoin', 'okx', 'bybit']:
        async with CCXTExchange(ex_id, timeout=5000) as ex:
            if ex is None:
                continue
            
            for stable in STABLES:
                try:
                    t = await ex.fetch_ticker(f"{symbol_upper}/{stable}")
                    if t and t.get('last'):
                        price = t['last']
                        p24h = t.get('percentage', 0)
                        result = {
                            'price': price,
                            'p24h': p24h,
                            'price_24h': price / (1 + p24h/100) if p24h else 0,
                            'vol': t.get('quoteVolume', 0),
                            'vol_unit': 'USDT'
                        }
                        return result, f"Sàn {ex.name}", t['symbol']
                except Exception:
                    continue
    
    return None, "", ""


async def get_dexscreener_price(symbol: str) -> tuple[dict | None, str, str]:
    """Fetch price from DexScreener for DEX-traded tokens."""
    url = f"{DEXSCREENER_API_URL}/search?q={symbol}"
    data = await fetch_json(url)
    
    if not data or not data.get('pairs'):
        return None, "", ""
    
    p = max(data['pairs'], key=lambda x: x.get('liquidity', {}).get('usd', 0))
    
    try:
        price = float(p['priceUsd'])
        price_data = {
            'price': price,
            'p5m': p.get('priceChange', {}).get('m5'),
            'p1h': p.get('priceChange', {}).get('h1'),
            'p24h': p.get('priceChange', {}).get('h24'),
            'vol': p.get('volume', {}).get('h24', 0),
            'vol_unit': 'USD'
        }
        
        if price_data['p5m']:
            price_data['price_5m'] = price / (1 + price_data['p5m']/100)
        if price_data['p1h']:
            price_data['price_1h'] = price / (1 + price_data['p1h']/100)
        if price_data['p24h']:
            price_data['price_24h'] = price / (1 + price_data['p24h']/100)
        
        source = f"DexScreener ({p.get('dexId')})"
        pair_name = f"{p['baseToken']['symbol']}/{p['quoteToken']['symbol']}"
        return price_data, source, pair_name
    except Exception:
        return None, "", ""


# ==============================================================================
# PRICE PREDICTION
# ==============================================================================

def calculate_price_prediction(price_data: dict, cg_info: dict | None) -> dict | None:
    """Calculate price prediction based on technical indicators."""
    current_price = price_data.get('price', 0)
    if current_price <= 0:
        return None
    
    signals = {'bullish': 0, 'bearish': 0, 'neutral': 0}
    
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
    
    # 24-hour trend
    if p24h > 3:
        signals['bullish'] += 3
    elif p24h < -3:
        signals['bearish'] += 3
    elif p24h > 0:
        signals['bullish'] += 1
    elif p24h < 0:
        signals['bearish'] += 1
    
    # ATH/ATL analysis
    if cg_info:
        ath_change = cg_info.get('ath_change', 0) or 0
        
        if ath_change < -80:
            signals['bullish'] += 3
        elif ath_change < -50:
            signals['bullish'] += 2
        
        if ath_change > -10:
            signals['bearish'] += 2
        
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
    
    # Calculate prediction
    total_signals = signals['bullish'] + signals['bearish'] + signals['neutral']
    if total_signals == 0:
        total_signals = 1
    
    bullish_score = signals['bullish'] / total_signals * 100
    bearish_score = signals['bearish'] / total_signals * 100
    
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
    
    volatility = abs(p24h) if p24h else 5
    volatility = max(volatility, 2)
    volatility = min(volatility, 30)
    
    if trend == 'BULLISH':
        target_pct_high = volatility * 1.5
        target_pct_low = volatility * 0.3
    elif trend == 'BEARISH':
        target_pct_high = volatility * 0.3
        target_pct_low = volatility * 1.5
    else:
        target_pct_high = volatility * 0.8
        target_pct_low = volatility * 0.8
    
    target_price_high = current_price * (1 + target_pct_high / 100)
    target_price_low = current_price * (1 - target_pct_low / 100)
    
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
        'recommendation': recommendation,
    }


# ==============================================================================
# MAIN REPORT GENERATION
# ==============================================================================

async def get_token_report(symbol: str) -> str:
    """Generate a comprehensive price report for a cryptocurrency."""
    symbol_upper = symbol.upper()
    
    cache_key = f"report_{symbol_upper}"
    cached = get_cached(cache_key, ttl=10)
    if cached:
        return cached
    
    # Try Binance first
    price_data, source, pair_name = await get_binance_price_data(symbol)
    
    # Try CCXT exchanges
    if not price_data:
        price_data, source, pair_name = await get_ccxt_price(symbol)
    
    # Try DexScreener
    if not price_data:
        price_data, source, pair_name = await get_dexscreener_price(symbol)
    
    if not price_data:
        return f"😕 Không tìm thấy thông tin cho **{symbol_upper}**."
    
    # Fetch CoinGecko info
    cg_info = await get_coingecko_info(symbol)
    
    # Format output
    msg = f"🪙 **{pair_name}**\n"
    msg += f"📍 Nguồn: `{source}`\n"
    msg += "══════════════════════\n"
    msg += f"💰 **Giá hiện tại:** `{format_price(price_data['price'])}`\n"
    msg += "══════════════════════\n"
    msg += "📊 **BIẾN ĐỘNG GIÁ**\n"
    
    if price_data.get('p5m') is not None:
        icon = '🟢' if price_data['p5m'] >= 0 else '🔴'
        val_5m = format_price(price_data.get('price_5m', 0)) if price_data.get('price_5m') else 'N/A'
        msg += f"├ 5 phút:  {icon} `{price_data['p5m']:+.2f}%` ({val_5m})\n"
    
    if price_data.get('p1h') is not None:
        icon = '🟢' if price_data['p1h'] >= 0 else '🔴'
        val_1h = format_price(price_data.get('price_1h', 0)) if price_data.get('price_1h') else 'N/A'
        msg += f"├ 1 giờ:   {icon} `{price_data['p1h']:+.2f}%` ({val_1h})\n"
    
    if price_data.get('p24h') is not None:
        icon = '🟢' if price_data['p24h'] >= 0 else '🔴'
        val_24h = format_price(price_data.get('price_24h', 0)) if price_data.get('price_24h') else 'N/A'
        msg += f"└ 24 giờ:  {icon} `{price_data['p24h']:+.2f}%` ({val_24h})\n"
    
    msg += "──────────────────────\n"
    msg += f"📈 KL giao dịch 24h: `${price_data['vol']:,.0f}`\n"
    
    if cg_info:
        msg += "══════════════════════\n"
        msg += "📋 **THÔNG TIN THỊ TRƯỜNG**\n"
        
        if cg_info.get('rank'):
            msg += f"├ Xếp hạng: `#{cg_info['rank']}`\n"
        
        if cg_info.get('high_24h') and cg_info.get('low_24h'):
            msg += f"├ Cao 24h: `{format_price(cg_info['high_24h'])}`\n"
            msg += f"├ Thấp 24h: `{format_price(cg_info['low_24h'])}`\n"
        
        if cg_info.get('change_7d'):
            icon = '🟢' if cg_info['change_7d'] >= 0 else '🔴'
            msg += f"├ 7 ngày: {icon} `{cg_info['change_7d']:+.2f}%`\n"
        if cg_info.get('change_30d'):
            icon = '🟢' if cg_info['change_30d'] >= 0 else '🔴'
            msg += f"└ 30 ngày: {icon} `{cg_info['change_30d']:+.2f}%`\n"
        
        msg += "──────────────────────\n"
        msg += "🏆 **ALL-TIME HIGH (ATH)**\n"
        msg += f"├ Giá ATH: `{format_price(cg_info['ath'])}`\n"
        msg += f"├ Từ ATH: `{cg_info['ath_change']:+.2f}%`\n"
        msg += f"└ Ngày: `{cg_info['ath_date']}`\n"
        
        if cg_info.get('atl'):
            msg += "──────────────────────\n"
            msg += "📉 **ALL-TIME LOW (ATL)**\n"
            msg += f"├ Giá ATL: `{format_price(cg_info['atl'])}`\n"
            msg += f"├ Từ ATL: `{cg_info['atl_change']:+.2f}%`\n"
            msg += f"└ Ngày: `{cg_info['atl_date']}`\n"
        
        msg += "──────────────────────\n"
        msg += "💎 **VỐN HÓA & CUNG**\n"
        msg += f"├ Vốn hóa: `${cg_info['cap']:,.0f}`\n"
        
        if cg_info.get('fdv'):
            msg += f"├ FDV: `${cg_info['fdv']:,.0f}`\n"
        
        msg += f"├ Lưu thông: `{cg_info['circulating']:,.0f}`\n"
        
        total_supply = f"{cg_info['total_supply']:,.0f}" if cg_info.get('total_supply') else "∞"
        msg += f"├ Tổng cung: `{total_supply}`\n"
        
        max_supply = f"{cg_info['max_supply']:,.0f}" if cg_info.get('max_supply') else "∞"
        msg += f"└ Cung tối đa: `{max_supply}`\n"
    
    # Price prediction
    prediction = calculate_price_prediction(price_data, cg_info)
    
    if prediction:
        msg += "══════════════════════\n"
        msg += "🔮 **DỰ ĐOÁN XU HƯỚNG**\n"
        msg += "──────────────────────\n"
        msg += f"📊 Xu hướng: {prediction['trend_icon']} **{prediction['trend']}**\n"
        msg += f"📈 Độ tin cậy: `{prediction['confidence']:.1f}%`\n"
        msg += "──────────────────────\n"
        msg += f"├ Tín hiệu Tăng: `{prediction['bullish_score']:.1f}%`\n"
        msg += f"└ Tín hiệu Giảm: `{prediction['bearish_score']:.1f}%`\n"
        msg += "──────────────────────\n"
        msg += "🎯 **MỤC TIÊU GIÁ (24h)**\n"
        msg += f"🟢 Nếu TĂNG: `{format_price(prediction['target_high'])}` (+{prediction['target_pct_high']:.2f}%)\n"
        msg += f"🔴 Nếu GIẢM: `{format_price(prediction['target_low'])}` (-{prediction['target_pct_low']:.2f}%)\n"
        msg += "──────────────────────\n"
        msg += f"💡 **Gợi ý:** {prediction['recommendation']}\n"
    
    # Disclaimer
    msg += "══════════════════════\n"
    msg += "⚠️ **CẢNH BÁO:** Đây chỉ là dự đoán!\n"
    msg += "Chúng tôi không chịu trách nhiệm mất mát.\n"
    msg += "Hãy DYOR!\n"
    msg += "══════════════════════\n"
    msg += f"🕐 `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
    
    set_cached(cache_key, msg)
    return msg


# ==============================================================================
# CHART GENERATION
# ==============================================================================

async def get_chart_image_from_api(symbol: str, interval: str = '1h', theme: str = 'light') -> tuple[io.BytesIO | None, str | None]:
    """Get chart image from external chart image APIs."""
    symbol_upper = symbol.upper()
    
    tv_intervals = {
        '1m': '1', '5m': '5', '15m': '15', '30m': '30',
        '1h': '60', '2h': '120', '4h': '240',
        '1d': 'D', '1w': 'W', '1M': 'M'
    }
    tv_interval = tv_intervals.get(interval, '60')
    tv_theme = 'dark' if theme == 'dark' else 'light'
    
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
    except Exception:
        pass
    
    for exchange in ['BYBIT', 'OKX', 'COINBASE']:
        try:
            session = await get_http_session()
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
        except Exception:
            continue
    
    return None, None


async def generate_ccxt_chart(symbol: str, interval: str, theme: str = 'light') -> tuple[io.BytesIO | None, str | None]:
    """Generate a candlestick chart using CCXT exchange data."""
    print(f"Drawing chart {symbol} {interval}...")
    
    for ex_id in PRIORITY_EXCHANGES:
        async with CCXTExchange(ex_id, timeout=10000) as ex:
            if ex is None:
                continue
            
            try:
                await ex.load_markets()
                
                for s in STABLES:
                    pair = f"{symbol.upper()}/{s}"
                    if pair in ex.markets:
                        ohlcv = await ex.fetch_ohlcv(pair, timeframe=interval, limit=100)
                        if not ohlcv:
                            continue
                        
                        df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
                        df = df.set_index('Date')
                        
                        mc = mpf.make_marketcolors(
                            up='#0ECB81',
                            down='#F6465D',
                            inherit=True
                        )
                        
                        s_style = mpf.make_mpf_style(
                            base_mpf_style='yahoo' if theme == 'light' else 'nightclouds',
                            marketcolors=mc,
                            y_on_right=True
                        )
                        
                        buf = io.BytesIO()
                        mpf.plot(
                            df,
                            type='candle',
                            style=s_style,
                            title=f'\n{ex.name}: {pair} - {interval}',
                            volume=True,
                            savefig=dict(fname=buf, dpi=100, pad_inches=0.25)
                        )
                        buf.seek(0)
                        return buf, f"{ex.name}: {pair}"
            except Exception:
                continue
    
    return None, None


async def generate_coingecko_chart(coin_id: str, interval: str, theme: str = 'light') -> tuple[io.BytesIO | None, str | None]:
    """Generate a candlestick chart using CoinGecko OHLC data."""
    days = '1' if interval in ['5m', '15m', '1h', '4h'] else '90'
    url = f"{COINGECKO_API_URL}/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
    
    data = await fetch_json(url, timeout=10)
    if not data:
        return None, None
    
    try:
        df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df = df.set_index('Date')
        
        rule = interval.replace('m', 'T') if interval in ['5m', '15m'] else '1h' if interval in ['1h', '4h'] else '1D'
        if interval != '1d':
            df = df.resample(rule).agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last'
            }).dropna()
        
        mc = mpf.make_marketcolors(up='#0ECB81', down='#F6465D', inherit=True)
        s_style = mpf.make_mpf_style(
            base_mpf_style='yahoo' if theme == 'light' else 'nightclouds',
            marketcolors=mc,
            y_on_right=True
        )
        
        buf = io.BytesIO()
        mpf.plot(
            df,
            type='candle',
            style=s_style,
            title=f'\n{coin_id.upper()} - {interval} (CG)',
            volume=False,
            savefig=dict(fname=buf, dpi=100, pad_inches=0.25)
        )
        buf.seek(0)
        return buf, coin_id.upper()
    except Exception:
        return None, None


# ==============================================================================
# VOLUME ANALYSIS
# ==============================================================================

async def get_buy_sell_vol(symbol: str, interval: str) -> str:
    """Analyze buy vs sell volume from Binance kline data."""
    tf_map = {'15m': '15m', '1h': '1h', '3h': '1h', '24h': '1d', '4h': '4h', '1d': '1d'}
    tf = tf_map.get(interval)
    limit = 3 if interval == '3h' else 1
    
    pair = f"{symbol.upper()}USDT"
    url = f"{BINANCE_API_URL}/klines?symbol={pair}&interval={tf}&limit={limit}"
    
    klines = await fetch_json(url, timeout=10)
    if not klines:
        return f"😕 Không tìm thấy volume Binance cho {symbol.upper()}."
    
    try:
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
    except Exception:
        return "Lỗi phân tích volume."


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

async def get_current_price(symbol: str) -> tuple[float | None, str | None]:
    """Get current price quickly."""
    cache_key = f"price_{symbol.lower()}"
    cached = get_cached(cache_key, ttl=10)
    if cached:
        return cached
    
    for stable in ['USDT', 'USDC']:
        url = f"{BINANCE_API_URL}/ticker/price?symbol={symbol.upper()}{stable}"
        data = await fetch_json(url, timeout=3)
        if data and 'price' in data:
            price = float(data['price'])
            result = (price, "Binance")
            set_cached(cache_key, result)
            return result
    
    for ex_id in ['gateio', 'mexc']:
        async with CCXTExchange(ex_id, timeout=5000) as ex:
            if ex is None:
                continue
            
            for s in STABLES:
                try:
                    t = await ex.fetch_ticker(f"{symbol.upper()}/{s}")
                    if t and t.get('last'):
                        result = (t['last'], ex.name)
                        set_cached(cache_key, result)
                        return result
                except Exception:
                    continue
    
    cid = await get_coingecko_id(symbol)
    if cid:
        data = await fetch_json(f"{COINGECKO_API_URL}/simple/price?ids={cid}&vs_currencies=usd")
        if data and data.get(cid, {}).get('usd'):
            result = (float(data[cid]['usd']), "CoinGecko")
            set_cached(cache_key, result)
            return result
    
    return None, "Không tìm thấy giá."


async def get_dex_data(address: str) -> str:
    """Get token data from DexScreener by contract address."""
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
    """Get volume data from CoinGecko."""
    cid = await get_coingecko_id(symbol)
    if not cid:
        return "❌ Không tìm thấy coin."
    
    try:
        if date_str:
            dt = datetime.strptime(date_str, '%Y%m%d').strftime('%d-%m-%Y')
            data = await fetch_json(f"{COINGECKO_API_URL}/coins/{cid}/history?date={dt}", timeout=10)
            if data:
                vol = data.get('market_data', {}).get('total_volume', {}).get('usd')
                return f"📊 **Vol {symbol.upper()} ngày {date_str}:** `${int(vol):,}`" if vol else "Không có dữ liệu."
        else:
            data = await fetch_json(f"{COINGECKO_API_URL}/coins/{cid}/market_chart?vs_currency=usd&days=max&interval=daily", timeout=10)
            if data:
                vols = data.get('total_volumes', [])
                total = sum(x[1] for x in vols)
                return f"📊 **Tổng Vol Tích Lũy {symbol.upper()}:** `${int(total):,}`"
    except Exception:
        pass
    return "Lỗi dữ liệu."


async def get_trending() -> str:
    """Get trending cryptocurrencies from CoinGecko."""
    cached = get_cached("trending", ttl=60)
    if cached:
        return cached
    
    data = await fetch_json(f"{COINGECKO_API_URL}/search/trending", timeout=10)
    if not data:
        return "Lỗi trending."
    
    msg = "🔥 **Trending:**\n"
    for i, c in enumerate(data.get('coins', [])[:7]):
        msg += f"{i+1}. {c['item']['symbol']}\n"
    
    set_cached("trending", msg)
    return msg


async def get_market(order: str, limit: int = 10) -> str:
    """Get top gainers or losers from CoinGecko."""
    cache_key = f"market_{order}"
    cached = get_cached(cache_key, ttl=60)
    if cached:
        return cached
    
    sort = 'price_change_percentage_24h_desc' if order == 'gainers' else 'price_change_percentage_24h_asc'
    url = f"{COINGECKO_API_URL}/coins/markets?vs_currency=usd&order={sort}&per_page={limit}"
    
    data = await fetch_json(url, timeout=10)
    if not data:
        return "Lỗi market data."
    
    msg = f"📊 **Top {order.title()}**\n"
    for c in data:
        change = c.get('price_change_percentage_24h', 0) or 0
        msg += f"{c['symbol'].upper()}: {format_price(c['current_price'])} ({change:+.2f}%)\n"
    
    set_cached(cache_key, msg)
    return msg


# ==============================================================================
# BINANCE ALPHA COMPETITION
# ==============================================================================

# Múi giờ Việt Nam UTC+7
VN_TIMEZONE = timezone(timedelta(hours=7))

# Binance Alpha API endpoint (internal API)
BINANCE_ALPHA_API = "https://www.binance.com/bapi/alpha/v1"

# Danh sách token Alpha đang active trên Binance (cập nhật thủ công)
# Format: symbol -> {reward_tokens, min_vol, total_vol, vol_change, end_time, status}
# reward_tokens: Số lượng token reward (sẽ tính giá trị USD realtime)
# min_vol: Volume tối thiểu để đủ điều kiện ($)
# total_vol: Tổng volume hiện tại của competition ($)
# vol_change: Thay đổi volume ($)
# end_time: Thời gian kết thúc (UTC) format "YYYY-MM-DD HH:MM"
ACTIVE_ALPHA_TOKENS = {
    'US': {
        'name': 'US Token',
        'reward_tokens': 1000000,  # Số lượng token reward
        'min_vol': 500,
        'total_vol': 5000000,
        'vol_change': 1200000,
        'end_time': '2025-01-05 23:59',  # UTC
        'status': 'active'
    },
    'STAR': {
        'name': 'StarHeroes',
        'reward_tokens': 50000,
        'min_vol': 0,
        'total_vol': 9800000,
        'vol_change': 6300000,
        'end_time': '2025-01-03 23:59',
        'status': 'active'
    },
    'KGEN': {
        'name': 'KGEN',
        'reward_tokens': 100000,
        'min_vol': 98882,
        'total_vol': 552900000,
        'vol_change': 59400000,
        'end_time': '2025-01-07 23:59',
        'status': 'active'
    },
    'RAVE': {
        'name': 'RAVE',
        'reward_tokens': 80000,
        'min_vol': 1526,
        'total_vol': 43400000,
        'vol_change': -9600000,
        'end_time': '2025-01-04 23:59',
        'status': 'active'
    },
    'ZKP': {
        'name': 'ZKP Token',
        'reward_tokens': 114486125,
        'min_vol': 489,
        'total_vol': 39400000,
        'vol_change': -6300000,
        'end_time': '2025-01-06 23:59',
        'status': 'active'
    },
    'CYS': {
        'name': 'Cysic',
        'reward_tokens': 500000,
        'min_vol': 1000,
        'total_vol': 25000000,
        'vol_change': 3500000,
        'end_time': '2025-01-08 23:59',
        'status': 'active'
    },
}


def get_vn_time() -> datetime:
    """Lấy thời gian hiện tại theo múi giờ Việt Nam (UTC+7)."""
    return datetime.now(VN_TIMEZONE)


def update_active_alpha_tokens(tokens: dict):
    """Cập nhật danh sách token Alpha đang active."""
    global ACTIVE_ALPHA_TOKENS
    ACTIVE_ALPHA_TOKENS.update(tokens)


async def get_token_price_from_sources(symbol: str) -> dict | None:
    """
    Lấy giá token từ nhiều nguồn: Binance, CCXT exchanges, DexScreener, CoinGecko.
    Trả về dict với price, volume, change hoặc None nếu không tìm thấy.
    """
    symbol_upper = symbol.upper()
    
    # 1. Thử Binance Spot
    for stable in ['USDT', 'USDC', 'FDUSD']:
        url = f"{BINANCE_API_URL}/ticker/24hr?symbol={symbol_upper}{stable}"
        data = await fetch_json(url, timeout=5)
        if data and 'lastPrice' in data:
            return {
                'price': float(data.get('lastPrice', 0)),
                'volume_24h': float(data.get('quoteVolume', 0)),
                'price_change_24h': float(data.get('priceChangePercent', 0)),
                'high_24h': float(data.get('highPrice', 0)),
                'low_24h': float(data.get('lowPrice', 0)),
                'source': 'Binance Spot',
                'pair': f"{symbol_upper}/{stable}"
            }
    
    # 2. Thử Binance Futures
    for stable in ['USDT']:
        url = f"{BINANCE_F_API_URL}/ticker/24hr?symbol={symbol_upper}{stable}"
        data = await fetch_json(url, timeout=5)
        if data and 'lastPrice' in data:
            return {
                'price': float(data.get('lastPrice', 0)),
                'volume_24h': float(data.get('quoteVolume', 0)),
                'price_change_24h': float(data.get('priceChangePercent', 0)),
                'high_24h': float(data.get('highPrice', 0)),
                'low_24h': float(data.get('lowPrice', 0)),
                'source': 'Binance Futures',
                'pair': f"{symbol_upper}/{stable}"
            }
    
    # 3. Thử CCXT exchanges (Gate.io, MEXC, KuCoin)
    for ex_id in ['gateio', 'mexc', 'kucoin']:
        async with CCXTExchange(ex_id, timeout=5000) as ex:
            if ex is None:
                continue
            
            for stable in STABLES:
                try:
                    ticker = await ex.fetch_ticker(f"{symbol_upper}/{stable}")
                    if ticker and ticker.get('last'):
                        return {
                            'price': ticker['last'],
                            'volume_24h': ticker.get('quoteVolume', 0) or 0,
                            'price_change_24h': ticker.get('percentage', 0) or 0,
                            'high_24h': ticker.get('high', 0) or 0,
                            'low_24h': ticker.get('low', 0) or 0,
                            'source': ex.name,
                            'pair': f"{symbol_upper}/{stable}"
                        }
                except Exception:
                    continue
    
    # 4. Thử DexScreener
    try:
        url = f"{DEXSCREENER_API_URL}/search?q={symbol_upper}"
        data = await fetch_json(url, timeout=5)
        if data and data.get('pairs'):
            # Lọc pair có symbol khớp
            pairs = [p for p in data['pairs'] if p.get('baseToken', {}).get('symbol', '').upper() == symbol_upper]
            if pairs:
                # Lấy pair có liquidity cao nhất
                best_pair = max(pairs, key=lambda x: x.get('liquidity', {}).get('usd', 0))
                return {
                    'price': float(best_pair.get('priceUsd', 0)),
                    'volume_24h': best_pair.get('volume', {}).get('h24', 0) or 0,
                    'price_change_24h': best_pair.get('priceChange', {}).get('h24', 0) or 0,
                    'high_24h': 0,
                    'low_24h': 0,
                    'source': f"DEX ({best_pair.get('dexId', 'unknown')})",
                    'pair': f"{best_pair['baseToken']['symbol']}/{best_pair['quoteToken']['symbol']}",
                    'chain': best_pair.get('chainId', 'unknown')
                }
    except Exception:
        pass
    
    # 5. Thử CoinGecko
    try:
        cg_id = await get_coingecko_id(symbol)
        if cg_id:
            url = f"{COINGECKO_API_URL}/simple/price?ids={cg_id}&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true"
            data = await fetch_json(url, timeout=5)
            if data and data.get(cg_id):
                return {
                    'price': float(data[cg_id].get('usd', 0)),
                    'volume_24h': data[cg_id].get('usd_24h_vol', 0) or 0,
                    'price_change_24h': data[cg_id].get('usd_24h_change', 0) or 0,
                    'high_24h': 0,
                    'low_24h': 0,
                    'source': 'CoinGecko',
                    'pair': f"{symbol_upper}/USD"
                }
    except Exception:
        pass
    
    return None


def parse_end_time(end_time_str: str) -> datetime | None:
    """Parse end time string to datetime object (UTC)."""
    try:
        dt = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M')
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def format_end_time_vn(end_time_str: str) -> str:
    """Format end time to Vietnam timezone string."""
    dt = parse_end_time(end_time_str)
    if not dt:
        return "N/A"
    # Convert to VN timezone
    dt_vn = dt.astimezone(VN_TIMEZONE)
    return dt_vn.strftime('%d/%m/%Y %H:%M')


def get_time_remaining(end_time_str: str) -> str:
    """Calculate remaining time until end."""
    dt = parse_end_time(end_time_str)
    if not dt:
        return "N/A"
    
    now = datetime.now(timezone.utc)
    diff = dt - now
    
    if diff.total_seconds() <= 0:
        return "Đã kết thúc"
    
    days = diff.days
    hours = diff.seconds // 3600
    minutes = (diff.seconds % 3600) // 60
    
    if days > 0:
        return f"{days}d {hours}h"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


async def get_binance_alpha_tokens() -> list:
    """
    Lấy danh sách token từ Binance Alpha Competition.
    Lấy giá realtime từ nhiều nguồn và tính giá trị reward.
    """
    competitions = []
    
    # Thử lấy từ Binance Alpha API (internal) - có thể không hoạt động
    try:
        alpha_url = f"{BINANCE_ALPHA_API}/public/alpha/queryAlphaList"
        data = await fetch_json(alpha_url, timeout=10)
        
        if data and data.get('data'):
            for token_data in data['data']:
                symbol = token_data.get('tokenSymbol', '').upper()
                if symbol:
                    # Lấy giá realtime
                    price_info = await get_token_price_from_sources(symbol)
                    price = price_info['price'] if price_info else 0
                    reward_tokens = token_data.get('rewardTokens', 0)
                    reward_usd = reward_tokens * price if price > 0 else 0
                    
                    competitions.append({
                        'symbol': symbol,
                        'name': token_data.get('tokenName', symbol),
                        'price': price,
                        'volume_24h': price_info.get('volume_24h', 0) if price_info else 0,
                        'price_change_24h': price_info.get('price_change_24h', 0) if price_info else 0,
                        'reward_tokens': reward_tokens,
                        'reward_usd': reward_usd,
                        'min_vol': token_data.get('minVol', 0),
                        'total_vol': token_data.get('totalVol', 0),
                        'vol_change': 0,
                        'end_time': token_data.get('endTime', ''),
                        'end_time_vn': '',
                        'time_remaining': '',
                        'status': 'active',
                        'source': price_info.get('source', 'N/A') if price_info else 'N/A'
                    })
            if competitions:
                return competitions
    except Exception:
        pass
    
    # Fallback: Sử dụng danh sách token đã lưu và lấy giá từ nhiều nguồn
    for symbol, info in ACTIVE_ALPHA_TOKENS.items():
        if info.get('status') != 'active':
            continue
        
        try:
            # Lấy giá từ nhiều nguồn
            price_info = await get_token_price_from_sources(symbol)
            
            price = price_info['price'] if price_info else 0
            reward_tokens = info.get('reward_tokens', 0)
            reward_usd = reward_tokens * price if price > 0 else 0
            
            end_time = info.get('end_time', '')
            end_time_vn = format_end_time_vn(end_time) if end_time else 'N/A'
            time_remaining = get_time_remaining(end_time) if end_time else 'N/A'
            
            competitions.append({
                'symbol': symbol,
                'name': info.get('name', symbol),
                'price': price,
                'volume_24h': price_info.get('volume_24h', 0) if price_info else 0,
                'price_change_24h': price_info.get('price_change_24h', 0) if price_info else 0,
                'high_24h': price_info.get('high_24h', 0) if price_info else 0,
                'low_24h': price_info.get('low_24h', 0) if price_info else 0,
                'reward_tokens': reward_tokens,
                'reward_usd': reward_usd,
                'min_vol': info.get('min_vol', 0),
                'total_vol': info.get('total_vol', 0),
                'vol_change': info.get('vol_change', 0),
                'end_time': end_time,
                'end_time_vn': end_time_vn,
                'time_remaining': time_remaining,
                'status': 'active',
                'source': price_info.get('source', 'N/A') if price_info else 'N/A',
                'pair': price_info.get('pair', '') if price_info else ''
            })
        except Exception:
            # Thêm token với info cơ bản nếu có lỗi
            end_time = info.get('end_time', '')
            competitions.append({
                'symbol': symbol,
                'name': info.get('name', symbol),
                'price': 0,
                'volume_24h': 0,
                'price_change_24h': 0,
                'reward_tokens': info.get('reward_tokens', 0),
                'reward_usd': 0,
                'min_vol': info.get('min_vol', 0),
                'total_vol': info.get('total_vol', 0),
                'vol_change': info.get('vol_change', 0),
                'end_time': end_time,
                'end_time_vn': format_end_time_vn(end_time) if end_time else 'N/A',
                'time_remaining': get_time_remaining(end_time) if end_time else 'N/A',
                'status': 'active',
                'source': 'N/A'
            })
    
    return competitions


def format_number(value: float, symbol: str = "$") -> str:
    """Format số lớn thành dạng đọc được (K, M, B)."""
    if value >= 1000000000:
        return f"{symbol}{value/1000000000:,.2f}B"
    elif value >= 1000000:
        return f"{symbol}{value/1000000:,.1f}M"
    elif value >= 1000:
        return f"{symbol}{value/1000:,.1f}K"
    else:
        return f"{symbol}{value:,.0f}"


def format_token_amount(amount: float) -> str:
    """Format số lượng token."""
    if amount >= 1000000000:
        return f"{amount/1000000000:,.2f}B"
    elif amount >= 1000000:
        return f"{amount/1000000:,.2f}M"
    elif amount >= 1000:
        return f"{amount/1000:,.1f}K"
    elif amount >= 1:
        return f"{amount:,.2f}"
    else:
        return f"{amount:,.6f}"


async def get_alpha_competition_report() -> str:
    """Tạo báo cáo Alpha Competition - lấy từ nhiều nguồn."""
    cache_key = "alpha_competition"
    cached = get_cached(cache_key, ttl=60)  # Cache 1 phút
    if cached:
        return cached
    
    # Lấy thời gian Việt Nam
    vn_now = get_vn_time()
    
    # Lấy dữ liệu từ nhiều nguồn
    competitions = await get_binance_alpha_tokens()
    
    if not competitions:
        msg = f"""
🏆 **DAILY ALPHA COMPETITION**
📅 **Date:** {vn_now.strftime('%Y-%m-%d')}
══════════════════════════════

⚠️ **Không có competition nào đang chạy**

📋 **Để cập nhật danh sách:**
Liên hệ admin hoặc kiểm tra:
🔗 https://www.binance.com/en/alpha

══════════════════════════════
🕐 `{vn_now.strftime('%H:%M:%S')} (UTC+7)`
"""
        return msg
    
    # Sắp xếp theo reward_usd (giá trị cao nhất trước)
    competitions.sort(key=lambda x: x.get('reward_usd', 0), reverse=True)
    
    msg = f"""
🏆 **DAILY ALPHA COMPETITION**
📅 **Date:** {vn_now.strftime('%Y-%m-%d')}
══════════════════════════════
"""
    
    for comp in competitions[:10]:
        symbol = comp.get('symbol', 'N/A')
        name = comp.get('name', symbol)
        price = comp.get('price', 0)
        reward_usd = comp.get('reward_usd', 0)
        reward_tokens = comp.get('reward_tokens', 0)
        min_vol = comp.get('min_vol', 0)
        total_vol = comp.get('total_vol', 0)
        vol_change = comp.get('vol_change', 0)
        price_change_24h = comp.get('price_change_24h', 0) or 0
        end_time_vn = comp.get('end_time_vn', 'N/A')
        time_remaining = comp.get('time_remaining', 'N/A')
        source = comp.get('source', 'N/A')
        
        # Icon thay đổi giá
        price_icon = '📈' if price_change_24h >= 0 else '📉'
        vol_icon = '📈' if vol_change >= 0 else '📉'
        
        # Format các giá trị
        price_str = format_price(price) if price > 0 else 'Đang tải...'
        reward_usd_str = format_number(reward_usd, "$") if reward_usd > 0 else 'Đang tính...'
        reward_tokens_str = format_token_amount(reward_tokens) if reward_tokens > 0 else 'N/A'
        min_vol_str = format_number(min_vol, "$")
        total_vol_str = format_number(total_vol, "$") if total_vol > 0 else 'N/A'
        vol_change_str = format_number(abs(vol_change), "+$" if vol_change >= 0 else "-$")
        
        msg += f"""
{'─' * 30}
🏆 **${symbol}** ({name})
├ 💰 Giá: `{price_str}` {price_icon} `{price_change_24h:+.1f}%`
├ 🎁 Reward: `{reward_tokens_str} {symbol}`
├ 💵 Giá trị: `{reward_usd_str}`
├ 📊 Min Vol: `{min_vol_str}` {vol_icon}
├ 📈 Total Vol: `{total_vol_str}` ({vol_change_str})
├ ⏰ Kết thúc: `{end_time_vn}` (UTC+7)
├ ⏳ Còn lại: `{time_remaining}`
└ 📍 Nguồn: `{source}`
"""
    
    msg += f"""
══════════════════════════════
💡 Gõ `/p <symbol>` để xem chi tiết
🔗 https://www.binance.com/en/alpha
🕐 `{vn_now.strftime('%H:%M:%S')} (UTC+7)`
"""
    
    set_cached(cache_key, msg)
    return msg


async def get_alpha_daily_report(date_str: str = None) -> str:
    """
    Tạo báo cáo Alpha hàng ngày với thông tin reward.
    Lấy dữ liệu từ nhiều nguồn.
    """
    # Lấy thời gian Việt Nam
    vn_now = get_vn_time()
    
    if not date_str:
        date_str = vn_now.strftime('%Y-%m-%d')
    
    cache_key = f"alpha_daily_{date_str}"
    cached = get_cached(cache_key, ttl=120)  # Cache 2 phút
    if cached:
        return cached
    
    # Lấy dữ liệu token từ nhiều nguồn
    competitions = await get_binance_alpha_tokens()
    
    msg = f"""
🏆 **DAILY ALPHA COMPETITION**
📅 **Date:** {date_str}
══════════════════════════════
"""
    
    if not competitions:
        msg += f"""
⚠️ Không có Alpha competition đang chạy.

📝 **Để cập nhật dữ liệu:**
Liên hệ admin hoặc kiểm tra:
🔗 https://www.binance.com/en/alpha
══════════════════════════════
🕐 `{vn_now.strftime('%H:%M:%S')} (UTC+7)`
"""
        return msg
    
    # Sắp xếp theo thời gian kết thúc (sớm nhất trước)
    def sort_by_end_time(comp):
        end_time = comp.get('end_time', '')
        if end_time:
            dt = parse_end_time(end_time)
            if dt:
                return dt
        return datetime.max.replace(tzinfo=timezone.utc)
    
    competitions.sort(key=sort_by_end_time)
    
    for comp in competitions[:10]:
        symbol = comp.get('symbol', 'N/A')
        name = comp.get('name', symbol)
        price = comp.get('price', 0)
        reward_usd = comp.get('reward_usd', 0)
        reward_tokens = comp.get('reward_tokens', 0)
        min_vol = comp.get('min_vol', 0)
        total_vol = comp.get('total_vol', 0)
        vol_change = comp.get('vol_change', 0)
        price_change_24h = comp.get('price_change_24h', 0) or 0
        end_time_vn = comp.get('end_time_vn', 'N/A')
        time_remaining = comp.get('time_remaining', 'N/A')
        source = comp.get('source', 'N/A')
        
        # Icon thay đổi
        price_icon = '📈' if price_change_24h >= 0 else '📉'
        vol_icon = '📈' if vol_change >= 0 else '📉'
        
        # Format các giá trị
        price_str = format_price(price) if price > 0 else 'Đang tải...'
        reward_usd_str = format_number(reward_usd, "$") if reward_usd > 0 else 'Đang tính...'
        reward_tokens_str = format_token_amount(reward_tokens) if reward_tokens > 0 else 'N/A'
        min_vol_str = format_number(min_vol, "$")
        total_vol_str = format_number(total_vol, "$") if total_vol > 0 else 'N/A'
        vol_change_str = format_number(abs(vol_change), "+$" if vol_change >= 0 else "-$")
        
        msg += f"""
{'─' * 30}
🏆 **${symbol}** ({name})
├ 💰 Giá: `{price_str}` {price_icon} `{price_change_24h:+.1f}%`
├ 🎁 Reward: `{reward_tokens_str} {symbol}`
├ 💵 Giá trị: `{reward_usd_str}`
├ 📊 Min Vol: `{min_vol_str}` {vol_icon}
├ 📈 Total Vol: `{total_vol_str}` ({vol_change_str})
├ ⏰ Kết thúc: `{end_time_vn}` (UTC+7)
└ ⏳ Còn lại: `{time_remaining}`
"""
    
    msg += f"""
══════════════════════════════
🔗 https://www.binance.com/en/alpha
🕐 `{vn_now.strftime('%Y-%m-%d %H:%M:%S')} (UTC+7)`
"""
    
    set_cached(cache_key, msg)
    return msg


# ==============================================================================
# TELEGRAM COMMAND HANDLERS
# ==============================================================================

async def start_cmd(update, context):
    """Handle start command - Display help message."""
    msg = update.message or update.channel_post
    if not msg:
        return
    
    help_text = """
🤖 **CRYPTO BOT PRO v3.0**
══════════════════════

📌 **Lưu ý:** Tất cả lệnh đều hoạt động CÓ hoặc KHÔNG có dấu `/`

📊 **XEM GIÁ & THÔNG TIN**
├ `p btc` hoặc `/p btc` - Xem giá chi tiết
├ `ath eth` - Xem ATH & thông tin
└ `$sol` - Xem giá nhanh

📈 **XEM CHART**
├ `ch btc` - Chart BTC 1h (mặc định light)
├ `ch btc 4h` - Chart 4 giờ
└ `ch btc 1d dark` - Chart ngày, nền tối

📉 **PHÂN TÍCH VOLUME**
├ `btc 1h` hoặc `/btc 1h` - Vol 1 giờ
├ `eth 15m` - Vol 15 phút
└ `sol 24h` - Vol 24 giờ

🧮 **TÍNH TOÁN**
├ `cal btc 0.5` - Tính giá trị 0.5 BTC
└ `val 100 * 2.5` - Máy tính

📋 **THỊ TRƯỜNG**
├ `trending` - Coin trending
├ `buy` - Top tăng giá
├ `sell` - Top giảm giá
├ `vol btc` - Volume tích lũy
├ `compe` - Alpha Competition list
└ `alpha` - Alpha Daily Report

🔍 **KHÁC**
└ `0x...` - Tra cứu contract

══════════════════════
⏱️ Timeframes: `15m, 1h, 3h, 4h, 24h`
🎨 Themes: `light` (mặc định), `dark`
"""
    await msg.reply_text(help_text, parse_mode=ParseMode.MARKDOWN)


async def chart_cmd(update, context):
    """Handle chart command - Generate and send candlestick chart."""
    msg = update.message or update.channel_post
    if not msg:
        return
    
    txt = msg.text.strip()
    if txt.startswith('/'):
        txt = txt[1:]
    
    parts = txt.split()
    if len(parts) < 2:
        return await msg.reply_text("⚠️ Thiếu symbol. Ví dụ: `ch btc` hoặc `/ch btc 4h`", parse_mode=ParseMode.MARKDOWN)
    
    sym = parts[1]
    tf = parts[2] if len(parts) > 2 else '1h'
    theme = 'dark' if 'dark' in [p.lower() for p in parts] else 'light'
    
    wait = await msg.reply_text(f"⏳ Đang tải chart {sym}...")
    
    buf, name = None, None
    
    # Try fast API-based chart first
    buf, name = await get_chart_image_from_api(sym, tf, theme)
    
    # Fallback to local CCXT generation
    if not buf:
        try:
            await context.bot.edit_message_text(
                chat_id=msg.chat.id,
                message_id=wait.message_id,
                text=f"⏳ Đang vẽ chart {sym}..."
            )
        except Exception:
            pass
        buf, name = await generate_ccxt_chart(sym, tf, theme)
    
    # Fallback to CoinGecko
    if not buf:
        cid = await get_coingecko_id(sym)
        if cid:
            buf, name = await generate_coingecko_chart(cid, tf, theme)
    
    try:
        await context.bot.delete_message(msg.chat.id, wait.message_id)
    except Exception:
        pass
    
    if buf:
        await msg.reply_photo(buf, caption=f"📊 {name} | {tf}")
    else:
        await msg.reply_text("❌ Không tìm thấy chart. Thử symbol khác.")


async def vol_analysis_handler(update, context):
    """Handle volume analysis command."""
    msg = update.message or update.channel_post
    if not msg:
        return
    
    txt = msg.text.strip()
    if txt.startswith('/'):
        txt = txt[1:]
    
    match = re.match(r'^([a-zA-Z0-9]{2,10})\s+(15m|1h|3h|4h|24h)$', txt, re.IGNORECASE)
    if not match:
        return
    
    sym = match.group(1)
    tf = match.group(2).lower()
    
    wait = await msg.reply_text(f"⏳ Phân tích vol {sym} {tf}...")
    res = await get_buy_sell_vol(sym, tf)
    try:
        await context.bot.edit_message_text(
            chat_id=msg.chat.id,
            message_id=wait.message_id,
            text=res,
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception:
        await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)


async def calculate_cmd(update, context):
    """Handle calculate command - Calculate token value."""
    msg = update.message or update.channel_post
    if not msg:
        return
    
    txt = msg.text.strip()
    if txt.startswith('/'):
        txt = txt[1:]
    
    parts = txt.split()
    if len(parts) < 3:
        await msg.reply_text("⚠️ Dùng: `cal <ký hiệu> <số lượng>`\nVí dụ: `cal btc 0.5`", parse_mode=ParseMode.MARKDOWN)
        return
    
    symbol = parts[1]
    try:
        amount = float(parts[2])
    except ValueError:
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
    
    try:
        await context.bot.edit_message_text(
            chat_id=msg.chat.id,
            message_id=wait.message_id,
            text=res,
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception:
        await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)


async def vol_cmd(update, context):
    """Handle volume command - Get volume data."""
    msg = update.message or update.channel_post
    if not msg:
        return
    
    txt = msg.text.strip()
    if txt.startswith('/'):
        txt = txt[1:]
    
    parts = txt.split()
    if len(parts) < 2:
        await msg.reply_text("⚠️ Dùng: `vol <symbol>` hoặc `vol <symbol> <YYYYMMDD>`", parse_mode=ParseMode.MARKDOWN)
        return
    
    sym = parts[1]
    date_str = parts[2] if len(parts) > 2 else None
    
    wait = await msg.reply_text("⏳ Check vol...")
    res = await get_volume_data(sym, date_str)
    try:
        await context.bot.edit_message_text(
            chat_id=msg.chat.id,
            message_id=wait.message_id,
            text=res,
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception:
        await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)


async def trending_cmd(update, context):
    """Handle trending command - Show trending coins."""
    msg = update.message or update.channel_post
    if not msg:
        return
    res = await get_trending()
    await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)


async def buy_cmd(update, context):
    """Handle buy command - Show top gainers."""
    msg = update.message or update.channel_post
    if not msg:
        return
    res = await get_market('gainers')
    await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)


async def sell_cmd(update, context):
    """Handle sell command - Show top losers."""
    msg = update.message or update.channel_post
    if not msg:
        return
    res = await get_market('losers')
    await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)


async def compe_cmd(update, context):
    """Handle competition command - Show Alpha competition list."""
    msg = update.message or update.channel_post
    if not msg:
        return
    
    wait = await msg.reply_text("⏳ Đang tải dữ liệu Alpha Competition...")
    res = await get_alpha_competition_report()
    
    try:
        await context.bot.edit_message_text(
            chat_id=msg.chat.id,
            message_id=wait.message_id,
            text=res,
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception:
        await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)


async def alpha_cmd(update, context):
    """Handle alpha daily report command."""
    msg = update.message or update.channel_post
    if not msg:
        return
    
    txt = msg.text.strip()
    if txt.startswith('/'):
        txt = txt[1:]
    
    parts = txt.split()
    date_str = parts[1] if len(parts) > 1 else None
    
    wait = await msg.reply_text("⏳ Đang tải Alpha Daily Report...")
    res = await get_alpha_daily_report(date_str)
    
    try:
        await context.bot.edit_message_text(
            chat_id=msg.chat.id,
            message_id=wait.message_id,
            text=res,
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception:
        await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)


async def value_cmd(update, context):
    """Handle value/math command - Evaluate math expression."""
    msg = update.message or update.channel_post
    if not msg:
        return
    
    txt = msg.text.strip()
    if txt.startswith('/'):
        txt = txt[1:]
    
    parts = txt.split(maxsplit=1)
    expr = parts[1] if len(parts) > 1 else ""
    
    if not expr:
        await msg.reply_text("⚠️ Dùng: `val <biểu thức>`\nVí dụ: `val 100 * 2.5 + 50`", parse_mode=ParseMode.MARKDOWN)
        return
    
    res = safe_eval(expr)
    if res is not None:
        await msg.reply_text(f"🧮 Kết quả: `{res}`", parse_mode=ParseMode.MARKDOWN)
    else:
        await msg.reply_text("⚠️ Biểu thức không hợp lệ!", parse_mode=ParseMode.MARKDOWN)


# ==============================================================================
# MAIN MESSAGE HANDLER
# ==============================================================================

def normalize_command(text: str) -> str:
    """Normalize command text by removing leading slash if present."""
    if text.startswith('/'):
        return text[1:]
    return text


async def handle_msg(update, context):
    """Main message handler - Routes incoming messages to appropriate handlers."""
    msg = update.message or update.channel_post
    if not msg or not msg.text:
        return
    
    txt = msg.text.strip()
    txt_normalized = normalize_command(txt)
    txt_norm_lower = txt_normalized.lower()
    
    # Volume analysis: [/]<coin> <timeframe>
    if re.match(r'^[a-zA-Z0-9]{2,10}\s+(15m|1h|3h|4h|24h)$', txt_normalized, re.IGNORECASE):
        return await vol_analysis_handler(update, context)
    
    # Chart command: [/]ch <symbol>
    if txt_norm_lower.startswith('ch '):
        return await chart_cmd(update, context)
    
    # Start/Help command: [/]start
    if txt_norm_lower == 'start' or txt_norm_lower.startswith('start '):
        return await start_cmd(update, context)
    
    # Price report command: [/]p <symbol>
    if txt_norm_lower.startswith('p '):
        parts = txt_normalized.split()
        if len(parts) > 1:
            res = await get_token_report(parts[1])
            await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)
        return
    
    # ATH command: [/]ath <symbol>
    if txt_norm_lower.startswith('ath '):
        parts = txt_normalized.split()
        if len(parts) > 1:
            res = await get_token_report(parts[1])
            await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)
        return
    
    # Calculator command: [/]cal <symbol> <amount>
    if txt_norm_lower.startswith('cal '):
        return await calculate_cmd(update, context)
    
    # Value/Math expression command: [/]val <expression>
    if txt_norm_lower.startswith('val '):
        return await value_cmd(update, context)
    
    # Trending command: [/]trending
    if txt_norm_lower == 'trending' or txt_norm_lower.startswith('trending '):
        return await trending_cmd(update, context)
    
    # Buy (top gainers) command: [/]buy
    if txt_norm_lower == 'buy' or txt_norm_lower.startswith('buy '):
        return await buy_cmd(update, context)
    
    # Sell (top losers) command: [/]sell
    if txt_norm_lower == 'sell' or txt_norm_lower.startswith('sell '):
        return await sell_cmd(update, context)
    
    # Competition command: [/]compe
    if txt_norm_lower == 'compe' or txt_norm_lower.startswith('compe '):
        return await compe_cmd(update, context)
    
    # Alpha daily report command: [/]alpha
    if txt_norm_lower == 'alpha' or txt_norm_lower.startswith('alpha '):
        return await alpha_cmd(update, context)
    
    # Volume command: [/]vol <symbol>
    if txt_norm_lower.startswith('vol '):
        return await vol_cmd(update, context)
    
    # Quick price lookup with $ prefix
    if txt.startswith('$') and len(txt) > 1:
        sym = txt[1:]
        if VALID_SYMBOL_PATTERN.match(sym):
            res = await get_token_report(sym)
            await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)
        return
    
    # Contract address lookup
    if is_contract_address(txt):
        res = await get_dex_data(txt)
        if res == "NOT_FOUND":
            res = "❌ Không tìm thấy contract."
        await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    """Initialize and run the Telegram bot."""
    print("Bot started...")
    
    app = Application.builder().token(BOT_TOKEN).connect_timeout(30).read_timeout(60).build()
    app.add_handler(MessageHandler(filters.TEXT | filters.COMMAND, handle_msg))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()
