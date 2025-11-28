import sys
import requests
import asyncio
import io
import re
import os
import pandas as pd
import mplfinance as mpf
import ccxt.async_support as ccxt
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode
from dotenv import load_dotenv

# --- CONFIG ---
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
except: pass

load_dotenv()
BOT_TOKEN = os.environ.get('BOT_TOKEN')
if not BOT_TOKEN:
    # Fallback
    pass

if not BOT_TOKEN:
    print("LỖI: Thiếu BOT_TOKEN.")
    sys.exit(1)

BINANCE_API_URL = "https://api.binance.com/api/v3"
BINANCE_F_API_URL = "https://fapi.binance.com/fapi/v1"
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
DEXSCREENER_API_URL = "https://api.dexscreener.com/latest/dex"
SUPPORTED_EXCHANGES = ['binance', 'kucoin', 'gateio', 'mexc', 'okx', 'bingx', 'bitget', 'coinbase', 'kraken', 'bitfinex', 'huobi', 'bitstamp', 'cex']
VALID_SYMBOL_PATTERN = re.compile(r'^[a-zA-Z0-9]{1,10}$')

# --- HELPER FUNCTIONS ---
def is_contract_address(address: str) -> bool:
    return address.startswith('0x') and len(address) == 42

def format_price(price: float) -> str:
    if price is None: return "N/A"
    if abs(price) >= 10: return f"${price:,.2f}"
    if abs(price) >= 1: return f"${price:,.3f}"
    if abs(price) >= 0.001: return f"${price:,.4f}"
    return f"${price:,.8f}"

def safe_eval(expression: str) -> float | None:
    if not re.match(r'^[0-9\s\+\-\*\/\(\)\.]+$', expression): return None
    try: return eval(expression, {"__builtins__": {}}, {})
    except: return None

# --- DATA FETCHING ---

def get_coingecko_id(symbol: str) -> str | None:
    try:
        d = requests.get(f"{COINGECKO_API_URL}/search?query={symbol}", timeout=10).json()
        for c in d.get('coins', []):
            if c.get('symbol', '').lower() == symbol.lower(): return c.get('id')
    except: pass
    return None

def get_coingecko_info(symbol: str):
    """Lấy thông tin bổ sung (ATH, Market Cap...) từ CoinGecko"""
    cid = get_coingecko_id(symbol)
    if not cid: return None
    try:
        d = requests.get(f"{COINGECKO_API_URL}/coins/{cid}", timeout=10).json()
        md = d.get('market_data', {})
        return {
            'ath': md.get('ath', {}).get('usd', 0),
            'ath_change': md.get('ath_change_percentage', {}).get('usd', 0),
            'ath_date': md.get('ath_date', {}).get('usd', '').split('T')[0],
            'cap': md.get('market_cap', {}).get('usd', 0),
            'circulating': md.get('circulating_supply', 0),
            'total_supply': md.get('total_supply', 0)
        }
    except: return None

def get_binance_kline_change(trading_pair: str, interval: str, is_future=False) -> float | None:
    api_url = BINANCE_F_API_URL if is_future else BINANCE_API_URL
    try:
        url = f"{api_url}/klines?symbol={trading_pair}&interval={interval}&limit=2"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200: return None
        k = resp.json()
        if len(k) < 2: return None
        prev_close = float(k[0][4])
        curr_price = float(k[1][4])
        if prev_close == 0: return 0.0
        return ((curr_price - prev_close) / prev_close) * 100
    except: return None

# --- MAIN REPORT GENERATION ---

async def get_token_report(symbol: str) -> str:
    """
    Tạo báo cáo chi tiết cho token (Dùng cho lệnh /p và $symbol)
    """
    symbol_upper = symbol.upper()
    stables = ['USDT', 'USDC', 'FDUSD']
    
    price_data = None
    source = ""
    pair_name = ""
    
    # 1. Thử Binance Spot & Futures
    for is_future in [False, True]:
        api_url = BINANCE_F_API_URL if is_future else BINANCE_API_URL
        src_name = "Binance Futures" if is_future else "Sàn Binance"
        
        for s in stables:
            pair = f"{symbol_upper}{s}"
            try:
                url = f"{api_url}/ticker/24hr?symbol={pair}"
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(None, requests.get, url)
                
                if resp.status_code == 200:
                    data = resp.json()
                    price = float(data['lastPrice'])
                    p24h = float(data['priceChangePercent'])
                    p1h = get_binance_kline_change(pair, '1h', is_future)
                    p5m = get_binance_kline_change(pair, '5m', is_future)
                    
                    price_24h = price / (1 + p24h/100) if p24h != -100 else 0
                    price_1h = price / (1 + p1h/100) if p1h else 0
                    price_5m = price / (1 + p5m/100) if p5m else 0
                    
                    price_data = {
                        'price': price,
                        'p5m': p5m, 'price_5m': price_5m,
                        'p1h': p1h, 'price_1h': price_1h,
                        'p24h': p24h, 'price_24h': price_24h,
                        'vol': float(data['quoteVolume']),
                        'vol_unit': s
                    }
                    source = src_name
                    pair_name = f"{symbol_upper}/{s}"
                    break 
            except: continue
        if price_data: break

    # 2. CCXT Exchanges
    if not price_data:
        for ex_id in ['gateio', 'mexc', 'kucoin', 'okx', 'bybit']:
            try:
                ex = getattr(ccxt, ex_id)()
                # Quick check
                for s in stables:
                    try:
                        t = await ex.fetch_ticker(f"{symbol_upper}/{s}")
                        price = t['last']
                        price_data = {
                            'price': price,
                            'p24h': t.get('percentage', 0),
                            'price_24h': price / (1 + t.get('percentage', 0)/100) if t.get('percentage') else 0,
                            'vol': t.get('quoteVolume', 0),
                            'vol_unit': 'USDT'
                        }
                        source = f"Sàn {ex.name}"
                        pair_name = t['symbol']
                        await ex.close()
                        break
                    except: continue
                if price_data: 
                    await ex.close()
                    break
                await ex.close()
            except: pass

    # 3. DexScreener
    if not price_data:
        try:
            url = f"{DEXSCREENER_API_URL}/search?q={symbol}"
            d = requests.get(url, timeout=5).json()
            if d.get('pairs'):
                p = max(d['pairs'], key=lambda x: x.get('liquidity', {}).get('usd', 0))
                price = float(p['priceUsd'])
                price_data = {
                    'price': price,
                    'p5m': p.get('priceChange', {}).get('m5'),
                    'p1h': p.get('priceChange', {}).get('h1'),
                    'p24h': p.get('priceChange', {}).get('h24'),
                    'vol': p.get('volume', {}).get('h24', 0),
                    'vol_unit': 'USD'
                }
                # Calc past prices
                if price_data['p5m']: price_data['price_5m'] = price / (1 + price_data['p5m']/100)
                if price_data['p1h']: price_data['price_1h'] = price / (1 + price_data['p1h']/100)
                if price_data['p24h']: price_data['price_24h'] = price / (1 + price_data['p24h']/100)
                
                source = f"DexScreener ({p.get('dexId')})"
                pair_name = f"{p['baseToken']['symbol']}/{p['quoteToken']['symbol']}"
        except: pass

    if not price_data:
        return f"😕 Không tìm thấy thông tin cho **{symbol_upper}**."

    cg_info = get_coingecko_info(symbol)
    
    # --- FORMAT OUTPUT (UPDATED) ---
    msg = f"🪙 {pair_name} ({source})\n"
    msg += "--------------------\n"
    msg += f"Giá: {format_price(price_data['price'])}\n"
    msg += "--------------------\n"
    msg += "Biến động (giá quá khứ):\n"
    
    if price_data.get('p5m') is not None:
        icon = '📈' if price_data['p5m'] >= 0 else '📉'
        val_5m = format_price(price_data.get('price_5m', 0)) if price_data.get('price_5m') else 'N/A'
        msg += f"- 5 phút:  {icon} {price_data['p5m']:+.2f}% ({val_5m})\n"
    
    if price_data.get('p1h') is not None:
        icon = '📈' if price_data['p1h'] >= 0 else '📉'
        val_1h = format_price(price_data.get('price_1h', 0)) if price_data.get('price_1h') else 'N/A'
        msg += f"- 1 giờ:   {icon} {price_data['p1h']:+.2f}% ({val_1h})\n"
        
    if price_data.get('p24h') is not None:
        icon = '📈' if price_data['p24h'] >= 0 else '📉'
        val_24h = format_price(price_data.get('price_24h', 0)) if price_data.get('price_24h') else 'N/A'
        msg += f"- 24 giờ: {icon} {price_data['p24h']:+.2f}% ({val_24h})\n"

    msg += "--------------------\n"
    msg += f"Tổng KL (24h): ${price_data['vol']:,.0f} {price_data['vol_unit']}\n"

    if cg_info:
        msg += "--------------------\n"
        msg += f"ATH (CoinGecko): {format_price(cg_info['ath'])} ({cg_info['ath_change']:+.2f}%)\n"
        msg += f"- Ngày ATH: {cg_info['ath_date']}\n"
        msg += "--------------------\n"
        msg += "Thống kê (CoinGecko):\n"
        msg += f"- Vốn hóa TT: ${cg_info['cap']:,.0f}\n"
        msg += f"- Lưu thông: {cg_info['circulating']:,.0f}\n"
        val_supply = f"{cg_info['total_supply']:,.0f}" if cg_info['total_supply'] else "∞"
        msg += f"- Tổng cung: {val_supply}\n"

    msg += f"\n_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
    return msg

# --- OTHER FEATURES ---
async def generate_ccxt_chart(symbol: str, interval: str, theme: str = 'dark') -> tuple[io.BytesIO | None, str | None]:
    """
    Vẽ chart với màu nến Xanh/Đỏ chuẩn cho cả Dark và Light theme.
    """
    print(f"Vẽ chart {symbol} {interval}...")
    stables = ['USDT', 'USDC', 'FDUSD', 'USD']
    for ex_id in SUPPORTED_EXCHANGES:
        ex = getattr(ccxt, ex_id)()
        try:
            await ex.load_markets()
            for s in stables:
                pair = f"{symbol.upper()}/{s}"
                if pair in ex.markets:
                    ohlcv = await ex.fetch_ohlcv(pair, timeframe=interval, limit=100)
                    if not ohlcv: continue
                    
                    df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
                    df = df.set_index('Date')
                    
                    # --- CẤU HÌNH MÀU SẮC (FIXED) ---
                    # up='green', down='red' cho cả hai theme
                    mc = mpf.make_marketcolors(up='#0ECB81', down='#F6465D', inherit=True) 
                    
                    if theme == 'light':
                        s = mpf.make_mpf_style(base_mpf_style='yahoo', marketcolors=mc)
                    else:
                        s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc)
                    
                    buf = io.BytesIO()
                    mpf.plot(df, type='candle', style=s, title=f'\n{ex.name}: {pair} - {interval}', 
                             volume=True, savefig=dict(fname=buf, dpi=100, pad_inches=0.25))
                    buf.seek(0)
                    await ex.close()
                    return buf, f"{ex.name}: {pair}"
        except: pass
        finally: await ex.close()
    return None, None

def generate_coingecko_chart(coin_id: str, interval: str, theme: str = 'dark') -> tuple[io.BytesIO | None, str | None]:
    days = '1' if interval in ['5m', '15m', '1h', '4h'] else '90'
    try:
        url = f"{COINGECKO_API_URL}/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
        data = requests.get(url, timeout=10).json()
        if not data: return None, None
        
        df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df = df.set_index('Date')
        
        rule = interval.replace('m', 'T') if interval in ['5m', '15m'] else '1h' if interval in ['1h', '4h'] else '1D'
        if interval != '1d': df = df.resample(rule).agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
        
        # --- CẤU HÌNH MÀU SẮC (FIXED) ---
        mc = mpf.make_marketcolors(up='#0ECB81', down='#F6465D', inherit=True)
        s = mpf.make_mpf_style(base_mpf_style='yahoo' if theme == 'light' else 'nightclouds', marketcolors=mc)
        
        buf = io.BytesIO()
        mpf.plot(df, type='candle', style=s, title=f'\n{coin_id.upper()} - {interval} (CG)', 
                 volume=False, savefig=dict(fname=buf, dpi=100, pad_inches=0.25))
        buf.seek(0)
        return buf, coin_id.upper()
    except: return None, None

# --- FEATURE: ADVANCED BUY/SELL VOLUME ---
async def get_buy_sell_vol(symbol: str, interval: str) -> str:
    tf_map = {'15m':'15m', '1h':'1h', '3h':'1h', '24h':'1d', '4h':'4h', '1d':'1d'}
    tf = tf_map.get(interval)
    limit = 3 if interval == '3h' else 1
    try:
        pair = f"{symbol.upper()}USDT"
        url = f"{BINANCE_API_URL}/klines?symbol={pair}&interval={tf}&limit={limit}"
        resp = requests.get(url, timeout=10)
        klines = resp.json() if resp.status_code == 200 else None
        if not klines: return f"😕 Không tìm thấy volume Binance cho {symbol.upper()}."
        total = sum(float(k[5]) for k in klines)
        buy = sum(float(k[9]) for k in klines)
        sell = total - buy
        if total == 0: return "Volume 0."
        pct = (buy/total)*100
        net = buy - sell
        state = "🟢 MUA > BÁN" if pct > 50 else "🔴 BÁN > MUA"
        return (f"📊 **Vol {symbol.upper()} ({interval})**\nNguồn: `Binance Spot`\n----------------\n**{state}**\n"
                f"🟢 Mua: `{format_price(buy)}` ({pct:.1f}%)\n🔴 Bán: `{format_price(sell)}` ({100-pct:.1f}%)\n"
                f"⚖️ Ròng: `{format_price(net)}` {symbol.upper()}")
    except: return "Lỗi phân tích volume."

async def get_current_price(symbol: str) -> tuple[float | None, str | None]:
    """Lấy giá hiện tại (dùng cho /cal)"""
    stables = ['USDT', 'USDC', 'FDUSD', 'USD']
    for ex_id in ['binance', 'gateio', 'mexc']: 
        ex = getattr(ccxt, ex_id)()
        try:
            for s in stables:
                t = await ex.fetch_ticker(f"{symbol.upper()}/{s}")
                if t and t.get('last'): 
                    await ex.close()
                    return t['last'], ex.name
        except: pass
        finally: await ex.close()
    
    cid = get_coingecko_id(symbol)
    if cid:
        try:
            p = requests.get(f"{COINGECKO_API_URL}/simple/price?ids={cid}&vs_currencies=usd", timeout=10).json()
            if p.get(cid, {}).get('usd'): return float(p[cid]['usd']), "CoinGecko"
        except: pass
    return None, "Không tìm thấy giá."

async def get_dex_data(address: str) -> str:
    try:
        d = requests.get(f"{DEXSCREENER_API_URL}/search?q={address}", timeout=10).json()
        if not d.get('pairs'): return "NOT_FOUND"
        p = d['pairs'][0]
        return (f"🪙 **{p['baseToken']['name']}** (DexScreener)\nPrice: {format_price(float(p['priceUsd']))}\n"
                f"Vol 24h: `${p.get('volume', {}).get('h24', 0):,.0f}`")
    except: return "NOT_FOUND"

async def get_volume_data(symbol: str, date_str: str = None) -> str:
    cid = get_coingecko_id(symbol)
    if not cid: return "❌ Không tìm thấy coin."
    try:
        if date_str:
            dt = datetime.strptime(date_str, '%Y%m%d').strftime('%d-%m-%Y')
            d = requests.get(f"{COINGECKO_API_URL}/coins/{cid}/history?date={dt}", timeout=10).json()
            vol = d.get('market_data', {}).get('total_volume', {}).get('usd')
            return f"📊 **Vol {symbol.upper()} ngày {date_str}:** `${int(vol):,}`" if vol else "Không có dữ liệu."
        else:
            d = requests.get(f"{COINGECKO_API_URL}/coins/{cid}/market_chart?vs_currency=usd&days=max&interval=daily", timeout=10).json()
            vols = d.get('total_volumes', [])
            total = sum(x[1] for x in vols)
            return f"📊 **Tổng Vol Tích Lũy {symbol.upper()}:** `${int(total):,}`"
    except: return "Lỗi dữ liệu."

async def get_trending() -> str:
    try:
        d = requests.get(f"{COINGECKO_API_URL}/search/trending", timeout=10).json()
        msg = "🔥 **Trending:**\n"
        for i, c in enumerate(d['coins'][:7]): msg += f"{i+1}. {c['item']['symbol']}\n"
        return msg
    except: return "Lỗi trending."

async def get_market(order, limit=10):
    sort = 'price_change_percentage_24h_desc' if order == 'gainers' else 'price_change_percentage_24h_asc'
    try:
        d = requests.get(f"{COINGECKO_API_URL}/coins/markets?vs_currency=usd&order={sort}&per_page={limit}", timeout=10).json()
        msg = f"📊 **Top {order.title()}**\n"
        for c in d: msg += f"{c['symbol'].upper()}: {format_price(c['current_price'])} ({c['price_change_percentage_24h']:+.2f}%)\n"
        return msg
    except: return "Lỗi market data."

# --- COMMAND HANDLERS ---
async def start_cmd(update, context):
    await update.message.reply_text(
        "🤖 **Crypto Bot Pro**\n"
        "`/<coin> {15m, 1h, 3h, 24h}` (vd /sui 1h)\n"
        "`/p <symbol>` - Xem giá chi tiết\n"
        "`/ch <symbol>` - Xem chart\n"
        "`/cal <coin> <amount>`\n"
        "`/vol <coin> [date]`, `/ath <coin>`\n"
        "`/trending`, `/buy`, `/sell`\n"
        "`$symbol`, `contract`",
        parse_mode=ParseMode.MARKDOWN
    )

async def chart_cmd(update, context):
    msg = update.message or update.channel_post
    if not msg: return
    parts = msg.text.split()
    if len(parts)<2: return await msg.reply_text("Thiếu symbol.")
    sym = parts[1]
    tf = parts[2] if len(parts)>2 else '1h'
    theme = 'light' if 'light' in parts else 'dark'
    wait = await msg.reply_text(f"⏳ Vẽ chart {sym}...")
    buf, name = await generate_ccxt_chart(sym, tf, theme)
    if not buf: 
        cid = get_coingecko_id(sym)
        if cid: buf, name = generate_coingecko_chart(cid, tf, theme)
    await context.bot.delete_message(msg.chat.id, wait.message_id)
    if buf: await msg.reply_photo(buf, caption=f"{name} {tf}")
    else: await msg.reply_text("Không vẽ được chart.")

async def vol_analysis_handler(update, context):
    msg = update.message or update.channel_post
    if not msg: return
    match = re.match(r'^/([a-zA-Z0-9]{2,10})\s+(15m|1h|3h|4h|24h)$', msg.text.strip(), re.IGNORECASE)
    if not match: return
    sym = match.group(1)
    tf = match.group(2).lower()
    wait = await msg.reply_text(f"⏳ Phân tích vol {sym} {tf}...")
    res = await get_buy_sell_vol(sym, tf)
    await context.bot.edit_message_text(chat_id=msg.chat.id, message_id=wait.message_id, text=res, parse_mode=ParseMode.MARKDOWN)

async def ath_cmd(update, context):
    msg = update.message or update.channel_post
    if not msg: return
    parts = msg.text.split()
    if len(parts) < 2: return
    res = await get_token_report(parts[1])
    await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)

async def calculate_cmd(update, context):
    msg = update.message or update.channel_post
    if not msg: return
    parts = msg.text.strip().split()
    if len(parts) < 3:
        await msg.reply_text("⚠️ Dùng: `/cal <ký hiệu> <số lượng>`", parse_mode=ParseMode.MARKDOWN)
        return
    symbol = parts[1]
    try: amount = float(parts[2])
    except: return
    wait = await msg.reply_text(f"🔍 Tính toán {symbol}...")
    p, src = await get_current_price(symbol)
    if p:
        total = p * amount
        res = (f"💰 **Kết quả tính toán**\n--------------------\n"
               f"💵 **Giá:** `{format_price(p)}` / {symbol.upper()}\n"
               f"🔢 **SL:** `{amount:g}`\n"
               f"--------------------\n💎 **Tổng:** `{format_price(total)}`")
    else: res = f"😕 Không tìm thấy giá {symbol.upper()}."
    await context.bot.edit_message_text(chat_id=msg.chat.id, message_id=wait.message_id, text=res, parse_mode=ParseMode.MARKDOWN)

async def vol_cmd(update, context):
    msg = update.message or update.channel_post
    parts = msg.text.split()
    if len(parts) < 2: return
    sym = parts[1]
    date_str = parts[2] if len(parts) > 2 else None
    wait = await msg.reply_text("⏳ Check vol...")
    res = await get_volume_data(sym, date_str)
    await context.bot.edit_message_text(chat_id=msg.chat.id, message_id=wait.message_id, text=res, parse_mode=ParseMode.MARKDOWN)

async def trending_cmd(update, context):
    res = await get_trending()
    await update.message.reply_text(res, parse_mode=ParseMode.MARKDOWN)

async def buy_cmd(update, context):
    res = await get_market('gainers')
    await update.message.reply_text(res, parse_mode=ParseMode.MARKDOWN)

async def sell_cmd(update, context):
    res = await get_market('losers')
    await update.message.reply_text(res, parse_mode=ParseMode.MARKDOWN)

async def value_cmd(update, context):
    msg = update.message or update.channel_post
    txt = msg.text.strip()
    expr = txt.split(maxsplit=1)[1] if len(txt.split()) > 1 else ""
    res = safe_eval(expr)
    if res is not None: await msg.reply_text(f"🧮 = `{res}`", parse_mode=ParseMode.MARKDOWN)

async def handle_msg(update, context):
    msg = update.message or update.channel_post
    if not msg or not msg.text: return
    txt = msg.text.strip()
    txt_lower = txt.lower()

    if re.match(r'^/[a-zA-Z0-9]{2,10}\s+(15m|1h|3h|4h|24h)$', txt, re.IGNORECASE):
        return await vol_analysis_handler(update, context)

    if txt_lower.startswith('/ch '): return await chart_cmd(update, context)
    if txt_lower.startswith('/start'): return await start_cmd(update, context)
    if txt_lower.startswith('/p ') or txt_lower.startswith('/ath '): 
        parts = txt.split()
        if len(parts) > 1:
            res = await get_token_report(parts[1])
            await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)
        return
    if txt_lower.startswith(('/cal ', 'cal ')): return await calculate_cmd(update, context)
    if txt_lower.startswith(('/val ', 'val ')): return await value_cmd(update, context)
    if txt_lower.startswith('/trending'): return await trending_cmd(update, context)
    if txt_lower.startswith('/buy'): return await buy_cmd(update, context)
    if txt_lower.startswith('/sell'): return await sell_cmd(update, context)
    if txt_lower.startswith('/vol'): return await vol_cmd(update, context)

    # Price Lookup ($symbol)
    if txt.startswith('$') and len(txt) > 1:
        sym = txt[1:]
        if VALID_SYMBOL_PATTERN.match(sym):
            res = await get_token_report(sym)
            await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)
        return

    if is_contract_address(txt):
        res = await get_dex_data(txt)
        if res == "NOT_FOUND":
             info = get_info_from_contract(txt)
             if info: res = await get_token_report(info['symbol'])
             else: res = "❌ Không tìm thấy contract."
        await msg.reply_text(res, parse_mode=ParseMode.MARKDOWN)

def main():
    print("Bot started...")
    app = Application.builder().token(BOT_TOKEN).connect_timeout(30).read_timeout(60).build()
    app.add_handler(MessageHandler(filters.TEXT | filters.COMMAND, handle_msg))
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()