import sys
import requests
from bs4 import BeautifulSoup
import asyncio
from datetime import datetime, date, timedelta
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.constants import ParseMode
import pandas as pd
import mplfinance as mpf
import io
import ccxt.async_support as ccxt  # Sử dụng phiên bản bất đồng bộ của ccxt
import re # Thêm thư viện để kiểm tra biểu thức an toàn
import os # Thêm thư viện OS để lấy biến môi trường

# --- CẤU HÌNH ENCODING TỰ ĐỘNG CHO TERMINAL ---
# (Phần này có thể không cần thiết trên server Linux nhưng giữ lại cũng không sao)
try:
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
except (AttributeError, TypeError):
    # Bỏ qua lỗi nếu không chạy trong môi trường có thể reconfigure
    pass
# ------------------------------------------------

# --- THÔNG TIN CẤU HÌNH ---
# Lấy BOT_TOKEN từ biến môi trường để đảm bảo an toàn
BOT_TOKEN = os.environ.get('BOT_TOKEN')
if not BOT_TOKEN:
    # Nếu chạy local mà không set env, bạn có thể điền token vào đây
    # BOT_TOKEN = "8291363338:AAEPw0C0QMriY3uEsGIX0Dh5fY1HgIUwGyo" 
    
    # Dòng code dưới sẽ thoát nếu không tìm thấy BOT_TOKEN
    print("LỖI: Không tìm thấy BOT_TOKEN. Vui lòng đặt biến môi trường.")
    sys.exit(1)
# -------------------------

# --- API Endpoints ---
BINANCE_API_URL = "https://api.binance.com/api/v3" # Giữ lại để lấy % change nhanh
COINGECKO_API_URL = "https://api.coingecko.com/api/v3"
DEXSCREENER_API_URL = "https://api.dexscreener.com/latest/dex"

# Danh sách các sàn được hỗ trợ bởi CCXT (Đã sửa coinbasepro -> coinbase)
SUPPORTED_EXCHANGES = [
    'binance', 'kucoin', 'gateio', 'mexc', 'okx', 'bingx', 'bitget',
    'coinbase', 'kraken', 'bitfinex', 'huobi', 'bitstamp', 'cex'
]

# --- Regex Patterns ---
VALID_SYMBOL_PATTERN = re.compile(r'^[a-zA-Z0-9]{1,10}$')
DATE_PATTERN = re.compile(r'^\d{8}$') # YYYYMMDD

def is_contract_address(address: str) -> bool:
    """Kiểm tra xem một chuỗi có phải là địa chỉ contract EVM hợp lệ hay không."""
    return address.startswith('0x') and len(address) == 42

def format_price(price: float) -> str:
    """Định dạng giá một cách linh hoạt dựa trên giá trị của nó."""
    if price is None: return "N/A"
    if abs(price) >= 10:
        return f"${price:,.2f}"
    if abs(price) >= 1:
        return f"${price:,.3f}"
    if abs(price) >= 0.001:
        return f"${price:,.4f}"
    return f"${price:,.8f}"

def safe_eval(expression: str) -> float | None:
    """Đánh giá một biểu thức toán học một cách an toàn."""
    allowed_pattern = re.compile(r'^[0-9\s\+\-\*\/\(\)\.]+$')
    if not allowed_pattern.match(expression):
        return None
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return result
    except (SyntaxError, ZeroDivisionError, TypeError, NameError):
        return None

def get_info_from_contract(contract_address: str) -> dict | None:
    """
    Sử dụng API CoinGecko để tìm thông tin (symbol, id, chain) của token từ địa chỉ contract.
    """
    print(f"Đang tìm thông tin cho contract: {contract_address}")
    chains = ['ethereum', 'binance-smart-chain', 'polygon-pos', 'arbitrum-one', 'solana']
    for chain in chains:
        try:
            url = f"{COINGECKO_API_URL}/coins/{chain}/contract/{contract_address}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                symbol = data.get('symbol')
                coin_id = data.get('id')
                if symbol and coin_id:
                    print(f"Tìm thấy: symbol '{symbol.upper()}', id '{coin_id}' trên chain '{chain}'")
                    return {'symbol': symbol.upper(), 'id': coin_id, 'chain': chain, 'contract': contract_address}
        except requests.RequestException:
            continue
    print("Không tìm thấy thông tin nào cho contract này trên CoinGecko.")
    return None

def get_coingecko_id_from_symbol(symbol: str) -> str | None:
    """
    Sử dụng API tìm kiếm của CoinGecko để tìm coin ID từ ký hiệu (chỉ khớp chính xác).
    """
    print(f"Đang tìm kiếm ID trên CoinGecko cho ký hiệu: {symbol}")
    try:
        url = f"{COINGECKO_API_URL}/search?query={symbol}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            coins = data.get('coins', [])
            for coin in coins:
                if coin.get('symbol', '').lower() == symbol.lower():
                    coin_id = coin.get('id')
                    print(f"Tìm thấy ID khớp chính xác '{coin_id}' cho ký hiệu '{symbol}'")
                    return coin_id
    except requests.RequestException as e:
        print(f"Lỗi khi tìm kiếm trên CoinGecko: {e}")
    
    print(f"Không tìm thấy ID khớp chính xác cho ký hiệu '{symbol}' trên CoinGecko.")
    return None


async def generate_ccxt_chart(symbol: str, interval: str, theme: str = 'dark') -> tuple[io.BytesIO | None, str | None]:
    """
    Tạo biểu đồ nến bằng CCXT, tự động dò tìm trên các sàn được hỗ trợ.
    """
    print(f"Đang tạo biểu đồ cho {symbol} khung {interval} từ CCXT...")
    stablecoins = ['USDT', 'USDC', 'FDUSD', 'TUSD', 'USD'] # Thêm USD cho một số sàn
    
    for exchange_id in SUPPORTED_EXCHANGES:
        exchange = getattr(ccxt, exchange_id)()
        try:
            await exchange.load_markets()
            for stable in stablecoins:
                trading_pair = f"{symbol.upper()}/{stable}"
                if trading_pair in exchange.markets:
                    print(f"Tìm thấy cặp {trading_pair} trên sàn {exchange.name}. Đang lấy dữ liệu OHLCV...")
                    ohlcv = await exchange.fetch_ohlcv(trading_pair, timeframe=interval, limit=100)
                    if not ohlcv:
                        continue

                    df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
                    df = df.set_index('Date')

                    if theme == 'light':
                        s = mpf.make_mpf_style(base_mpf_style='yahoo')
                    else:
                        mc = mpf.make_marketcolors(up='#00b894', down='#d63031', inherit=True)
                        s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc)

                    buf = io.BytesIO()
                    mpf.plot(df, type='candle', style=s,
                             title=f'\n{exchange.name}: {trading_pair} - Khung {interval}',
                             volume=True,
                             savefig=dict(fname=buf, dpi=100, pad_inches=0.25))
                    buf.seek(0)
                    await exchange.close()
                    print(f"Tạo biểu đồ từ {exchange.name} thành công.")
                    return buf, f"{exchange.name}: {trading_pair}"
        except Exception as e:
            print(f"Lỗi với sàn {exchange.name}: {e}")
        finally:
            await exchange.close()
            
    return None, None

def generate_coingecko_chart(coin_id: str, interval: str, theme: str = 'dark') -> tuple[io.BytesIO | None, str | None]:
    """
    Tạo biểu đồ nến từ dữ liệu OHLC của CoinGecko.
    """
    print(f"Fallback: Đang tạo biểu đồ từ CoinGecko cho {coin_id} khung {interval}...")
    days = '90'
    resample_rule = '1D'
    if interval in ['5m', '15m', '1h', '4h']:
        days = '1'
        resample_rule = interval.replace('m', 'T')
    
    try:
        info_url = f"{COINGECKO_API_URL}/coins/{coin_id}"
        info_response = requests.get(info_url, timeout=10)
        coin_info = info_response.json()
        symbol = coin_info.get('symbol', 'N/A').upper()
        name = coin_info.get('name', 'N/A')

        url = f"{COINGECKO_API_URL}/coins/{coin_id}/ohlc?vs_currency=usd&days={days}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not data:
            return None, None

        df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df = df.set_index('Date')

        if interval != '1d':
            df = df.resample(resample_rule).agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last'}).dropna()
        
        if df.empty:
             print("DataFrame trống sau khi resample.")
             return None, None
             
        if theme == 'light':
            s = mpf.make_mpf_style(base_mpf_style='yahoo')
        else:
            mc = mpf.make_marketcolors(up='#00b894', down='#d63031', inherit=True)
            s = mpf.make_mpf_style(base_mpf_style='nightclouds', marketcolors=mc)

        buf = io.BytesIO()
        mpf.plot(df, type='candle', style=s,
                 title=f'\n{name} ({symbol}) - Khung {interval} (Nguồn: CoinGecko)',
                 volume=False,
                 savefig=dict(fname=buf, dpi=100, pad_inches=0.25))
        buf.seek(0)
        print("Tạo biểu đồ CoinGecko thành công.")
        return buf, f"{symbol}/USD"
    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ CoinGecko: {e}")
        return None, None
    
def get_binance_kline_change(trading_pair: str, interval: str) -> float | None:
    """Lấy % thay đổi từ dữ liệu Kline cho một khoảng thời gian cụ thể (chỉ dùng cho Binance)."""
    try:
        url = f"{BINANCE_API_URL}/klines?symbol={trading_pair}&interval={interval}&limit=2"
        response = requests.get(url, timeout=10)
        if response.status_code != 200: return None
        kline_data = response.json()
        if len(kline_data) < 2: return None
        previous_close_price = float(kline_data[0][4])
        current_price = float(kline_data[1][4])
        if previous_close_price == 0: return 0.0
        return ((current_price - previous_close_price) / previous_close_price) * 100
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu kline cho {trading_pair} ({interval}): {e}")
        return None

def get_binance_price_data(symbol: str) -> str:
    """
    Lấy dữ liệu giá từ API của Binance và bổ sung thông tin từ CoinGecko.
    """
    stablecoins = ['USDT', 'USDC', 'FDUSD', 'TUSD']
    binance_message = "NOT_FOUND_ON_BINANCE"
    binance_price = None
    trading_pair_found = ""

    for stable in stablecoins:
        trading_pair = f"{symbol.upper()}{stable}"
        print(f"Đang thử lấy dữ liệu từ Binance cho cặp: {trading_pair}")
        try:
            price_url = f"{BINANCE_API_URL}/ticker/price?symbol={trading_pair}"
            price_response = requests.get(price_url, timeout=10)
            if price_response.status_code != 200:
                continue
            binance_price = float(price_response.json().get('price', 0))

            stats_url = f"{BINANCE_API_URL}/ticker/24hr?symbol={trading_pair}"
            stats_response = requests.get(stats_url, timeout=10)
            if stats_response.status_code == 200:
                stats_data = stats_response.json()
                
                p24h = float(stats_data.get('priceChangePercent', 0))
                # Chỉ lấy Klines từ Binance khi nguồn là Binance
                p5m = get_binance_kline_change(trading_pair, '5m')
                p1h = get_binance_kline_change(trading_pair, '1h')
                
                price_24h_ago = binance_price / (1 + p24h / 100) if p24h != -100 else 0
                price_1h_ago = binance_price / (1 + p1h / 100) if p1h is not None and p1h != -100 else None
                price_5m_ago = binance_price / (1 + p5m / 100) if p5m is not None and p5m != -100 else None
                
                changes_lines = []
                if p5m is not None:
                    changes_lines.append(f"- 5 phút:  {'📈' if p5m >= 0 else '📉'} `{p5m:+.2f}%` ({format_price(price_5m_ago)})")
                if p1h is not None:
                    changes_lines.append(f"- 1 giờ:   {'📈' if p1h >= 0 else '📉'} `{p1h:+.2f}%` ({format_price(price_1h_ago)})")
                changes_lines.append(f"- 24 giờ: {'📈' if p24h >= 0 else '📉'} `{p24h:+.2f}%` ({format_price(price_24h_ago)})")
                changes_str = "\n".join(changes_lines)
                
                quote_volume = float(stats_data.get('quoteVolume', 0))
                
                binance_message = (
                    f"🪙 **{symbol.upper()}/{stable}** (Sàn Binance)\n"
                    f"--------------------\n"
                    f"**Giá:** {format_price(binance_price)}\n"
                    f"--------------------\n"
                    f"**Biến động (giá quá khứ):**\n"
                    f"{changes_str}\n"
                    f"--------------------\n"
                    f"**Tổng KL (24h):** `${int(quote_volume):,}` {stable}\n"
                )
                trading_pair_found = trading_pair
                print(f"Lấy dữ liệu Binance thành công cho cặp {trading_pair}.")
                break
        except requests.RequestException:
            continue
    
    # Bổ sung thông tin từ CoinGecko nếu tìm thấy trên Binance
    if "NOT_FOUND_ON_BINANCE" not in binance_message and binance_price is not None:
        coin_id = get_coingecko_id_from_symbol(symbol)
        if coin_id:
            try:
                url = f"{COINGECKO_API_URL}/coins/{coin_id}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    market_data = data.get('market_data', {})
                    market_cap = market_data.get('market_cap', {}).get('usd', 0)
                    circulating_supply = market_data.get('circulating_supply', 0)
                    total_supply = market_data.get('total_supply', 0)
                    ath = market_data.get('ath', {}).get('usd', 0)
                    ath_change = market_data.get('ath_change_percentage', {}).get('usd', 0.0)
                    ath_date_str = market_data.get('ath_date', {}).get('usd', '')
                    ath_date = datetime.fromisoformat(ath_date_str.replace('Z', '+00:00')).strftime('%Y-%m-%d') if ath_date_str else 'N/A'
                    
                    binance_message += (
                        f"--------------------\n"
                        f"**ATH (CoinGecko):** {format_price(ath)} (`{ath_change:+.2f}%`)\n"
                        f"- Ngày ATH: `{ath_date}`\n"
                        f"--------------------\n"
                        f"**Thống kê (CoinGecko):**\n"
                        f"- Vốn hóa TT: `${int(market_cap):,}`\n"
                        f"- Lưu thông: `{int(circulating_supply):,}`\n"
                        f"- Tổng cung: `{int(total_supply) if total_supply else 'N/A'}`\n"
                    )
            except Exception as e:
                 print(f"Lỗi khi lấy thêm dữ liệu từ CoinGecko: {e}")
        binance_message += f"\n_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        return binance_message

    return f"NOT_FOUND_ON_BINANCE"

def get_coingecko_price_data(coin_id: str, known_contract_info: dict = None) -> str:
    print(f"Fallback: Đang lấy dữ liệu từ CoinGecko cho coin id: {coin_id}")
    try:
        url = f"{COINGECKO_API_URL}/coins/{coin_id}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        name = data.get('name', 'N/A')
        symbol = data.get('symbol', 'N/A').upper()
        market_data = data.get('market_data', {})
        
        price = market_data.get('current_price', {}).get('usd', 0)
        p1h = market_data.get('price_change_percentage_1h_in_currency', {}).get('usd', 0.0) or 0.0
        p24h = market_data.get('price_change_percentage_24h_in_currency', {}).get('usd', 0.0) or 0.0
        p7d = market_data.get('price_change_percentage_7d_in_currency', {}).get('usd', 0.0) or 0.0
        
        market_cap = market_data.get('market_cap', {}).get('usd', 0)
        volume_24h = market_data.get('total_volume', {}).get('usd', 0)
        circulating_supply = market_data.get('circulating_supply', 0)
        total_supply = market_data.get('total_supply', 0)
        genesis_date = data.get('genesis_date')
        
        ath = market_data.get('ath', {}).get('usd', 0)
        ath_change = market_data.get('ath_change_percentage', {}).get('usd', 0.0)
        ath_date_str = market_data.get('ath_date', {}).get('usd', '')
        ath_date = datetime.fromisoformat(ath_date_str.replace('Z', '+00:00')).strftime('%Y-%m-%d') if ath_date_str else 'N/A'

        contract_str = ""
        if known_contract_info:
            contract_str = f"**Contract ({known_contract_info['chain']}):**\n`{known_contract_info['contract']}`\n--------------------\n"
        elif data.get('contract_address'):
             contract_str = f"**Contract (ETH):**\n`{data.get('contract_address')}`\n--------------------\n"
        
        message = (
            f"🪙 **{name} ({symbol})** (Nguồn: CoinGecko/DEX)\n"
            f"--------------------\n"
            f"{contract_str}"
            f"**Giá:** {format_price(price)}\n"
            f"**Biến động:**\n"
            f"- 1 giờ:   {'📈' if p1h >= 0 else '📉'} `{p1h:+.2f}%`\n"
            f"- 24 giờ: {'📈' if p24h >= 0 else '📉'} `{p24h:+.2f}%`\n"
            f"- 7 ngày:  {'📈' if p7d >= 0 else '📉'} `{p7d:+.2f}%`\n"
            f"--------------------\n"
            f"**ATH:** {format_price(ath)} (`{ath_change:+.2f}%`)\n"
            f"- Ngày ATH: `{ath_date}`\n"
            f"--------------------\n"
            f"**Thống kê:**\n"
            f"- Vốn hóa TT: `${int(market_cap):,}`\n"
            f"- Khối lượng (24h): `${int(volume_24h):,}`\n"
            f"- Lưu thông: `{int(circulating_supply):,}`\n"
            f"- Tổng cung: `{int(total_supply) if total_supply else 'N/A'}`\n"
        )
        if genesis_date:
            message += f"- Ngày list alpha: `{genesis_date}`\n"
        
        message += f"\n_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"

        print("Lấy dữ liệu CoinGecko thành công.")
        return message
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu từ CoinGecko: {e}")
        return f"NOT_FOUND_ON_CG"

def get_dexscreener_data(contract_address: str) -> str:
    url = f"{DEXSCREENER_API_URL}/search?q={contract_address}"
    print(f"Fallback: Đang lấy dữ liệu từ DexScreener cho contract: {contract_address}")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        if not data or not data.get('pairs'):
            return "NOT_FOUND_ON_DEXSCREENER"
        pairs = sorted(data['pairs'], key=lambda p: p.get('liquidity', {}).get('usd', 0), reverse=True)
        best_pair = pairs[0]
        name = best_pair['baseToken'].get('name', 'N/A')
        symbol = best_pair['baseToken'].get('symbol', 'N/A')
        price_usd = float(best_pair.get('priceUsd', '0'))
        price_change = best_pair.get('priceChange', {})
        p5m = price_change.get('m5', 0.0)
        p1h = price_change.get('h1', 0.0)
        p24h = price_change.get('h24', 0.0)
        volume_24h = best_pair.get('volume', {}).get('h24', 0)
        liquidity = best_pair.get('liquidity', {}).get('usd', 0)
        listing_timestamp = best_pair.get('pairCreatedAt', 0) / 1000
        listing_date = datetime.fromtimestamp(listing_timestamp).strftime('%Y-%m-%d %H:%M:%S') if listing_timestamp else "N/A"
        message = (
            f"🪙 **{name} ({symbol})** (Nguồn: DexScreener)\n"
            f"--------------------\n"
            f"**Contract:**\n`{contract_address}`\n"
            f"--------------------\n"
            f"**Giá:** {format_price(price_usd)}\n"
            f"**Biến động:**\n"
            f"- 5 phút:   {'📈' if p5m >= 0 else '📉'} `{p5m:+.2f}%`\n"
            f"- 1 giờ:    {'📈' if p1h >= 0 else '📉'} `{p1h:+.2f}%`\n"
            f"- 24 giờ:  {'📈' if p24h >= 0 else '📉'} `{p24h:+.2f}%`\n"
            f"--------------------\n"
            f"**Thống kê:**\n"
            f"- Thanh khoản: `${int(liquidity):,}`\n"
            f"- Khối lượng (24h): `${int(volume_24h):,}`\n"
            f"- Ngày list alpha: `{listing_date}`\n\n"
            f"_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        )
        print("Lấy dữ liệu DexScreener thành công.")
        return message
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu từ DexScreener: {e}")
        return "NOT_FOUND_ON_DEXSCREENER"

async def get_current_price(symbol: str, exchange_id: str = None) -> tuple[float | None, str | None]:
    print(f"Đang lấy giá cho ký hiệu: {symbol}")
    stablecoins = ['USDT', 'USDC', 'FDUSD', 'TUSD', 'USD']

    if exchange_id:
        if exchange_id not in ccxt.exchanges:
            return None, f"Sàn `{exchange_id}` không được hỗ trợ."
        exchange = getattr(ccxt, exchange_id)()
        try:
            for stable in stablecoins:
                trading_pair = f"{symbol.upper()}/{stable}"
                ticker = await exchange.fetch_ticker(trading_pair)
                if ticker and ticker.get('last'):
                    return ticker['last'], exchange.name
        except Exception:
            pass 
        finally:
            await exchange.close()
        return None, f"Không tìm thấy cặp giao dịch cho `{symbol.upper()}` trên sàn {exchange.name}."

    for ex_id in SUPPORTED_EXCHANGES:
        exchange = getattr(ccxt, ex_id)()
        try:
            for stable in stablecoins:
                trading_pair = f"{symbol.upper()}/{stable}"
                ticker = await exchange.fetch_ticker(trading_pair)
                if ticker and ticker.get('last'):
                    print(f"Tìm thấy giá trên {exchange.name}")
                    return ticker['last'], exchange.name
        except Exception:
            continue
        finally:
            await exchange.close()
    
    coin_id = get_coingecko_id_from_symbol(symbol)
    if coin_id:
        try:
            url = f"{COINGECKO_API_URL}/simple/price?ids={coin_id}&vs_currencies=usd"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                price = data.get(coin_id, {}).get('usd')
                if price:
                    return float(price), "CoinGecko"
        except requests.RequestException:
            pass
            
    return None, "Không thể tìm thấy giá cho ký hiệu này trên mọi nguồn."

async def get_exchange_specific_data(symbol: str, exchange_id: str) -> str:
    """
    Lấy dữ liệu giá chi tiết, bao gồm cả ATH, từ một sàn giao dịch cụ thể.
    """
    print(f"Đang lấy dữ liệu chi tiết cho {symbol} từ sàn {exchange_id}...")
    stablecoins = ['USDT', 'USDC', 'FDUSD', 'TUSD', 'USD']
    
    if exchange_id not in ccxt.exchanges:
        return f"😕 Sàn `{exchange_id}` không được hỗ trợ hoặc gõ sai tên."

    exchange = getattr(ccxt, exchange_id)()
    try:
        await exchange.load_markets()
        trading_pair = None
        for stable in stablecoins:
            pair = f"{symbol.upper()}/{stable}"
            if pair in exchange.markets:
                trading_pair = pair
                break
        
        if not trading_pair:
            return f"😕 Không tìm thấy cặp giao dịch cho `{symbol.upper()}` trên sàn {exchange.name}."

        ticker = await exchange.fetch_ticker(trading_pair)
        current_price = ticker.get('last')

        # Lấy lịch sử giá khung D1 để tìm ATH
        limit = 2000 
        if exchange.has['fetchOHLCV']:
             ohlcv = await exchange.fetch_ohlcv(trading_pair, timeframe='1d', limit=limit)
        else:
             ohlcv = None
        ath_price = None
        ath_date = "N/A"

        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            if not df.empty and 'high' in df.columns:
                ath_row = df.loc[df['high'].idxmax()]
                ath_price = ath_row['high']
                ath_date = pd.to_datetime(ath_row['timestamp'], unit='ms').strftime('%Y-%m-%d')
            
        message = (
            f"🪙 **{trading_pair}** (Sàn: {exchange.name})\n"
            f"--------------------\n"
            f"**Giá hiện tại:** {format_price(current_price)}\n"
        )
        
        if ath_price:
            ath_change = ((current_price - ath_price) / ath_price) * 100 if ath_price and current_price else 0
            message += (
                f"**ATH trên sàn:** {format_price(ath_price)} (`{ath_change:.2f}%`)\n"
                f"- Ngày ATH: `{ath_date}`\n"
            )
        else:
             message += "**ATH trên sàn:** `Không đủ dữ liệu lịch sử`\n"

        
        message += f"\n_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        return message

    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu từ {exchange.name}: {e}")
        return f"🛠 Đã xảy ra lỗi khi kết nối đến sàn {exchange.name}."
    finally:
        await exchange.close()

def crawl_new_listings() -> str:
    """Crawl thông báo niêm yết token mới từ trang Binance Announcements."""
    url = "https://www.binance.com/en/support/announcement/new-crypto-listings"
    print("Đang crawl thông báo niêm yết mới từ Binance...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        announcement_links = soup.find_all('a', class_='css-1ej4hfo', limit=5)
        if not announcement_links:
            return "😕 Không thể crawl được danh sách token mới. Cấu trúc trang Binance có thể đã thay đổi."
        message_lines = ["📢 **Các thông báo niêm yết mới nhất trên Binance**\n"]
        base_url = "https://www.binance.com"
        for link in announcement_links:
            title = link.text.strip()
            href = link.get('href', '')
            full_url = href if href.startswith('http') else base_url + href
            message_lines.append(f"▪️ [{title}]({full_url})")
        return "\n".join(message_lines)
    except requests.exceptions.RequestException as e:
        print(f"Lỗi khi crawl trang thông báo: {e}")
        return "🛠 Lỗi kết nối đến trang thông báo của Binance."

async def get_trending_coins(limit: int = 7) -> str:
    """Lấy danh sách token trending từ CoinGecko."""
    print(f"Đang lấy Top {limit} token trending từ CoinGecko...")
    try:
        url = f"{COINGECKO_API_URL}/search/trending"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        coins = data.get('coins', [])
        
        if not coins:
            return "😕 Không thể lấy được danh sách trending."

        message_lines = [f"🔥 **Top {limit} Token Trending (CoinGecko)**\n"]
        tasks = []
        coin_info_list = []

        for i, item in enumerate(coins[:limit]):
            coin = item.get('item', {})
            symbol = coin.get('symbol', 'N/A').upper()
            name = coin.get('name', 'N/A')
            coin_info_list.append({'symbol': symbol, 'name': name})
            tasks.append(get_current_price(symbol))
        
        prices_sources = await asyncio.gather(*tasks)

        for i, info in enumerate(coin_info_list):
            price, _ = prices_sources[i] if prices_sources[i] else (None, None)
            price_str = format_price(price) if price is not None else "`N/A`"
            message_lines.append(f"{i+1}. **{info['name']}** ({info['symbol']}) - {price_str}")

        message_lines.append(f"\n_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
        return "\n".join(message_lines)

    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu trending: {e}")
        return "🛠 Đã xảy ra lỗi khi lấy dữ liệu trending."

async def get_market_data(order_by: str, limit: int = 10) -> str:
    """Lấy top gainers hoặc losers từ CoinGecko."""
    sort_param = 'market_cap_desc'
    title = f"Top {limit} Vốn hoá lớn nhất"
    if order_by == 'gainers':
        sort_param = 'price_change_percentage_24h_desc'
        title = f"📈 Top {limit} Token Tăng giá mạnh nhất (24h)"
    elif order_by == 'losers':
        sort_param = 'price_change_percentage_24h_asc'
        title = f"📉 Top {limit} Token Giảm giá mạnh nhất (24h)"

    print(f"Đang lấy {title} từ CoinGecko...")
    try:
        url = f"{COINGECKO_API_URL}/coins/markets?vs_currency=usd&order={sort_param}&per_page={limit}&page=1&sparkline=false&price_change_percentage=24h"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        if not data:
            return f"😕 Không thể lấy được danh sách {order_by}."

        message_lines = [f"**{title}**\n"]
        for i, coin in enumerate(data):
            name = coin.get('name', 'N/A')
            symbol = coin.get('symbol', 'N/A').upper()
            price = coin.get('current_price', 0)
            change_24h = coin.get('price_change_percentage_24h_in_currency', 0.0) or 0.0
            icon = '📈' if change_24h >= 0 else '📉'
            message_lines.append(f"{i+1}. **{name}** ({symbol}) - {format_price(price)} ({icon} `{change_24h:+.2f}%`)")
        
        message_lines.append(f"\n_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
        return "\n".join(message_lines)

    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu {order_by}: {e}")
        return f"🛠 Đã xảy ra lỗi khi lấy dữ liệu {order_by}."

async def get_historical_volume(symbol: str, date_str: str) -> str:
    """Lấy volume giao dịch lịch sử từ CoinGecko."""
    print(f"Đang lấy volume cho {symbol} vào ngày {date_str}...")
    
    try:
        target_date = datetime.strptime(date_str, '%Y%m%d').date()
        today = date.today()
        if target_date > today:
            return "⚠️ Không thể xem volume của ngày trong tương lai."
        if target_date == today:
             return "⚠️ Vui lòng sử dụng lệnh tra cứu giá thông thường (`$symbol`) để xem volume 24h hiện tại."

    except ValueError:
        return "⚠️ Định dạng ngày không hợp lệ. Vui lòng sử dụng `YYYYMMDD` (ví dụ: `20251024`)."

    coin_id = get_coingecko_id_from_symbol(symbol)
    if not coin_id:
        return f"😕 Không tìm thấy token có ký hiệu **{symbol.upper()}**."

    date_formatted_for_api = target_date.strftime('%d-%m-%Y')
    
    try:
        url = f"{COINGECKO_API_URL}/coins/{coin_id}/history?date={date_formatted_for_api}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        volume = data.get('market_data', {}).get('total_volume', {}).get('usd')
        
        if volume is not None:
             message = (
                 f"📊 **Volume {symbol.upper()}** (Nguồn: CoinGecko)\n"
                 f"--------------------\n"
                 f"Ngày: `{target_date.strftime('%Y-%m-%d')}`\n"
                 f"Volume (24h): `${int(volume):,}`\n\n"
                 f"_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
             )
             return message
        else:
             days_diff = (today - target_date).days
             if days_diff > 365 * 2: 
                 return f"⏳ Không có dữ liệu volume cho ngày quá xa trong quá khứ (`{date_str}`)."
             else:
                 return f"😕 Không tìm thấy dữ liệu volume cho **{symbol.upper()}** vào ngày `{date_str}`."

    except requests.exceptions.HTTPError as e:
         if e.response.status_code == 404 or e.response.status_code == 400:
             return f"😕 Không tìm thấy dữ liệu volume cho **{symbol.upper()}** vào ngày `{date_str}`."
         print(f"Lỗi HTTP khi lấy volume lịch sử: {e}")
         return f"🛠 Đã xảy ra lỗi khi lấy volume lịch sử."
    except Exception as e:
        print(f"Lỗi khi lấy volume lịch sử: {e}")
        return f"🛠 Đã xảy ra lỗi khi lấy volume lịch sử."

async def get_total_volume(symbol: str) -> str:
    """Tính tổng volume giao dịch từ CoinGecko."""
    print(f"Đang tính tổng volume cho {symbol}...")
    coin_id = get_coingecko_id_from_symbol(symbol)
    if not coin_id:
        return f"😕 Không tìm thấy token có ký hiệu **{symbol.upper()}**."

    try:
        url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart?vs_currency=usd&days=max&interval=daily"
        response = requests.get(url, timeout=30) 
        response.raise_for_status()
        data = response.json()
        
        volumes = data.get('total_volumes', [])
        if not volumes:
            return f"😕 Không có dữ liệu volume lịch sử cho **{symbol.upper()}**."

        total_vol = sum(item[1] for item in volumes if len(item) > 1)
        start_date = datetime.fromtimestamp(volumes[0][0]/1000).strftime('%Y-%m-%d') if volumes else 'N/A'
        
        info_url = f"{COINGECKO_API_URL}/coins/{coin_id}"
        info_response = requests.get(info_url, timeout=10)
        name = info_response.json().get('name', symbol.upper())

        message = (
            f"📊 **Tổng Volume {name} ({symbol.upper()})** (Nguồn: CoinGecko)\n"
            f"--------------------\n"
            f"Từ ngày: `{start_date}`\n"
            f"Tổng Volume: `${int(total_vol):,}`\n\n"
            f"_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        )
        return message

    except Exception as e:
        print(f"Lỗi khi tính tổng volume: {e}")
        return f"🛠 Đã xảy ra lỗi khi tính tổng volume."


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = (
        "👋 **Chào bạn! Tôi là Crypto Bot Pro.**\n\n"
        "Gửi cho tôi ký hiệu (ví dụ: `$BTC`) hoặc địa chỉ contract của một token để nhận dữ liệu giá.\n\n"
        "**Lệnh có sẵn:**\n"
        "`/p <ký hiệu> [sàn]` - Tra giá & ATH theo sàn.\n"
        "`/ch <ký hiệu> [khung giờ] [theme]` - Xem biểu đồ (mặc định 1h, dark).\n"
        "`/ath <ký hiệu> [sàn]` - Tra cứu ATH của token.\n"
        "`/cal <ký hiệu> <số lượng>` - Tính toán giá trị token.\n"
        "`/val <phép tính>` - Máy tính cơ bản.\n"
        "`/trending [số lượng]` - Xem token đang trending.\n"
        "`/vol <ký hiệu> [YYYYMMDD]` - Xem volume (mặc định: tổng volume).\n"
        "`/buy [số lượng]` - Xem top token tăng giá 24h.\n"
        "`/sell [số lượng]` - Xem top token giảm giá 24h.\n"
        "`/newlistings` - Xem các thông báo niêm yết token mới nhất."
    )
    message = update.message or update.channel_post
    if message:
        await message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)

async def new_listings_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message or update.channel_post
    if not message: return
    thinking_message = await message.reply_text("⏳ Đang lấy danh sách niêm yết mới...")
    listings_message = crawl_new_listings()
    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=thinking_message.message_id, text=listings_message, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)

async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lấy giá của một token từ một sàn cụ thể hoặc dò tìm."""
    message = update.message or update.channel_post
    if not message: return
    
    parts = message.text.strip().split()
    command_parts = parts[1:] if parts and parts[0].lower() in ['/p', 'p'] else []

    if not 1 <= len(command_parts) <= 2:
        await message.reply_text("⚠️ Cú pháp không đúng. Sử dụng: `/p <ký hiệu> [sàn]`\nVí dụ: `/p btc` hoặc `/p btc binance`", parse_mode=ParseMode.MARKDOWN)
        return
    
    symbol = command_parts[0]
    if not VALID_SYMBOL_PATTERN.match(symbol):
        await message.reply_text("⚠️ Ký hiệu token không hợp lệ (chỉ chứa chữ cái, số).", parse_mode=ParseMode.MARKDOWN)
        return
        
    exchange_id = command_parts[1].lower() if len(command_parts) > 1 else None

    thinking_message = await message.reply_text(f"Đang tìm giá của {symbol.upper()}...")
    
    if exchange_id:
        result_text = await get_exchange_specific_data(symbol, exchange_id)
    else:
        price, source = await get_current_price(symbol)
        if price is not None:
            result_text = (
                f"🪙 **{symbol.upper()}** (Nguồn: {source})\n"
                f"**Giá:** {format_price(price)}"
            )
        else:
            result_text = f"😕 {source}"

    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=thinking_message.message_id, text=result_text, parse_mode=ParseMode.MARKDOWN)


async def chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Vẽ và gửi biểu đồ nến cho một token."""
    message = update.message or update.channel_post
    if not message: return
    
    text_parts = message.text.strip().split()
    command_parts = text_parts[1:] if text_parts and text_parts[0].lower() in ['/ch', 'ch'] else []
    
    supported_intervals = ['5m', '15m', '1h', '4h', '1d']
    
    if not command_parts:
        await message.reply_text("⚠️ Cú pháp không đúng. Sử dụng: `/ch <ký hiệu> [khung giờ] [theme]`", parse_mode=ParseMode.MARKDOWN)
        return

    user_input = command_parts[0]
    if not VALID_SYMBOL_PATTERN.match(user_input):
        await message.reply_text("⚠️ Ký hiệu token không hợp lệ (chỉ chứa chữ cái, số).", parse_mode=ParseMode.MARKDOWN)
        return

    interval = '1h'
    theme = 'dark'

    if len(command_parts) > 1:
        arg2 = command_parts[1].lower()
        if arg2 in supported_intervals:
            interval = arg2
            if len(command_parts) > 2 and command_parts[2].lower() in ['light', 'dark']:
                theme = command_parts[2].lower()
        elif arg2 in ['light', 'dark']:
            theme = arg2
        else:
            await message.reply_text(f"⚠️ Khung giờ không hợp lệ: `{arg2}`.", parse_mode=ParseMode.MARKDOWN)
            return

    thinking_message = await message.reply_text(f"⏳ Đang vẽ biểu đồ cho {user_input.upper()} khung {interval}...")
    
    chart_buffer, trading_pair = await generate_ccxt_chart(user_input, interval, theme)

    if not chart_buffer:
        print(f"Không có trên các sàn CEX, đang thử CoinGecko...")
        coin_id = get_coingecko_id_from_symbol(user_input)
        if coin_id:
            chart_buffer, trading_pair = generate_coingecko_chart(coin_id, interval, theme)

    await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=thinking_message.message_id)

    if chart_buffer:
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=chart_buffer,
            caption=f"Biểu đồ nến cho **{trading_pair}** khung **{interval}**\n_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await message.reply_text(f"😕 Không thể tạo biểu đồ. Token **{user_input.upper()}** có thể không có đủ dữ liệu lịch sử.", parse_mode=ParseMode.MARKDOWN)

async def calculate_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Tính toán giá trị của một lượng token nhất định."""
    message = update.message or update.channel_post
    if not message: return
    text_parts = message.text.strip().split()
    if text_parts and text_parts[0].lower() in ['/cal', 'cal']:
        command_parts = text_parts[1:]
    else:
        command_parts = text_parts
    if len(command_parts) != 2:
        await message.reply_text("⚠️ Cú pháp không đúng. Sử dụng: `/cal <ký hiệu> <số lượng>`\nVí dụ: `/cal bnb 2`", parse_mode=ParseMode.MARKDOWN)
        return
        
    symbol = command_parts[0]
    if not VALID_SYMBOL_PATTERN.match(symbol):
        await message.reply_text("⚠️ Ký hiệu token không hợp lệ (chỉ chứa chữ cái, số).", parse_mode=ParseMode.MARKDOWN)
        return
        
    try:
        amount = float(command_parts[1])
    except ValueError:
        await message.reply_text("⚠️ Số lượng không hợp lệ. Vui lòng nhập một con số.")
        return
    thinking_message = await message.reply_text(f"Đang tìm giá của {symbol.upper()}...")
    price, source = await get_current_price(symbol)
    if price is not None:
        total_value = price * amount
        result_text = (
            f"💰 **Kết quả tính toán**\n\n"
            f"`{amount:g}` **{symbol.upper()}** ≈ `{format_price(total_value)}`\n\n"
            f"_Dựa trên giá hiện tại là `{format_price(price)}` / {symbol.upper()}_"
        )
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=thinking_message.message_id,
            text=result_text,
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=thinking_message.message_id,
            text=f"😕 Không thể tìm thấy giá cho token **{symbol.upper()}**.",
            parse_mode=ParseMode.MARKDOWN
        )

async def value_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Đánh giá một biểu thức toán học cơ bản."""
    message = update.message or update.channel_post
    if not message: return
    
    text_parts = message.text.strip().split()
    if text_parts and text_parts[0].lower() in ['/val', 'val']:
        expression = "".join(text_parts[1:])
    else:
        return

    if not expression:
        await message.reply_text("⚠️ Cú pháp không đúng. Sử dụng: `/val <phép tính>`\nVí dụ: `/val 2+3*4/5`", parse_mode=ParseMode.MARKDOWN)
        return
        
    print(f"Đang tính toán biểu thức: {expression}")
    result = safe_eval(expression)
    
    if result is not None:
        result_text = (
            f"🧮 **Kết quả:**\n\n"
            f"`{expression}` = `{result:g}`"
        )
        await message.reply_text(result_text, parse_mode=ParseMode.MARKDOWN)
    else:
        await message.reply_text(f"⚠️ Biểu thức không hợp lệ hoặc chứa các ký tự không được phép.", parse_mode=ParseMode.MARKDOWN)

async def ath_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Tra cứu ATH của một token."""
    message = update.message or update.channel_post
    if not message: return
    
    text_parts = message.text.strip().split()
    command_parts = text_parts[1:] if text_parts and text_parts[0].lower() in ['/ath', 'ath'] else []

    if not 1 <= len(command_parts) <= 2:
        await message.reply_text("⚠️ Cú pháp không đúng. Sử dụng: `/ath <ký hiệu> [sàn]`", parse_mode=ParseMode.MARKDOWN)
        return

    symbol = command_parts[0]
    if not VALID_SYMBOL_PATTERN.match(symbol):
        await message.reply_text("⚠️ Ký hiệu token không hợp lệ (chỉ chứa chữ cái, số).", parse_mode=ParseMode.MARKDOWN)
        return
        
    exchange_id = command_parts[1].lower() if len(command_parts) > 1 else None

    thinking_message = await message.reply_text(f"Đang tra cứu ATH cho {symbol.upper()}...")

    if exchange_id:
        result_text = await get_exchange_specific_data(symbol, exchange_id)
    else:
        coin_id = get_coingecko_id_from_symbol(symbol)
        if coin_id:
            full_data = get_coingecko_price_data(coin_id)
            if "NOT_FOUND_ON_CG" not in full_data:
                lines = full_data.split('\n')
                name_line_list = [line for line in lines if line.startswith("🪙")]
                ath_lines = [line for line in lines if "ATH:" in line or "Ngày ATH:" in line]
                price_line_list = [line for line in lines if line.startswith("**Giá:**")]

                if name_line_list and price_line_list and ath_lines:
                     result_text = f"{name_line_list[0]}\n--------------------\n{price_line_list[0]}\n" + "\n".join(ath_lines)
                else: 
                     result_text = full_data 
            else:
                result_text = f"😕 Không tìm thấy dữ liệu ATH cho **{symbol.upper()}** trên CoinGecko."
        else:
            result_text = f"😕 Không tìm thấy token có ký hiệu **{symbol.upper()}**."
            
    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=thinking_message.message_id, text=result_text, parse_mode=ParseMode.MARKDOWN)

async def trending_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lấy danh sách token trending."""
    message = update.message or update.channel_post
    if not message: return
    
    parts = message.text.strip().split()
    limit = 7 
    if len(parts) > 1:
        try:
            limit = int(parts[1])
            if not 1 <= limit <= 20: 
                limit = 7
        except ValueError:
            pass 

    thinking_message = await message.reply_text("⏳ Đang lấy danh sách trending...")
    trending_message = await get_trending_coins(limit)
    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=thinking_message.message_id, text=trending_message, parse_mode=ParseMode.MARKDOWN)

async def volume_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lấy volume lịch sử hoặc tổng volume."""
    message = update.message or update.channel_post
    if not message: return

    parts = message.text.strip().split()
    command_parts = parts[1:] if parts and parts[0].lower() in ['/vol', 'vol'] else []

    if not 1 <= len(command_parts) <= 2:
        await message.reply_text("⚠️ Cú pháp không đúng. Sử dụng: `/vol <ký hiệu> [YYYYMMDD]`\nVí dụ: `/vol btc` hoặc `/vol btc 20251024`", parse_mode=ParseMode.MARKDOWN)
        return
        
    symbol = command_parts[0]
    if not VALID_SYMBOL_PATTERN.match(symbol):
        await message.reply_text("⚠️ Ký hiệu token không hợp lệ (chỉ chứa chữ cái, số).", parse_mode=ParseMode.MARKDOWN)
        return
        
    date_str = command_parts[1] if len(command_parts) == 2 else None

    if date_str:
        if not DATE_PATTERN.match(date_str):
            await message.reply_text("⚠️ Định dạng ngày không hợp lệ. Vui lòng sử dụng `YYYYMMDD` (ví dụ: `20251024`).", parse_mode=ParseMode.MARKDOWN)
            return
        thinking_message = await message.reply_text(f"⏳ Đang lấy volume cho {symbol.upper()} vào ngày {date_str}...")
        volume_message = await get_historical_volume(symbol, date_str)
    else:
        thinking_message = await message.reply_text(f"⏳ Đang tính tổng volume cho {symbol.upper()}...")
        volume_message = await get_total_volume(symbol)

    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=thinking_message.message_id, text=volume_message, parse_mode=ParseMode.MARKDOWN)

async def get_total_volume(symbol: str) -> str:
    """Tính tổng volume giao dịch từ CoinGecko."""
    print(f"Đang tính tổng volume cho {symbol}...")
    coin_id = get_coingecko_id_from_symbol(symbol)
    if not coin_id:
        return f"😕 Không tìm thấy token có ký hiệu **{symbol.upper()}**."

    try:
        url = f"{COINGECKO_API_URL}/coins/{coin_id}/market_chart?vs_currency=usd&days=max&interval=daily"
        response = requests.get(url, timeout=30) 
        response.raise_for_status()
        data = response.json()
        
        volumes = data.get('total_volumes', [])
        if not volumes:
            return f"😕 Không có dữ liệu volume lịch sử cho **{symbol.upper()}**."

        total_vol = sum(item[1] for item in volumes if len(item) > 1)
        start_date = datetime.fromtimestamp(volumes[0][0]/1000).strftime('%Y-%m-%d') if volumes else 'N/A'
        
        info_url = f"{COINGECKO_API_URL}/coins/{coin_id}"
        info_response = requests.get(info_url, timeout=10)
        name = info_response.json().get('name', symbol.upper())

        message = (
            f"📊 **Tổng Volume {name} ({symbol.upper()})** (Nguồn: CoinGecko)\n"
            f"--------------------\n"
            f"Từ ngày: `{start_date}`\n"
            f"Tổng Volume: `${int(total_vol):,}`\n\n"
            f"_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
        )
        return message

    except Exception as e:
        print(f"Lỗi khi tính tổng volume: {e}")
        return f"🛠 Đã xảy ra lỗi khi tính tổng volume."


async def buy_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lấy top gainers."""
    message = update.message or update.channel_post
    if not message: return
    
    parts = message.text.strip().split()
    limit = 10
    if len(parts) > 1:
        try:
            limit = int(parts[1])
            if not 1 <= limit <= 20:
                limit = 10
        except ValueError:
            pass

    thinking_message = await message.reply_text("⏳ Đang lấy danh sách top tăng giá...")
    gainers_message = await get_market_data('gainers', limit)
    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=thinking_message.message_id, text=gainers_message, parse_mode=ParseMode.MARKDOWN)

async def sell_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lấy top losers."""
    message = update.message or update.channel_post
    if not message: return
    
    parts = message.text.strip().split()
    limit = 10
    if len(parts) > 1:
        try:
            limit = int(parts[1])
            if not 1 <= limit <= 20:
                limit = 10
        except ValueError:
            pass

    thinking_message = await message.reply_text("⏳ Đang lấy danh sách top giảm giá...")
    losers_message = await get_market_data('losers', limit)
    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=thinking_message.message_id, text=losers_message, parse_mode=ParseMode.MARKDOWN)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = update.message or update.channel_post
    if not message or not message.text: 
        return

    user_input = message.text.strip()
    user_input_lower = user_input.lower()

    # --- ROUTER for commands ---
    command_map = {
        '/start': start_command,
        '/newlistings': new_listings_command,
        '/ch': chart_command,
        '/p': price_command,
        '/cal': calculate_command,
        '/val': value_command,
        '/ath': ath_command,
        '/trending': trending_command,
        '/vol': volume_command,
        '/buy': buy_command,
        '/sell': sell_command,
        # Handle commands without '/' prefix
        'ch ': chart_command, 
        'p ': price_command,
        'cal ': calculate_command,
        'val ': value_command,
        'ath ': ath_command,
        'trending ': trending_command,
        'vol ': volume_command,
        'buy ': buy_command,
        'sell ': sell_command,
    }

    for prefix, handler in command_map.items():
        if user_input_lower.startswith(prefix):
            return await handler(update, context)

    # --- PRICE LOOKUP HANDLER ---
    is_contract = is_contract_address(user_input)
    is_dollar_prefixed_symbol = (
        user_input.startswith('$') and 
        len(user_input) > 1 and 
        len(user_input.split()) == 1 and # Chỉ một từ
        VALID_SYMBOL_PATTERN.match(user_input[1:]) # Kiểm tra ký hiệu hợp lệ
    )
    is_dollar_prefixed_symbol_exchange = (
        user_input.startswith('$') and
        len(user_input.split()) == 2 and # Hai từ: $symbol exchange
        VALID_SYMBOL_PATTERN.match(user_input.split()[0][1:]) and # Kiểm tra symbol
        user_input.split()[1].lower() in ccxt.exchanges # Kiểm tra tên sàn hợp lệ
    )


    if not (is_contract or is_dollar_prefixed_symbol or is_dollar_prefixed_symbol_exchange):
        print(f"Bỏ qua tin nhắn (không phải lệnh hoặc tra cứu giá hợp lệ): '{user_input}'")
        return

    thinking_message = await message.reply_text("🔍 Đang xử lý yêu cầu, vui lòng chờ...")
    
    final_message = ""
    
    if is_contract:
        identifier = user_input
        final_message = get_dexscreener_data(identifier)
        if "NOT_FOUND_ON_DEXSCREENER" in final_message:
            token_info = get_info_from_contract(identifier)
            if token_info:
                final_message = get_binance_price_data(token_info['symbol'])
                if "NOT_FOUND_ON_BINANCE" in final_message:
                    final_message = get_coingecko_price_data(token_info['id'], known_contract_info=token_info)
            else:
                final_message = f"😕 Không thể tìm thấy token tương ứng với địa chỉ contract này trên mọi nguồn dữ liệu."
    
    elif is_dollar_prefixed_symbol_exchange:
        parts = user_input.split()
        symbol = parts[0][1:]
        exchange_id = parts[1].lower()
        final_message = await get_exchange_specific_data(symbol, exchange_id)
        
    elif is_dollar_prefixed_symbol: 
        identifier = user_input[1:]
        final_message = get_binance_price_data(identifier)
        if "NOT_FOUND_ON_BINANCE" in final_message:
            coin_id = get_coingecko_id_from_symbol(identifier)
            if coin_id:
                final_message = get_coingecko_price_data(coin_id)
            else:
                final_message = f"😕 Không tìm thấy token có ký hiệu **{identifier.upper()}** trên các nguồn được hỗ trợ."
    
    await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=thinking_message.message_id, text=final_message, parse_mode=ParseMode.MARKDOWN)

def main():
    print("Bắt đầu khởi tạo bot Telegram...")
    application = (
        Application.builder()
        .token(BOT_TOKEN)
        .connect_timeout(30)
        .read_timeout(60) # Tăng read timeout
        .get_updates_read_timeout(60) # Tăng timeout cho getUpdates
        .pool_timeout(60) # Tăng pool timeout
        .build()
    )
    
    # Sử dụng một handler duy nhất cho tất cả các tin nhắn
    application.add_handler(MessageHandler(filters.TEXT | filters.COMMAND, handle_message))

    print("Bot đã sẵn sàng và đang lắng nghe...")
    application.run_polling(allowed_updates=Update.ALL_TYPES) # Đảm bảo nhận mọi loại update

if __name__ == '__main__':
    print("\n===================================")
    print("===>      CRYPTO BOT ĐANG CHẠY     <===")
    print("===> (Nhấn Ctrl+C để dừng bot) <===")
    print("===================================\n")
    main()