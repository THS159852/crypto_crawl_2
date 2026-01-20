# 🤖 Crypto Telegram Bot Pro

A high-performance Telegram bot for real-time cryptocurrency price tracking, candlestick charts, and market analysis.

## ✨ Features

- **Real-time Price Data** - Fetches prices from Binance, multiple CEX exchanges via CCXT, and DexScreener for DEX tokens
- **Candlestick Charts** - Beautiful charts with customizable themes (light/dark) and price axis on the right
- **Buy/Sell Volume Analysis** - Analyze taker buy vs sell volume from Binance
- **Market Statistics** - ATH (All-Time High), market cap, circulating supply from CoinGecko
- **Smart Caching** - TTL-based caching for fast response times
- **Async Architecture** - High concurrency with aiohttp and asyncio

## 📋 Requirements

- Python 3.10+
- Telegram Bot Token (from [@BotFather](https://t.me/BotFather))

## 🚀 Installation

### 1. Clone or Download

```bash
git clone <repository-url>
cd telegram_number_bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the project root:

```env
BOT_TOKEN=your_telegram_bot_token_here
```

### 4. Run the Bot

```bash
python bot.py
```

## 📖 Commands

| Command | Description | Example |
|---------|-------------|---------|
| `/start` | Show help message | `/start` |
| `/p <symbol>` | Get detailed price report | `/p btc` |
| `/ath <symbol>` | Get ATH and market info | `/ath eth` |
| `/ch <symbol> [timeframe] [theme]` | Generate candlestick chart | `/ch btc 4h dark` |
| `/<symbol> <timeframe>` | Analyze buy/sell volume | `/sui 1h` |
| `/cal <symbol> <amount>` | Calculate token value in USD | `/cal btc 0.5` |
| `/vol <symbol> [date]` | Get volume data | `/vol btc 20231225` |
| `/trending` | Show trending coins | `/trending` |
| `/buy` | Show top gainers (24h) | `/buy` |
| `/sell` | Show top losers (24h) | `/sell` |
| `/val <expression>` | Math calculator | `/val 100 * 2.5` |
| `$<symbol>` | Quick price lookup | `$btc` |
| `0x...` | Contract address lookup | `0x1234...` |

## 📊 Chart Options

### Timeframes
- `5m` - 5 minutes
- `15m` - 15 minutes  
- `1h` - 1 hour (default)
- `4h` - 4 hours
- `1d` - 1 day

### Themes
- `light` - Light background (default)
- `dark` - Dark background

### Examples

```
/ch btc           # BTC 1h chart, light theme
/ch eth 4h        # ETH 4h chart, light theme
/ch sol 1d dark   # SOL daily chart, dark theme
```

## 🔧 Architecture

### Data Sources (Priority Order)

1. **Binance** - Spot and Futures API (fastest, most liquid)
2. **CCXT Exchanges** - Gate.io, MEXC, KuCoin, OKX, Bybit
3. **DexScreener** - For DEX-only tokens
4. **CoinGecko** - For market data and fallback pricing

### Caching Strategy

| Cache | TTL | Purpose |
|-------|-----|---------|
| CoinGecko ID | 1 hour | Map symbols to CoinGecko IDs |
| Price Data | 10 seconds | Real-time price with minimal latency |
| Market Data | 60 seconds | Trending, gainers, losers |
| CoinGecko Info | 5 minutes | ATH, market cap, supply |

### Performance Optimizations

- **aiohttp** - Async HTTP client with connection pooling
- **Parallel Requests** - Fetch multiple data sources simultaneously
- **Early Return** - Stop searching when data is found
- **DNS Caching** - 5-minute DNS cache to reduce lookups

## 📁 Project Structure

```
telegram_number_bot/
├── bot.py              # Main bot script
├── requirements.txt    # Python dependencies
├── .env               # Environment variables (create this)
└── README.md          # This file
```

## 📦 Dependencies

```
python-telegram-bot>=21.7    # Telegram Bot API
aiohttp>=3.11.0              # Async HTTP client
cachetools>=5.5.0            # TTL caching
ccxt>=4.4.0                  # Crypto exchange library
pandas>=2.2.0                # Data manipulation
mplfinance>=0.12.10b0        # Candlestick charts
matplotlib>=3.9.0            # Plotting backend
python-dotenv>=1.0.0         # Environment variables
```

## 🔒 Security Notes

- Never commit your `.env` file or expose your `BOT_TOKEN`
- The `/val` command uses restricted `eval()` for safety
- Contract address validation prevents injection attacks

## 🐛 Troubleshooting

### Bot not responding
1. Check if `BOT_TOKEN` is set correctly
2. Ensure the bot is running (`python bot.py`)
3. Try `/start` command first

### Chart generation fails
1. Check if the symbol exists on supported exchanges
2. Try a different timeframe
3. Some tokens may not have OHLCV data

### Slow responses
1. First request may be slower (cold cache)
2. Subsequent requests should be faster due to caching
3. Check your internet connection

## 📄 License

MIT License - Feel free to use and modify.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

Made with ❤️ for the crypto community
