"""
Price Handler for Binance - Optimized for Speed & Multi-user Cache
Fetches and displays cryptocurrency prices, gainers, losers, and volume leaders
✅ Tek “/ticker/24h” toplu çağrı	Tüm coin verisi tek API çağrısında alınır.
✅ Global cache (endpoint-level)	Aynı 60sn içinde tekrar istek yapılmaz.
✅ orjson JSON hızlandırıcı	Daha hızlı parsing.
✅ heapq ile sıralama	sort() yerine nlargest/nsmallest.
✅ Regex precompile	normalize_symbol çok daha hızlı.
✅ Async semaphore	Rate limit dostu.
✅ Daha az string concat	Formatlar property’e taşındı.

# orjson
data = orjson.loads(json_string)

# yerine
import json
data = json.loads(json_string)

performans hedefleniyorsa orjson daha iyi bir tercihtir.
"""

import asyncio
import re
import heapq
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from functools import wraps
import orjson

from aiogram import Router, F, Bot
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.enums import ChatAction

from utils.context_logger import get_context_logger
from utils.binance_api.binance_a import BinanceAggregator

# ============================================================
# CONFIGURATION
# ============================================================

logger = get_context_logger(__name__)
router = Router(name="price_handler")

SCAN_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "PEPEUSDT",
    "ARPAUSDT", "FETUSDT", "SUSDT", "ALICEUSDT", "TRXUSDT"
]

SCAN_DEFAULT = 20
CACHE_TTL = 60  # seconds

# ============================================================
# GLOBAL CACHE (endpoint-level)
# ============================================================

_cache_lock = asyncio.Lock()
_price_cache: Dict[str, tuple[Any, float]] = {}  # key -> (data, timestamp)

# ============================================================
# REGEX PRECOMPILE
# ============================================================

_SYMBOL_RE = re.compile(r'[^A-Z0-9]')

# ============================================================
# DATA CONTAINER
# ============================================================

@dataclass
class CoinData:
    symbol: str
    price: float
    change_percent: float
    volume: float

    @property
    def formatted_price(self) -> str:
        if self.price >= 1000:
            return f"{self.price:,.0f}"
        elif self.price >= 1:
            return f"{self.price:,.2f}"
        elif self.price >= 0.01:
            return f"{self.price:.4f}"
        else:
            return f"{self.price:.8f}".rstrip('0').rstrip('.')

    @property
    def formatted_volume(self) -> str:
        v = self.volume
        if v >= 1_000_000_000:
            return f"${v/1_000_000_000:.1f}B"
        elif v >= 1_000_000:
            return f"${v/1_000_000:.1f}M"
        elif v >= 1_000:
            return f"${v/1_000:.1f}K"
        else:
            return f"${v:.0f}"

    @property
    def formatted_change(self) -> str:
        emoji = "🟢" if self.change_percent > 0 else "🔴" if self.change_percent < 0 else "⚪"
        return f"{emoji} {abs(self.change_percent):.2f}%"


# ============================================================
# CACHE DECORATOR
# ============================================================

def cache_decorator(ttl: int = CACHE_TTL):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = func.__name__
            now = asyncio.get_event_loop().time()
            async with _cache_lock:
                if key in _price_cache:
                    data, ts = _price_cache[key]
                    if now - ts < ttl:
                        return data

            data = await func(*args, **kwargs)
            async with _cache_lock:
                _price_cache[key] = (data, now)
            return data
        return wrapper
    return decorator


# ============================================================
# MAIN CLASS
# ============================================================

class PriceHandler:
    def __init__(self):
        self.binance = None
        self._initialized = False
        self._sem = asyncio.Semaphore(5)  # API concurrency limit

    async def initialize(self):
        if not self._initialized:
            self.binance = BinanceAggregator.get_instance()
            self._initialized = True
            logger.info("✅ PriceHandler initialized")

    def normalize_symbol(self, symbol: str) -> str:
        s = _SYMBOL_RE.sub('', symbol.upper().strip())
        return s if s.endswith('USDT') else s + 'USDT'

    @cache_decorator(ttl=CACHE_TTL)
    async def get_all_tickers(self, user_id: int) -> List[Dict[str, Any]]:
        """Fetch all tickers once and cache"""
        async with self._sem:
            try:
                data = await self.binance.get_data(user_id, "get_ticker_24h")
                return data or []
            except Exception as e:
                logger.error(f"Error fetching all tickers: {e}")
                return []

    async def get_filtered_tickers(self, user_id: int, symbols: List[str]) -> List[CoinData]:
        """Return only requested symbols"""
        all_tickers = await self.get_all_tickers(user_id)
        symbol_set = set(symbols)
        result = []
        for t in all_tickers:
            s = t.get("symbol")
            if s in symbol_set:
                try:
                    c = CoinData(
                        s,
                        float(t.get("lastPrice", 0)),
                        float(t.get("priceChangePercent", 0)),
                        float(t.get("volume", 0)),
                    )
                    result.append(c)
                except Exception:
                    continue
        return result

    async def get_top_by(self, user_id: int, key: str, reverse=True, limit=20) -> List[CoinData]:
        """Generic sorter: key in {'priceChangePercent', 'volume'}"""
        all_tickers = await self.get_all_tickers(user_id)
        filtered = []
        for t in all_tickers:
            if not t["symbol"].endswith("USDT"):
                continue
            try:
                c = CoinData(
                    t["symbol"],
                    float(t.get("lastPrice", 0)),
                    float(t.get("priceChangePercent", 0)),
                    float(t.get("volume", 0)),
                )
                filtered.append(c)
            except Exception:
                continue
        if not filtered:
            return []

        if key == "volume":
            return heapq.nlargest(limit, filtered, key=lambda x: x.volume)
        elif key == "gainers":
            return heapq.nlargest(limit, [c for c in filtered if c.change_percent > 0],
                                  key=lambda x: x.change_percent)
        elif key == "losers":
            return heapq.nsmallest(limit, [c for c in filtered if c.change_percent < 0],
                                   key=lambda x: x.change_percent)
        return []


# ============================================================
# GLOBAL INSTANCE
# ============================================================

price_handler = PriceHandler()


async def initialize_handler():
    await price_handler.initialize()


# ============================================================
# COMMANDS
# ============================================================

async def _send_coin_list(message: Message, title: str, coins: List[CoinData]):
    if not coins:
        await message.answer("❌ Veri bulunamadı.")
        return

    header = f"{title}\n⚡Coin | Değişim | Hacim | Fiyat\n"
    body = "\n".join(
        f"{i}. {c.symbol[:-4]}: {c.formatted_change} | {c.formatted_volume} | {c.formatted_price}"
        for i, c in enumerate(coins, 1)
    )
    await message.answer(header + body)


@router.message(Command("p"))
async def price_command(message: Message, bot: Bot):
    user_id = message.from_user.id
    args = message.text.split()[1:]
    symbols = [price_handler.normalize_symbol(a) for a in args] if args else SCAN_SYMBOLS

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    try:
        data = await price_handler.get_filtered_tickers(user_id, symbols)
        await _send_coin_list(message, "💰 **Coin Fiyatları**", data)
    except Exception as e:
        logger.error(f"Error in /p: {e}")
        await message.answer("❌ Veri alınırken bir hata oluştu.")


@router.message(Command("pg"))
async def gainers_command(message: Message, bot: Bot):
    user_id = message.from_user.id
    args = message.text.split()[1:]
    limit = min(int(args[0]) if args else SCAN_DEFAULT, 50)

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    try:
        data = await price_handler.get_top_by(user_id, "gainers", limit=limit)
        await _send_coin_list(message, f"📈 **En Çok Yükselen {len(data)} Coin**", data)
    except Exception as e:
        logger.error(f"Error in /pg: {e}")
        await message.answer("❌ Veri alınırken hata oluştu.")


@router.message(Command("pl"))
async def losers_command(message: Message, bot: Bot):
    user_id = message.from_user.id
    args = message.text.split()[1:]
    limit = min(int(args[0]) if args else SCAN_DEFAULT, 50)

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    try:
        data = await price_handler.get_top_by(user_id, "losers", limit=limit)
        await _send_coin_list(message, f"📉 **En Çok Düşen {len(data)} Coin**", data)
    except Exception as e:
        logger.error(f"Error in /pl: {e}")
        await message.answer("❌ Veri alınırken hata oluştu.")


@router.message(Command("pv"))
async def volume_command(message: Message, bot: Bot):
    user_id = message.from_user.id
    args = message.text.split()[1:]
    limit = min(int(args[0]) if args else SCAN_DEFAULT, 50)

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    try:
        data = await price_handler.get_top_by(user_id, "volume", limit=limit)
        await _send_coin_list(message, f"🔥 **En Yüksek Hacimli {len(data)} Coin**", data)
    except Exception as e:
        logger.error(f"Error in /pv: {e}")
        await message.answer("❌ Veri alınırken hata oluştu.")


# ============================================================
# STARTUP / SHUTDOWN HOOKS
# ============================================================

@router.startup()
async def on_startup():
    await initialize_handler()
    logger.info("✅ Price handler initialized successfully")

@router.shutdown()
async def on_shutdown():
    logger.info("🛑 Price handler shutdown")
