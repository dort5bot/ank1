"""
handlers/p22_handler.py
Price Handler for Binance - Optimized Multi-user Crypto Price Tracker
🔄 Multi-user support with user-specific caching
⚡ Ultra-fast with orjson, heapq, regex precompilation
💡 Smart fallback system (Aggregator → HTTP)
🕒 TTL cache with automatic cleanup
📊 Volume formatting (M/K/B)
💰 Smart price formatting
🔍 USDT pairs only with flexible symbol matching
"""

import os
import asyncio
import re
import heapq
import orjson

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from functools import wraps
from datetime import datetime, timedelta

from aiogram import Router, F, Bot
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.enums import ChatAction

from utils.context_logger import get_context_logger
from utils.binance_api.binance_a import BinanceAggregator

from config import config

import logging
logger = logging.getLogger(__name__)

# ============================================================
# HANDLER KENDİ CONFIG'İ
# ============================================================

# Handler'ın kendi sabit değişkenleri
SCAN_DEFAULT = 30      # Varsayılan coin sayısı
SCAN_MAX = 100         # Maksimum coin sayısı
CACHE_TTL = 60         # 60 saniye cache ömrü

# Sembol listesini config'den al
SCAN_SYMBOLS = config.SCAN_SYMBOLS
# ============================================================

# Helper function for dynamic symbol loading
def get_default_symbols(count: int = None) -> List[str]:
    """Get default symbols with optional count limit"""
    if count is None:
        count = SCAN_DEFAULT
    count = min(count, SCAN_MAX)
    return SCAN_SYMBOLS[:count]

# Pre-loaded default symbols for performance
DEFAULT_SYMBOLS = get_default_symbols()

# Logger ve Router
logger = get_context_logger(__name__)
router = Router(name="price_handler")




# ============================================================
# OPTIMIZED CACHE SYSTEM
# ============================================================

class MultiUserCache:
    """Multi-user TTL cache with automatic cleanup"""
    
    def __init__(self, ttl: int = 60):
        self.ttl = ttl
        self._cache: Dict[Tuple[int, str], Tuple[Any, datetime]] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, user_id: int, key: str) -> Optional[Any]:
        """Get cached data for user"""
        async with self._lock:
            cache_key = (user_id, key)
            if cache_key in self._cache:
                data, timestamp = self._cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                    return data
                else:
                    # Remove expired entry
                    del self._cache[cache_key]
            return None
    
    async def set(self, user_id: int, key: str, data: Any):
        """Set cached data for user"""
        async with self._lock:
            self._cache[(user_id, key)] = (data, datetime.now())
    
    async def cleanup(self):
        """Cleanup expired entries"""
        async with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if now - timestamp > timedelta(seconds=self.ttl)
            ]
            for key in expired_keys:
                del self._cache[key]
            if expired_keys:
                logger.debug(f"🧹 Cleaned {len(expired_keys)} expired cache entries")

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class CoinData:
    """Optimized coin data container"""
    symbol: str
    price: float
    change_percent: float
    volume: float
    
    @property
    def base_symbol(self) -> str:
        """Get base symbol without USDT"""
        return self.symbol.replace('USDT', '')
    
    @property
    def formatted_price(self) -> str:
        """Smart price formatting"""
        price = self.price
        if price >= 1000:
            return f"{price:,.0f}"
        elif price >= 1:
            return f"{price:,.2f}"
        elif price >= 0.01:
            return f"{price:.4f}"
        else:
            formatted = f"{price:.8f}"
            return formatted.rstrip('0').rstrip('.')
    
    @property
    def formatted_volume(self) -> str:
        """Volume formatting (K/M/B)"""
        vol = self.volume
        if vol >= 1_000_000_000:
            return f"${vol/1_000_000_000:.1f}B"
        elif vol >= 1_000_000:
            return f"${vol/1_000_000:.1f}M"
        elif vol >= 1_000:
            return f"${vol/1_000:.1f}K"
        else:
            return f"${vol:.0f}"
    
    @property
    def formatted_change(self) -> str:
        """Change percentage with emoji"""
        emoji = "🟢" if self.change_percent > 0 else "🔴" if self.change_percent < 0 else "⚪"
        return f"{emoji} {abs(self.change_percent):.2f}%"

# ============================================================
# MAIN PRICE HANDLER
# ============================================================

class PriceHandler:
    """Multi-user Binance price handler with fallback support"""
    
    def __init__(self):
        self.binance = None
        self._initialized = False
        self._semaphore = asyncio.Semaphore(5)
        self._cache = MultiUserCache(ttl=CACHE_TTL)
        self._symbol_regex = re.compile(r'[^A-Z0-9]')
        
        # ✅ DEBUG için ek bilgiler
        self._last_error = None
        self._aggregator_status = "not_initialized"    
    
        
    async def initialize(self):
        """Initialize Binance aggregator with proper error handling"""
        if self._initialized:
            return True
            
        try:
            logger.info("🔄 Initializing BinanceAggregator...")
            
            # ✅ DÜZELTME: BinanceAggregator'ı doğru şekilde başlat
            logger.info("🔧 Creating BinanceAggregator instance...")
            self.binance = await BinanceAggregator.get_instance()
            logger.info("✅ BinanceAggregator instance created")
            
            # ✅ Basit bir test yap - public endpoint ile
            logger.info("🔧 Testing aggregator basic functionality...")
            test_result = await self.binance.get_public_data("server_time")
            logger.info(f"✅ Aggregator test successful: {test_result}")
            
            self._initialized = True
            self._aggregator_status = "initialized"
            logger.info("✅ PriceHandler initialized successfully")
            return True
            
        except Exception as e:
            self._last_error = str(e)
            self._aggregator_status = f"error: {e}"
            logger.error(f"❌ PriceHandler initialization failed: {e}", exc_info=True)
            return False
            
 
    def normalize_symbol(self, symbol: str) -> str:
        """Fast symbol normalization"""
        cleaned = self._symbol_regex.sub('', symbol.upper().strip())
        return cleaned if cleaned.endswith('USDT') else cleaned + 'USDT'

        
    async def get_all_tickers(self, user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get limited tickers - CORRECTED VERSION"""
        
        cached_data = await self._cache.get(user_id, f"all_tickers_{limit}")
        if cached_data:
            return cached_data
        
        try:
            # ✅ DÜZELTME: ticker_24hr endpoint'ini kullan
            # BinanceAggregator'da bu endpoint "ticker_24hr" olarak tanımlı
            data = await self.binance.get_public_data("ticker_24hr")
            
            if data and isinstance(data, list):
                # USDT pair'lerini filtrele ve volume'a göre sırala
                usdt_pairs = [
                    t for t in data 
                    if isinstance(t, dict) and t.get('symbol', '').endswith('USDT')
                ]
                usdt_pairs.sort(key=lambda x: float(x.get('volume', 0)), reverse=True)
                limited_pairs = usdt_pairs[:limit]
                
                await self._cache.set(user_id, f"all_tickers_{limit}", limited_pairs)
                return limited_pairs
            else:
                logger.warning("❌ Aggregator returned empty or invalid data")
                return await self._http_fallback(user_id, limit)
                
        except Exception as e:
            logger.error(f"❌ Aggregator error in get_all_tickers: {e}")
            return await self._http_fallback(user_id, limit)
        

    async def _http_fallback(self, user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """HTTP fallback when aggregator fails"""
        try:
            import aiohttp
            async with aiohttp.ClientSession(json_serialize=orjson.dumps) as session:
                url = "https://api.binance.com/api/v3/ticker/24hr"
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = orjson.loads(await response.read())
                        usdt_pairs = [
                            t for t in data 
                            if isinstance(t, dict) and t.get('symbol', '').endswith('USDT')
                        ]
                        usdt_pairs.sort(key=lambda x: float(x.get('volume', 0)), reverse=True)
                        limited_pairs = usdt_pairs[:limit]
                        # Cache fallback result
                        await self._cache.set(user_id, f"all_tickers_{limit}", limited_pairs)
                        logger.info(f"✅ HTTP Fallback: {len(limited_pairs)} USDT pairs")
                        return limited_pairs
                    else:
                        logger.error(f"❌ HTTP Fallback Error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"❌ HTTP Fallback failed: {e}")
            return []
    
   
    async def get_filtered_tickers(self, user_id: int, symbols: List[str]) -> List[CoinData]:
        """Get specific symbols data - TÜM coin'ler arasından ara"""
        all_tickers = await self.get_all_tickers(user_id, limit=1000)  # Daha fazla coin getir
        symbol_set = set(symbols)
        result = []
        
        for ticker in all_tickers:
            symbol = ticker.get("symbol")
            if symbol in symbol_set:
                try:
                    coin_data = CoinData(
                        symbol=symbol,
                        price=float(ticker.get("lastPrice", 0)),
                        change_percent=float(ticker.get("priceChangePercent", 0)),
                        volume=float(ticker.get("volume", 0)),
                    )
                    result.append(coin_data)
                except (ValueError, TypeError) as e:
                    logger.debug(f"⚠️ Invalid data for {symbol}: {e}")
                    continue
        
        return result
       
    
    async def get_top_gainers(self, user_id: int, limit: int = SCAN_DEFAULT) -> List[CoinData]:
        """Get top gaining coins"""
        return await self._get_top_by_change(user_id, limit, positive=True)
    
    async def get_top_losers(self, user_id: int, limit: int = SCAN_DEFAULT) -> List[CoinData]:
        """Get top losing coins"""
        return await self._get_top_by_change(user_id, limit, positive=False)
    
    async def get_top_volume(self, user_id: int, limit: int = SCAN_DEFAULT) -> List[CoinData]:
        """Get top volume coins"""
        all_tickers = await self.get_all_tickers(user_id)
        coins = []
        
        for ticker in all_tickers:
            try:
                coin_data = CoinData(
                    symbol=ticker["symbol"],
                    price=float(ticker.get("lastPrice", 0)),
                    change_percent=float(ticker.get("priceChangePercent", 0)),
                    volume=float(ticker.get("volume", 0)),
                )
                coins.append(coin_data)
            except (ValueError, TypeError):
                continue
        
        return heapq.nlargest(limit, coins, key=lambda x: x.volume)
    
    async def _get_top_by_change(self, user_id: int, limit: int, positive: bool = True) -> List[CoinData]:
        """Generic method for gainers/losers"""
        all_tickers = await self.get_all_tickers(user_id)
        coins = []
        
        for ticker in all_tickers:
            try:
                change = float(ticker.get("priceChangePercent", 0))
                if (positive and change > 0) or (not positive and change < 0):
                    coin_data = CoinData(
                        symbol=ticker["symbol"],
                        price=float(ticker.get("lastPrice", 0)),
                        change_percent=change,
                        volume=float(ticker.get("volume", 0)),
                    )
                    coins.append(coin_data)
            except (ValueError, TypeError):
                continue
        
        key_func = lambda x: x.change_percent
        return heapq.nlargest(limit, coins, key=key_func) if positive else heapq.nsmallest(limit, coins, key=key_func)

# ============================================================
# GLOBAL INSTANCE & MESSAGE HANDLERS
# ============================================================

price_handler = PriceHandler()

async def initialize_handler():
    """Initialize price handler with retry logic"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            success = await price_handler.initialize()
            if success:
                logger.info("✅ PriceHandler initialized successfully")
                return
            else:
                logger.warning(f"⚠️ PriceHandler initialization failed on attempt {attempt + 1}")
        except Exception as e:
            logger.error(f"❌ PriceHandler initialization error on attempt {attempt + 1}: {e}")
        
        if attempt < max_retries - 1:
            logger.info(f"🔄 Retrying in {retry_delay} seconds...")
            await asyncio.sleep(retry_delay)
    
    logger.error("💥 PriceHandler initialization failed after all retries")

async def send_coin_list(message: Message, title: str, coins: List[CoinData]):
    """Send formatted coin list to user"""
    if not coins:
        await message.answer("❌ Veri bulunamadı.")
        return
    
    header = f"**{title}**\n⚡Coin | Değişim | Hacim | Fiyat\n"
    body_lines = []
    
    for i, coin in enumerate(coins, 1):
        line = f"{i}. {coin.base_symbol}: {coin.formatted_change} | {coin.formatted_volume} | {coin.formatted_price}"
        body_lines.append(line)
    
    # Split long messages if needed
    full_message = header + "\n".join(body_lines)
    if len(full_message) > 4000:
        # Send in chunks
        chunks = [full_message[i:i+4000] for i in range(0, len(full_message), 4000)]
        for chunk in chunks:
            await message.answer(chunk)
    else:
        await message.answer(full_message)

"""
/p komutundan beklenenler
bu 3 özelliği sağlamalı
/p  → default coinler
/p n → hacimli ilk n coin
/p btc → btc bilgisi
/p bnb sol btc → 3 coine ait bilgi
"""

"""1 @router.message(Command("p"))
async def price_command(message: Message, bot: Bot):
    user_id = message.from_user.id
    args = message.text.split()[1:]
    
    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    
    try:
        # Sayı kontrolü - /p 12
        if args and args[0].isdigit():
            limit = min(int(args[0]), SCAN_MAX)
            symbols = get_default_symbols(limit)  # ✅ Dinamik sembol yükleme
            data = await price_handler.get_filtered_tickers(user_id, symbols)
            title = f"💰 **İlk {len(data)} Coin**"
            await send_coin_list(message, title, data)
            
        # Belirli coin sorgusu - /p btc eth
        elif args:
            symbols = [price_handler.normalize_symbol(arg) for arg in args]
            data = await price_handler.get_filtered_tickers(user_id, symbols)
            title = f"💰 **Coin Fiyatları** ({len(data)} coin)"
            await send_coin_list(message, title, data)
            
        # Default semboller - /p
        else:
            symbols = DEFAULT_SYMBOLS  # ✅ Önceden yüklenmiş semboller
            data = await price_handler.get_filtered_tickers(user_id, symbols)
            title = f"💰 **Coin Fiyatları** ({len(data)} coin)"
            await send_coin_list(message, title, data)
            
    except Exception as e:
        logger.error(f"❌ Error in /p for user {user_id}: {e}")
        await message.answer("❌ Veri alınırken bir hata oluştu.")
"""


"""2 @router.message(Command("p"))
async def price_command(message: Message, bot: Bot):
    user_id = message.from_user.id
    args = message.text.split()[1:]

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    try:
        # 1️⃣ /p  → default coinler (hacme göre)
        if not args:
            limit = SCAN_DEFAULT
            data = await price_handler.get_top_volume(user_id, limit)

            title = f"🔥 **Binance Top {limit} Coin (Hacme Göre)**"
            await send_coin_list(message, title, data)
            return

        # 2️⃣ /p n → hacimli ilk n coin
        if len(args) == 1 and args[0].isdigit():
            limit = int(args[0])

            if limit <= 0:
                raise ValueError("❌ n 0'dan büyük olmalı")

            limit = min(limit, SCAN_MAX)
            data = await price_handler.get_top_volume(user_id, limit)

            title = f"🔥 **Binance Top {limit} Coin (Hacme Göre)**"
            await send_coin_list(message, title, data)
            return

        # 3️⃣ /p btc → tek coin bilgisi
        if len(args) == 1 and args[0].isalpha():
            symbol = price_handler.normalize_symbol(args[0])
            data = await price_handler.get_filtered_tickers(user_id, [symbol])

            if not data:
                raise RuntimeError(f"{args[0].upper()} için veri bulunamadı")

            title = f"💰 **{symbol.replace('USDT', '')} Fiyat Bilgisi**"
            await send_coin_list(message, title, data)
            return

        # ❌ Diğer her şey hatadır
        raise ValueError(
            "❌ Kullanım:\n"
            "/p\n"
            "/p 20\n"
            "/p btc"
        )

    except Exception as e:
        logger.error(f"❌ /p error (user {user_id}): {e}", exc_info=True)
        await message.answer(str(e))
"""


@router.message(Command("p"))
async def price_command(message: Message, bot: Bot):
    user_id = message.from_user.id
    args = message.text.split()[1:]

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    try:
        # 1️⃣ /p → default coinler
        if not args:
            limit = SCAN_DEFAULT
            data = await price_handler.get_top_volume(user_id, limit)

            title = f"🔥 **Binance Top {limit} Coin (Hacme Göre)**"
            await send_coin_list(message, title, data)
            return

        # 2️⃣ /p n → hacimli ilk n coin
        if len(args) == 1 and args[0].isdigit():
            limit = int(args[0])

            if limit <= 0:
                raise ValueError("❌ n 0'dan büyük olmalı")

            limit = min(limit, SCAN_MAX)
            data = await price_handler.get_top_volume(user_id, limit)

            title = f"🔥 **Binance Top {limit} Coin (Hacme Göre)**"
            await send_coin_list(message, title, data)
            return

        # 3️⃣ /p btc  OR  /p bnb sol btc → coin bilgileri
        symbols = [price_handler.normalize_symbol(a) for a in args]

        data = await price_handler.get_filtered_tickers(user_id, symbols)

        if not data or len(data) != len(symbols):
            missing = set(symbols) - {c.symbol for c in data}
            raise RuntimeError(
                f"❌ Veri bulunamadı: {', '.join(m.replace('USDT','') for m in missing)}"
            )

        title = f"💰 **Coin Fiyatları ({len(data)} coin)**"
        await send_coin_list(message, title, data)

    except Exception as e:
        logger.error(f"❌ /p error (user {user_id}): {e}", exc_info=True)
        await message.answer(str(e))



@router.message(Command("pg"))
async def gainers_command(message: Message, bot: Bot):
    """Top gainers command"""
    user_id = message.from_user.id
    args = message.text.split()[1:]
    
    limit = SCAN_DEFAULT
    if args and args[0].isdigit():
        limit = min(int(args[0]), SCAN_MAX)
    
    logger.info(f"📈 /pg command from user {user_id}, limit: {limit}")
    
    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    
    try:
        data = await price_handler.get_top_gainers(user_id, limit)
        title = f"📈 **En Çok Yükselen {len(data)} Coin**"
        await send_coin_list(message, title, data)
    except Exception as e:
        logger.error(f"❌ Error in /pg for user {user_id}: {e}")
        await message.answer("❌ Veri alınırken bir hata oluştu.")

@router.message(Command("pl"))
async def losers_command(message: Message, bot: Bot):
    """Top losers command"""
    user_id = message.from_user.id
    args = message.text.split()[1:]
    
    limit = SCAN_DEFAULT
    if args and args[0].isdigit():
        limit = min(int(args[0]), SCAN_MAX)
    
    logger.info(f"📉 /pl command from user {user_id}, limit: {limit}")
    
    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    
    try:
        data = await price_handler.get_top_losers(user_id, limit)
        title = f"📉 **En Çok Düşen {len(data)} Coin**"
        await send_coin_list(message, title, data)
    except Exception as e:
        logger.error(f"❌ Error in /pl for user {user_id}: {e}")
        await message.answer("❌ Veri alınırken bir hata oluştu.")

@router.message(Command("pv"))
async def volume_command(message: Message, bot: Bot):
    """Top volume command"""
    user_id = message.from_user.id
    args = message.text.split()[1:]
    
    limit = SCAN_DEFAULT
    if args and args[0].isdigit():
        limit = min(int(args[0]), SCAN_MAX)
    
    logger.info(f"🔥 /pv command from user {user_id}, limit: {limit}")
    
    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    
    try:
        data = await price_handler.get_top_volume(user_id, limit)
        title = f"🔥 **En Yüksek Hacimli {len(data)} Coin**"
        await send_coin_list(message, title, data)
    except Exception as e:
        logger.error(f"❌ Error in /pv for user {user_id}: {e}")
        await message.answer("❌ Veri alınırken bir hata oluştu.")


@router.message(Command("debug_config"))
async def debug_config(message: Message):
    """Debug config and symbols"""
    user_id = message.from_user.id
    
    debug_info = {
        "SCAN_SYMBOLS": SCAN_SYMBOLS[:10],  # İlk 10'u göster
        "SCAN_DEFAULT": SCAN_DEFAULT,
        "SCAN_MAX": SCAN_MAX,
        "all_symbols_count": len(SCAN_SYMBOLS),
        "handler_scan_default": SCAN_DEFAULT,
        "handler_scan_max": SCAN_MAX,
        "cache_ttl": CACHE_TTL
    }
    
    response = "🔧 **Handler Config Debug Info**\n"
    for key, value in debug_info.items():
        response += f"{key}: {value}\n"
    
    await message.answer(response)

@router.message(Command("debug_endpoints"))
async def debug_endpoints(message: Message):
    """Debug available endpoints"""
    user_id = message.from_user.id
    
    try:
        aggregator = price_handler.binance
        if not aggregator:
            await message.answer("❌ Aggregator not initialized")
            return
        
        # ✅ DÜZELTME: Hard-coded endpoint'leri listele
        endpoints = []
        for endpoint_name, endpoint in aggregator.map_loader.maps["hardcoded"].items():
            endpoints.append(f"{endpoint_name} ({endpoint.base}) - signed: {endpoint.signed}")
        
        response = "📋 **Available Endpoints**\n"
        response += f"Total: {len(endpoints)}\n"
        response += "\n".join(endpoints[:20])  # İlk 20'yi göster
        
        if len(endpoints) > 20:
            response += f"\n... and {len(endpoints) - 20} more"
        
        await message.answer(response, parse_mode=None)
        
    except Exception as e:
        await message.answer(f"❌ Endpoints debug failed: {e}")

@router.message(Command("debug_status"))
async def debug_status(message: Message):
    """Debug handler status"""
    user_id = message.from_user.id
    
    status_info = {
        "handler_initialized": price_handler._initialized,
        "aggregator_status": price_handler._aggregator_status,
        "last_error": price_handler._last_error,
        "binance_instance": "✅ Available" if price_handler.binance else "❌ None"
    }
    
    response = "🔧 **Handler Status**\n"
    for key, value in status_info.items():
        response += f"{key}: {value}\n"
    
    await message.answer(response)

@router.message(Command("plist"))
async def plist(message: Message):
    """List all available commands"""
    
    commands = [
        "/debug_config - p handler için config bilgisi",
        "/debug_endpoints - yaml deki endpoints listesi", 
        "/debug_status - handler durum bilgisi",
        "/p - Price command",
        "/pg - Top gainers", 
        "/pl - Top losers",
        "/pv - Top volume"
    ]
    
    response = "📋 **Available Commands**\n" + "\n".join(commands)
    await message.answer(response, parse_mode=None)

# ============================================================
# STARTUP / SHUTDOWN HOOKS
# ============================================================

@router.startup()
async def on_startup():
    """Initialize on bot startup"""
    await initialize_handler()
    logger.info("✅ Price handler initialized successfully")

@router.shutdown()
async def on_shutdown():
    """Cleanup on bot shutdown"""
    logger.info("🛑 Price handler shutdown")

# Export for handler_loader
__all__ = ['router', 'price_handler']