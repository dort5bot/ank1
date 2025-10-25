"""
handlers/p_handler.py
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

from config import ScanConfig




# Debug Logging Aktif Et main.py veya handler başlangıcına ekle:
import logging
logging.basicConfig(level=logging.DEBUG)
# ============================================================
# CONFIGURATION
# ============================================================

scan_config = ScanConfig()
SCAN_SYMBOLS = scan_config.SCAN_SYMBOLS
SCAN_DEFAULT = scan_config.SCAN_DEFAULT_COUNT
SCAN_MAX = scan_config.SCAN_MAX_COUNT
CACHE_TTL = 60  # 60 seconds cache

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
            
            # ✅ DÜZELTME: Config'i import et
            from config import get_config_sync
            config = get_config_sync()
            
            # ✅ DÜZELTME: Base path'i doğru ver
            base_path = os.path.abspath(os.path.join(
                os.path.dirname(__file__), 
                "..", 
                "utils", 
                "binance_api"
            ))
            
            # ✅ CRITICAL DÜZELTME: await ekle
            self.binance = await BinanceAggregator.get_instance(base_path=base_path)
            
            # ✅ Test connection
            if hasattr(self.binance, 'test_connection'):
                test_result = await self.binance.test_connection()
                logger.info(f"🔧 Aggregator connection test: {test_result}")
            
            # ✅ Global credentials kontrolü
            if hasattr(self.binance, 'validate_global_credentials'):
                global_valid = await self.binance.validate_global_credentials()
                logger.info(f"🔑 Global credentials valid: {global_valid}")
            
            self._initialized = True
            self._aggregator_status = "initialized"
            logger.info("✅ PriceHandler initialized successfully")
            return True
            
        except Exception as e:
            self._last_error = str(e)
            self._aggregator_status = f"error: {e}"
            logger.error(f"❌ PriceHandler initialization failed: {e}")
            return False
            
        
    
    def normalize_symbol(self, symbol: str) -> str:
        """Fast symbol normalization"""
        cleaned = self._symbol_regex.sub('', symbol.upper().strip())
        return cleaned if cleaned.endswith('USDT') else cleaned + 'USDT'
        
    # p_handler.py - veri sorgusu

    async def get_all_tickers(self, user_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get limited tickers - CORRECTED VERSION"""
        
        cached_data = await self._cache.get(user_id, f"all_tickers_{limit}")
        if cached_data:
            return cached_data
        
        try:
            # ✅ DÜZELTİLDİ: Doğru parametre sırası
            data = await self.binance.get_data("ticker_24hr_all", user_id=user_id)
            
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
                return await self._http_fallback(user_id)
                
        except Exception as e:
            logger.error(f"❌ Aggregator error in get_all_tickers: {e}")
            return await self._http_fallback(user_id)
            
            
     

    #----
    async def _http_fallback(self, user_id: int) -> List[Dict[str, Any]]:
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
                        # Cache fallback result
                        await self._cache.set(user_id, "all_tickers", usdt_pairs)
                        logger.info(f"✅ HTTP Fallback: {len(usdt_pairs)} USDT pairs")
                        return usdt_pairs
                    else:
                        logger.error(f"❌ HTTP Fallback Error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"❌ HTTP Fallback failed: {e}")
            return []
    
    async def get_filtered_tickers(self, user_id: int, symbols: List[str]) -> List[CoinData]:
        """Get specific symbols data"""
        all_tickers = await self.get_all_tickers(user_id)
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

@router.message(Command("p"))
async def price_command(message: Message, bot: Bot):
    """Price command for specific coins"""
    user_id = message.from_user.id
    args = message.text.split()[1:]
    
    # Determine symbols to show
    if args:
        symbols = [price_handler.normalize_symbol(arg) for arg in args]
    else:
        symbols = SCAN_SYMBOLS
    
    logger.info(f"📊 /p command from user {user_id} for symbols: {symbols}")
    
    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)
    
    try:
        data = await price_handler.get_filtered_tickers(user_id, symbols)
        title = f"💰 **Coin Fiyatları** ({len(data)} coin)"
        await send_coin_list(message, title, data)
    except Exception as e:
        logger.error(f"❌ Error in /p for user {user_id}: {e}")
        await message.answer("❌ Veri alınırken bir hata oluştu.")

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




    
# p_handler.py - debug_command'a ekle
@router.message(Command("debug_config"))
async def debug_config(message: Message):
    """Debug config and symbols"""
    user_id = message.from_user.id
    
    debug_info = {
        "SCAN_SYMBOLS": SCAN_SYMBOLS,
        "SCAN_DEFAULT": SCAN_DEFAULT,
        "SCAN_MAX": SCAN_MAX,
        "config_source": "from scan_config",
        "all_symbols_count": len(SCAN_SYMBOLS),
        "symbols_sample": SCAN_SYMBOLS[:5] if SCAN_SYMBOLS else "EMPTY"
    }
    
    # Config instance'ını da kontrol et
    try:
        from config import get_config_sync
        config = get_config_sync()
        debug_info["config_scan_symbols"] = config.SCAN.SCAN_SYMBOLS[:5] if config.SCAN.SCAN_SYMBOLS else "EMPTY"
        debug_info["config_scan_count"] = len(config.SCAN.SCAN_SYMBOLS)
    except Exception as e:
        debug_info["config_error"] = str(e)
    
    response = "🔧 **Config Debug Info**\n"
    for key, value in debug_info.items():
        response += f"{key}: {value}\n"
    
    await message.answer(response)
    

@router.message(Command("test_aggregator"))
async def test_aggregator(message: Message):
    """Test BinanceAggregator directly"""
    user_id = message.from_user.id
    
    try:
        # Aggregator instance'ını doğrudan test et
        aggregator = price_handler.binance
        
        if not aggregator:
            await message.answer("❌ Aggregator not initialized")
            return
        
        # Metodları test et
        test_results = []
        
        # Test 1: Basit ping
        if hasattr(aggregator, 'ping'):
            ping_result = await aggregator.ping()
            test_results.append(f"✅ Ping: {ping_result}")
        else:
            test_results.append("❌ Ping method not available")
        
        # Test 2: get_data metodunu test et
        if hasattr(aggregator, 'get_data'):
            try:
                test_data = await aggregator.get_data(user_id, "test")
                test_results.append(f"✅ get_data: {type(test_data)}")
            except Exception as e:
                test_results.append(f"❌ get_data error: {e}")
        else:
            test_results.append("❌ get_data not available")
        
        # Test 3: Available endpoints
        if hasattr(aggregator, 'get_available_endpoints'):
            endpoints = await aggregator.get_available_endpoints()
            test_results.append(f"✅ Endpoints: {endpoints}")
        
        response = "🔧 **Aggregator Test Results**\n" + "\n".join(test_results)
        await message.answer(response)
        
    except Exception as e:
        await message.answer(f"❌ Test error: {e}")
        


# p_handler.py'ye ekle
@router.message(Command("test_ping"))
async def test_ping(message: Message):
    """Test basic connectivity"""
    user_id = message.from_user.id
    
    try:
        # SpotClient ile basit ping testi
        from utils.binance_api.b_spotclient import SpotClient
        from utils.binance_api.binance_request import BinanceHTTPClient
        from config import get_config_sync
        
        config = get_config_sync()
        http_client = BinanceHTTPClient(
            api_key=config.BINANCE_API_KEY,
            secret_key=config.BINANCE_API_SECRET, 
            user_id=user_id
        )
        
        spot_client = SpotClient.get_instance(http_client)
        result = await spot_client.ping()
        
        await message.answer(f"✅ Ping successful: {result}")
        
    except Exception as e:
        await message.answer(f"❌ Ping failed: {e}")


# p_handler.py - test_ticker komutunu debug version yapalım
@router.message(Command("test_ticker"))
async def test_ticker_endpoint(message: Message):
    """Test ticker_24hr_all endpoint - DEBUG VERSION"""
    user_id = message.from_user.id
    
    try:
        logger.info(f"🔄 test_ticker DEBUG called for user {user_id}")
        
        # DEBUG: Aggregator durumunu kontrol et
        if not price_handler.binance:
            await message.answer("❌ Aggregator not initialized", parse_mode=None)
            return
            
        # DEBUG: Endpoint'i doğrudan test et
        logger.info("🔄 Calling ticker_24hr_all directly...")
        data = await price_handler.binance.get_data(user_id, "ticker_24hr_all")
        logger.info(f"📊 Direct call completed - Type: {type(data)}, Length: {len(data) if data else 0}")
        
        if data is None:
            await message.answer("❌ Aggregator returned None", parse_mode=None)
            return
            
        response = f"✅ **Ticker DEBUG Results**\n"
        response += f"Data type: {type(data)}\n"
        response += f"Data length: {len(data) if data else 0}\n"
        
        if data and len(data) > 0:
            # Sadece ilk 3 item'i göster
            sample = data[:3]
            response += f"First 3 items:\n"
            for i, item in enumerate(sample):
                if isinstance(item, dict):
                    symbol = item.get('symbol', 'N/A')
                    response += f"{i+1}. {symbol}\n"
                else:
                    response += f"{i+1}. {type(item)}: {str(item)[:50]}\n"
        else:
            response += "❌ Empty data received\n"
        
        await message.answer(response, parse_mode=None)
        
    except Exception as e:
        error_msg = f"❌ Ticker DEBUG failed: {str(e)[:400]}"
        logger.error(f"test_ticker DEBUG error: {e}", exc_info=True)
        await message.answer(error_msg, parse_mode=None)
        
    
# p_handler.py'ye ekle
@router.message(Command("debug_router"))
async def debug_router(message: Message):
    """Debug router and handler registration"""
    
    # Mevcut handler'ları listele
    handler_info = []
    
    # Router'daki handler'ları al
    for observer in router.message._observers:
        for handler in observer.handlers:
            handler_info.append(f"Filter: {handler.filters} -> Handler: {handler.callback.__name__}")
    
    response = "🔧 **Router Debug**\n"
    response += f"Total handlers: {len(handler_info)}\n"
    response += "Registered handlers:\n"
    
    for info in handler_info[:10]:  # İlk 10'u göster
        response += f"• {info}\n"
    
    if len(handler_info) > 10:
        response += f"... and {len(handler_info) - 10} more\n"
    
    await message.answer(response, parse_mode=None)


@router.message(Command("debug_endpoints"))
async def debug_endpoints(message: Message):
    """Debug available endpoints"""
    user_id = message.from_user.id
    
    try:
        aggregator = price_handler.binance
        if not aggregator:
            await message.answer("❌ Aggregator not initialized")
            return
        
        # Map loader'dan endpoint'leri listele
        endpoints = []
        for map_name, map_data in aggregator.map_loader.maps.items():
            for section, section_data in map_data.items():
                if section == "meta":
                    continue
                for endpoint_name, endpoint_config in section_data.items():
                    endpoints.append(f"{endpoint_name} ({section})")
        
        response = "📋 **Available Endpoints**\n"
        response += f"Total: {len(endpoints)}\n"
        response += "\n".join(endpoints[:20])  # İlk 20'yi göster
        
        if len(endpoints) > 20:
            response += f"\n... and {len(endpoints) - 20} more"
        
        await message.answer(response, parse_mode=None)
        
    except Exception as e:
        await message.answer(f"❌ Endpoints debug failed: {e}")        
 

# p_handler.py'ye ekle - TÜM KOMUTLARI LİSTELE
@router.message(Command("list_commands"))
async def list_commands(message: Message):
    """List all available commands"""
    
    commands = [
        "/echo - Simple echo test",
        "/debug_router - Debug router registration", 
        "/test_simple - Test simple ticker",
        "/test_http_client - Test HTTP client",
        "/test_ticker - Test ticker endpoint",
        "/test_alternative - Test alternative endpoints",
        "/debug_price - Price handler debug",
        "/debug_aggregator - Aggregator debug",
        "/debug_credentials - Credentials debug",
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