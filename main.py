# main.py 
# zorunlu olarak seçmeli polling + webhook
"""
# Local development (polling)
USE_WEBHOOK=false python main.py

# Production (webhook)  
USE_WEBHOOK=true python main.py
"""

# main.py - İYİLEŞTİRİLMİŞ IMPORT
import os
import asyncio
import logging
import signal
import time
import aiosqlite
from typing import Optional, Dict, Any, Union
from contextlib import asynccontextmanager

# Üçüncü parti kütüphaneler
import aiohttp
from datetime import datetime
from aiohttp import web
from dotenv import load_dotenv

# Aiogram
from aiogram import Bot, Dispatcher, Router
from aiogram.types import Update, Message, ErrorEvent
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

# Proje modülleri
from config import BotConfig, get_telegram_token, get_admins, get_config
from utils.handler_loader import HandlerLoader
from utils.apikey_manager import APIKeyManager, AlarmManager, BaseManager, TradeSettingsManager
from utils.context_logger import setup_context_logging, get_context_logger, ContextAwareLogger
from utils.performance_monitor import PerformanceMonitor
from utils.security_auditor import security_auditor
from utils.binance_api.binance_a import BinanceAggregator
from utils.binance_api.binance_exceptions import BinanceAPIError, BinanceAuthenticationError


# ---------------------------------------------------------------------
# Global instances - LOGGER
# ---------------------------------------------------------------------
bot: Optional[Bot] = None
dispatcher: Optional[Dispatcher] = None
binance_api: Optional[Union[BinanceAggregator]] = None
app_config: Optional[BotConfig] = None
runner: Optional[web.AppRunner] = None
shutdown_event = asyncio.Event()

# ✅ LOGGER
logger: Optional[logging.Logger] = None

# Configure logging
load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

def setup_logger():
    """Logger'ı global olarak setup et"""
    global logger
    logger = logging.getLogger(__name__)
    setup_context_logging()
    logger = get_context_logger(__name__)
    return logger

# ✅ Logger initialize
logger = setup_logger()

# ---------------------------------------------------------------------
# Bot Factory & Data Structure (main-eski'den)
# ---------------------------------------------------------------------
async def create_bot_instance(config: Optional[BotConfig] = None) -> Bot:
    """Merkezi bot instance oluşturucu"""
    bot_instance = Bot(
        token=get_telegram_token(),
        default=DefaultBotProperties(
            parse_mode=ParseMode.HTML,
        )
    )
    init_bot_data(bot_instance)
    
    if config:
        bot_instance.data['config'] = config

    logger.info("✅ Bot instance created with consistent data dict")
    return bot_instance

def init_bot_data(bot_instance: Bot) -> None:
    """Bot data structure'ını standardize et"""
    if not hasattr(bot_instance, 'data') or bot_instance.data is None:
        bot_instance.data = {}
    
    standard_data = {
        'binance_api': None,
        'start_time': datetime.now(),
        'user_sessions': {},
        'circuit_breakers': {},
        'metrics': {
            'messages_processed': 0,
            'errors_count': 0,
            'last_health_check': None,
            'active_users': 0
        },
        'config': None,
        'aggregator': None,
        'health_status': 'initializing'
    }
    
    # Deep merge yap (sadece eksik key'leri ekle)
    for key, default_value in standard_data.items():
        if key not in bot_instance.data:
            bot_instance.data[key] = default_value
        elif isinstance(default_value, dict) and isinstance(bot_instance.data[key], dict):
            for sub_key, sub_value in default_value.items():
                if sub_key not in bot_instance.data[key]:
                    bot_instance.data[key][sub_key] = sub_value

# ---------------------------------------------------------------------
# ENHANCED STARTUP SEQUENCE 
# ---------------------------------------------------------------------

async def startup_sequence(dispatcher_instance: Dispatcher) -> bool:
    """Basitleştirilmiş startup sequence - SADECE KONTROL"""
    try:
        logger.info("🔍 Starting simplified startup check...")
        

        if dispatcher_instance and hasattr(dispatcher_instance, 'sub_routers'):
            router_count = len(dispatcher_instance.sub_routers)
            logger.info(f"📋 Found {router_count} routers")
            
            if router_count > 0:
                logger.info("✅ Startup check: SUCCESS")
                return True
        
        # Eğer router yoksa emergency handler ekle
        logger.error("❌ No routers found - adding emergency handlers")
        await add_emergency_handlers(dispatcher_instance)
        return True
        
    except Exception as e:
        logger.error(f"❌ Startup check failed: {e}")
        await add_emergency_handlers(dispatcher_instance)
        return True


   
# ---------------------------------------------------------------------
# Error Handler
# ---------------------------------------------------------------------
async def error_handler(event: ErrorEvent) -> None:
    """Global error handler for aiogram."""
    exception = event.exception
    
    # Security audit log
    try:
        user_id = getattr(event.update, 'from_user', None)
        if user_id:
            user_id = user_id.id
            await security_auditor.audit_request(
                user_id, 
                "error", 
                {"error_type": type(exception).__name__, "message": str(exception)}
            )
    except Exception as audit_error:
        logger.error(f"Security audit failed: {audit_error}")
    
    # ✅ METRİK GÜNCELLEME
    if bot and hasattr(bot, 'data') and 'metrics' in bot.data:
        bot.data['metrics']['errors_count'] = bot.data['metrics'].get('errors_count', 0) + 1
        
    # Kritik hatalarda admin'e bildir
    if isinstance(exception, (BinanceAuthenticationError, ConnectionError)):
        await notify_admins_about_critical_error(exception)
    
    # Hata türlerine göre loglama
    if isinstance(exception, (ConnectionError, asyncio.TimeoutError)):
        logger.warning(f"🌐 Network error in update {event.update.update_id}: {exception}")
        
    elif isinstance(exception, BinanceAuthenticationError):
        logger.error(f"🔐 Authentication error: {exception}")
        
    elif isinstance(exception, BinanceAPIError):
        error_code = getattr(exception, 'code', 'N/A')
        logger.error(f"📊 Binance API error (code: {error_code}): {exception}")
        
    elif isinstance(exception, ValueError):
        logger.warning(f"⚠️ Validation error: {exception}")
        
    elif hasattr(exception, 'code'):
        logger.error(f"🔧 API error (code: {exception.code}): {exception}")
        
    elif "auth" in str(exception).lower() or "token" in str(exception).lower():
        logger.error(f"🔐 Authentication error (detected): {exception}")
        
    else:
        logger.error(f"❌ Unexpected error in update {event.update.update_id}: {exception}", 
                    exc_info=True)

    # Kullanıcıya hata mesajı gönder (güvenli şekilde)
    try:
        if getattr(event.update, "message", None):
            await event.update.message.answer("❌ Bir hata oluştu, lütfen daha sonra tekrar deneyin.")
    except Exception as e:
        logger.error(f"❌ Failed to send error message: {e}")

# ---------------------------------------------------------------------
# Middleware Implementation
# ---------------------------------------------------------------------
class LoggingMiddleware:
    """Middleware for request logging and monitoring."""
    
    async def __call__(self, handler, event, data):
        logger.info(f"📨 Update received: {getattr(event, 'update_id', 'unknown')}")
        start_time = asyncio.get_event_loop().time()
        
        try:
            result = await handler(event, data)
            processing_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"✅ Update processed: {getattr(event, 'update_id', 'unknown')} in {processing_time:.2f}s")
            return result
        except Exception as e:
            logger.error(f"❌ Error processing update {getattr(event, 'update_id', 'unknown')}: {e}")
            raise

class AuthenticationMiddleware:
    """Middleware for user authentication and authorization."""
    
    async def __call__(self, handler, event, data):
        global app_config
        
        user = getattr(event, "from_user", None)
        if user:
            user_id = user.id
            data['user_id'] = user_id
            data['is_admin'] = app_config.is_admin(user_id) if app_config else False
            logger.debug(f"👤 User {user_id} - Admin: {data['is_admin']}")
        
        return await handler(event, data)

# ---------------------------------------------------------------------
# Dependency Injection Container
# ---------------------------------------------------------------------
class DIContainer:
    """Simple dependency injection container for global instances."""
    
    _instances: Dict[str, Any] = {}
    
    @classmethod
    def register(cls, key: str, instance: Any) -> None:
        """Register an instance with a key."""
        cls._instances[key] = instance
        logger.debug(f"📦 DI Container: Registered {key}")
    
    @classmethod
    def resolve(cls, key: str) -> Optional[Any]:
        """Resolve an instance by key."""
        return cls._instances.get(key)
    
    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get all registered instances."""
        return cls._instances.copy()

# ---------------------------------------------------------------------
# .db initialize_binance_api
# ---------------------------------------------------------------------

async def initialize_binance_api() -> Optional[Any]:
    """Initialize Binance API with proper factory pattern."""
    global app_config
    
    if not app_config.ENABLE_TRADING:
        logger.info("ℹ️ Binance API not initialized (trading disabled)")
        return None
    
    try:
        logger.info("🔄 Initializing Binance API...")
        aggregator = BinanceAggregator.get_instance()
        logger.info("✅ Binance API initialized successfully")
        return aggregator
    except Exception as e:
        logger.error(f"❌ Binance API initialization failed: {e}")
        return None

# ---------------------------------------------------------------------
# Handler Loading - ENHANCED HANDLER LOADING SYSTEM
# ---------------------------------------------------------------------

async def initialize_handlers(dispatcher_instance: Dispatcher) -> Dict[str, int]:
    """Initialize handlers with comprehensive logging."""
    try:
        # 2- Handler'ları yükle 
        loader = HandlerLoader(dispatcher=dispatcher_instance)
        load_results = await loader.load_handlers(dispatcher_instance)
        
        # 3- Router bilgilerini BASİT şekilde logla
        if dispatcher_instance and hasattr(dispatcher_instance, 'sub_routers'):
            router_count = len(dispatcher_instance.sub_routers)
            logger.info(f"📋 Loaded {router_count} routers")
            
            # Router isimlerini logla
            for i, router in enumerate(dispatcher_instance.sub_routers):
                router_name = getattr(router, 'name', f'router_{i}')
                logger.info(f"   🎯 Router {i+1}: {router_name}")
        
        logger.info(f"📊 Handler loading results: {load_results}")
        
        # 4- Eğer hiç handler yüklenmediyse, emergency handler ekle
        if load_results.get('loaded', 0) == 0:
            logger.warning("⚠️ No handlers loaded - adding emergency handler")
            await add_emergency_handlers(dispatcher_instance)
            load_results['emergency'] = 1
        
        return load_results
        
    except Exception as e:
        logger.error(f"❌ Handler loading error: {e}")
        return {"loaded": 0, "failed": 1, "error_type": str(e)}

   

async def add_emergency_handlers(dispatcher_instance: Dispatcher):
    """Acil durum handler'ları ekle"""
    
    if dispatcher_instance is None:
        logger.critical("❌ Dispatcher instance is None - cannot add emergency handlers!")
        return

    from aiogram import Router
    from aiogram.filters import Command
    from aiogram.types import Message

    emergency_router = Router()

    @emergency_router.message(Command("start"))
    async def emergency_start(message: Message):
        await message.answer("🆘 Bot acil durum modunda çalışıyor. Handler'lar yüklenemedi.")

    @emergency_router.message(Command("help"))
    async def emergency_help(message: Message):
        await message.answer("ℹ️ Bot şu anda acil durum modunda. Lütfen daha sonra tekrar deneyin.")

    dispatcher_instance.include_router(emergency_router)
    logger.info("✅ Emergency handlers added")



# ---------------------------------------------------------------------
# Webhook Setup Functions
# ---------------------------------------------------------------------
# webhook'u sıfırla
async def reset_webhook(bot_instance: Bot):
    """Webhook'u sıfırla ve kontrol et"""
    try:
        await bot_instance.delete_webhook(drop_pending_updates=True)
        logger.info("✅ Webhook resetlendi")
        
        # Webhook bilgilerini kontrol et
        webhook_info = await bot_instance.get_webhook_info()
        logger.info(f"📊 Webhook bilgileri: {webhook_info}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Webhook resetleme hatası: {e}")
        return False



async def on_startup(bot: Bot) -> None:
    """Minimal startup - webhook bash script tarafından yönetiliyor"""
    global app_config, logger
    
    try:
        # ✅ Logger kontrolü
        if logger is None:
            setup_logger()
            
        # ✅ BOT ADI ile başlatma mesajı
        bot_info = await bot.get_me()
        bot_name = bot_info.username
        bot_first_name = bot_info.first_name
        
        # .env'den TELEGRAM_NAME al veya bot info'dan kullan
        env_bot_name = os.environ.get('TELEGRAM_NAME', '')
        display_name = env_bot_name if env_bot_name else f"{bot_first_name} (@{bot_name})"
        
        logger.info(f"🤖 BOT BAŞLATILDI: {display_name}")
        logger.info(f"   ├─ Username: @{bot_name}")
        logger.info(f"   ├─ First Name: {bot_first_name}")
        logger.info(f"   └─ ID: {bot_info.id}")
        
        # ✅ SADECE handler yükleme ve basit kontrol
        logger.info("🔄 Starting bot with external webhook management...")
        
        # Handler durumunu kontrol et - BASİT VERSİYON
        if dispatcher and hasattr(dispatcher, 'sub_routers'):
            router_count = len(dispatcher.sub_routers)
            logger.info(f"📋 Loaded routers: {router_count}")
            
            # ✅ Router isimlerini logla (handler saymaya GEREK YOK)
            for i, router in enumerate(dispatcher.sub_routers):
                router_name = getattr(router, 'name', f'router_{i}')
                logger.info(f"   🎯 Router {i+1}: {router_name}")
        
        # Webhook bilgilerini sadece oku (ayarlama değil)
        webhook_info = await bot.get_webhook_info()
        logger.info(f"🌐 Webhook URL: {webhook_info.url}")
        logger.info(f"📊 Pending updates: {webhook_info.pending_update_count}")
        
        if webhook_info.pending_update_count > 0:
            logger.info("🔄 Processing pending updates...")
        
        logger.info("✅ Bot started successfully (external webhook management)")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
        

async def on_shutdown(bot: Bot) -> None:
    """Execute on application shutdown."""
    logger.info("🛑 Shutting down application...")
    
    try:
        # Delete webhook
        if app_config and app_config.WEBHOOK_HOST:
            try:
                await bot.delete_webhook()
                logger.info("✅ Webhook deleted")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete webhook: {e}")
    except Exception as e:
        logger.warning(f"⚠️ on_shutdown encountered an error: {e}")

# ---------------------------------------------------------------------
# Health Check Endpoints
# ---------------------------------------------------------------------

async def health_check(request: web.Request) -> web.Response:
    """Enhanced health check with comprehensive metrics."""
    try:
        # Handler bilgisi daha güvenli şekilde
        handler_info = {
            "total_routers": len(dispatcher.sub_routers) if dispatcher else 0,
            "router_names": [getattr(r, 'name', 'unnamed') for r in dispatcher.sub_routers] if dispatcher else [],
            "loaded_handlers": getattr(dispatcher, '_handlers_count', 0) if dispatcher else 0
        }
        
        async with asyncio.timeout(10):
            return await _perform_health_check(handler_info)  # handler_info parametre olarak
    except TimeoutError:
        logger.warning("⏰ Health check timeout - services responding slowly")
        return web.json_response({
            "status": "timeout", 
            "message": "Health check took too long",
            "timestamp": datetime.now().isoformat()
        }, status=503)
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return web.json_response({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "critical": True
        }, status=500)



async def initialize_managers():
    """TEK ve AÇIK initialization, gerekirse utils/__init__.py içine taşınıp merkezi olabilir"""
    try:
        logger.info("🔄 Initializing managers...")
        
        # Database initialization
        success = await BaseManager.initialize_database()
        if success:
            # Manager instance'larını oluştur
            APIKeyManager.get_instance()
            AlarmManager.get_instance() 
            TradeSettingsManager.get_instance()
            
            logger.info("✅ All managers initialized successfully")
        else:
            logger.error("❌ Database initialization failed")
            
        return success
        
    except Exception as e:
        logger.error(f"❌ Manager initialization failed: {e}")
        return False

async def _perform_health_check(handler_info: dict = None) -> web.Response:
    """Internal health check implementation without timeout."""
    services_status = await check_services()
    
    # Performance metrics
    performance_metrics = {}
    try:
        monitor = PerformanceMonitor.get_instance()
        performance_summary = monitor.get_summary()
        performance_metrics = {
            'monitored_functions': performance_summary['total_functions_monitored'],
            'total_calls': performance_summary['total_calls'],
            'avg_call_time': round(performance_summary['average_call_time'], 3),
            'top_slow_functions': performance_summary['top_slow_functions']
        }
    except Exception as e:
        performance_metrics = {'error': str(e)}
    
    # Bot metrics
    bot_metrics = {"basic": {}, "performance": {}, "business": {}, "handlers": {}}
    
    # Handler bilgilerini ekle
    if handler_info:
        bot_metrics["handlers"] = handler_info
    
    if bot and hasattr(bot, 'data') and bot.data:
        basic_metrics = bot.data.get('metrics', {})
        bot_metrics["basic"] = basic_metrics
        
        if 'start_time' in bot.data:
            uptime = datetime.now() - bot.data['start_time']
            bot_metrics["performance"]["uptime_seconds"] = uptime.total_seconds()
            bot_metrics["performance"]["uptime_human"] = str(uptime).split('.')[0]
            bot_metrics["performance"]["start_time"] = bot.data['start_time'].isoformat()
        
        if 'user_sessions' in bot.data:
            bot_metrics["business"]["active_users"] = len(bot.data['user_sessions'])
            bot_metrics["business"]["user_ids"] = list(bot.data['user_sessions'].keys())[:10]
        
        if 'circuit_breakers' in bot.data:
            bot_metrics["business"]["active_circuit_breakers"] = len(bot.data['circuit_breakers'])
            
        if 'binance_api' in bot.data and bot.data['binance_api']:
            bot_metrics["business"]["binance_api_connected"] = True
        else:
            bot_metrics["business"]["binance_api_connected"] = False
    
    # Platform detection improvement
    platform = "local"
    if "RENDER" in os.environ:
        platform = "render"
    elif "HEROKU" in os.environ:
        platform = "heroku"
    elif "RAILWAY" in os.environ:
        platform = "railway"
    
    return web.json_response({
        "status": "healthy",
        "service": "telegram-bot",
        "platform": platform,
        "timestamp": datetime.now().isoformat(),
        "services": services_status,
        "bot_metrics": bot_metrics,
        "performance_metrics": performance_metrics,
        "multi_user_enabled": True,
        "version": "1.0.0"  # Sabit versiyon bilgisi
    })


async def detailed_health_check(request: web.Request) -> web.Response:
    """Detaylı sistem sağlık kontrolü"""
    health_info = {
        "database": {
            "initialized": BaseManager._db_initialized,
            "connections": len(BaseManager._db_connections),
            "cache_size": len(APIKeyManager._cache)
        },
        "encryption": {
            "initialized": BaseManager._fernet is not None
        },
        "managers": {
            "api_key_manager": APIKeyManager._instance is not None,
            "alarm_manager": AlarmManager._instance is not None,
            "trade_manager": TradeSettingsManager._instance is not None
        }
    }
    
    return web.json_response(health_info)

async def readiness_check(request: web.Request) -> web.Response:
    """Readiness check for Kubernetes and load balancers."""
    global bot, binance_api, app_config
    
    if bot and app_config:
        if app_config.ENABLE_TRADING and not binance_api:
            return web.json_response({"status": "not_ready"}, status=503)
        
        essential_services = ['bot', 'dispatcher', 'config']
        missing_services = [svc for svc in essential_services if not DIContainer.resolve(svc)]
        
        if missing_services:
            return web.json_response({
                "status": "not_ready",
                "missing_services": missing_services
            }, status=503)
            
        return web.json_response({"status": "ready"})
    else:
        return web.json_response({"status": "not_ready"}, status=503)

# ---------------------------------------------------------------------
# Service Check
# ---------------------------------------------------------------------
async def check_services() -> Dict[str, Any]:
    """Check connectivity to all external services."""
    global bot, binance_api, app_config
    
    services_status = {}
    
    # Check Telegram API
    try:
        if bot:
            me = await bot.get_me()
            services_status["telegram"] = {
                "status": "connected",
                "bot_username": me.username,
                "bot_id": me.id,
                "first_name": me.first_name
            }
        else:
            services_status["telegram"] = {"status": "disconnected", "error": "Bot not initialized"}
    except Exception as e:
        services_status["telegram"] = {
            "status": "disconnected",
            "error": str(e)
        }
    
    # Check Binance API
    if app_config.ENABLE_TRADING:
        try:
            if binance_api:
                ping_result = await binance_api.ping()
                services_status["binance"] = {
                    "status": "connected" if ping_result else "disconnected",
                    "ping": ping_result,
                    "trading_enabled": True
                }
            else:
                services_status["binance"] = {"status": "disconnected", "error": "Binance API not initialized", "trading_enabled": True}
        except Exception as e:
            services_status["binance"] = {
                "status": "disconnected",
                "error": str(e),
                "trading_enabled": True
            }
    else:
        services_status["binance"] = {
            "status": "disabled",
            "trading_enabled": False
        }
    
    return services_status

# ---------------------------------------------------------------------
# LIFESPAN MANAGEMENT - sadeleştir - tekrar olmasın

@asynccontextmanager
async def lifespan(config: BotConfig):
    """Basitleştirilmiş lifespan - TÜM initialization burada"""
    global bot, dispatcher, binance_api, app_config
    
    try:
        app_config = config
        
        # ✅ # ✅ ✅ ✅ CRITICAL: MANAGER'LARI EN BAŞTA BAŞLAT
        if not await initialize_managers():
            raise RuntimeError("Manager initialization failed")
        
        
        # ✅ 1-PERFORMANCE MONITORING
        ContextAwareLogger.add_context('lifecycle_phase', 'bot_initialization')
        
        # ✅ 2-CRITICAL: Tüm bileşenleri sırayla başlat
        bot = await create_bot_instance(config=app_config)
        dispatcher = Dispatcher()
        
        # ✅ 3-Error handler & middleware
        dispatcher.errors.register(error_handler)
        dispatcher.update.outer_middleware(LoggingMiddleware())
        dispatcher.update.outer_middleware(AuthenticationMiddleware())
        
        # ✅ 4-DI Container
        DIContainer.register('bot', bot)
        DIContainer.register('dispatcher', dispatcher)
        DIContainer.register('config', app_config)
        
        # ✅ 5-Binance API (sadece trading enabled ise)
        binance_api = await initialize_binance_api()
        if binance_api:
            bot.data["binance_api"] = binance_api
            DIContainer.register('binance_api', binance_api)
        
        # ✅ 5-HANDLER'ları YÜKLE
        logger.info("🔄 Loading handlers...")
        load_results = await initialize_handlers(dispatcher)
        logger.info(f"📊 Handler loading results: {load_results}")
        
        # ✅ 6- STARTUP KONTROLÜ (YENİ & DOĞRU fonksiyonla)
        startup_ok = await startup_sequence(dispatcher)
        if not startup_ok:
            raise RuntimeError("Startup sequence failed")
        
        logger.info("✅ All components initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"❌ Bot initialization error: {e}")
        raise
    finally:
        ContextAwareLogger.remove_context('lifecycle_phase')

# ---------------------------------------------------------------------
# CLEANUP FUNCTION -  PERIODIC CLEANUP TASKS
# ---------------------------------------------------------------------
async def cleanup_resources():
    """Tüm kaynakları temizle"""
    global runner, bot
    
    logger.info("🧹 Cleaning up resources...")
    
    cleanup_tasks = []
    
    # ✅ API Manager kaynaklarını temizle
    try:
        api_manager = APIKeyManager.get_instance()
        if hasattr(api_manager, 'cleanup'):
            cleanup_tasks.append(api_manager.cleanup()) 
        cleanup_tasks.append(BaseManager.cleanup_all())
        logger.info("✅ API managers cleanup scheduled")
    except Exception as e:
        logger.warning(f"⚠️ API manager cleanup failed: {e}")
    
    
    
    if runner:
        cleanup_tasks.append(runner.cleanup())
        logger.info("✅ App runner cleanup scheduled")
    
    if bot and hasattr(bot, 'session'):
        cleanup_tasks.append(bot.session.close())
        logger.info("✅ Bot session cleanup scheduled")
    
    # Binance API cleanup
    global binance_api
    if binance_api:
        if hasattr(binance_api, 'close'):
            cleanup_tasks.append(binance_api.close())
            logger.info("✅ Binance API cleanup scheduled")
    
    if cleanup_tasks:
        try:
            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"⚠️ Cleanup task failed: {result}")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup gathering failed: {e}")
    
    logger.info("✅ All resources cleaned up")


#sil/ kullan
async def start_periodic_cleanup():
    """Periodic cleanup tasks for performance optimization"""
    try:
        while not shutdown_event.is_set():
            await asyncio.sleep(3600)  # Her 1 saatte bir
            
            # Clean old user sessions
            if bot and hasattr(bot, 'data') and 'user_sessions' in bot.data:
                current_time = datetime.now()
                expired_sessions = []
                
                for user_id, session_data in bot.data['user_sessions'].items():
                    if 'last_activity' in session_data:
                        time_diff = current_time - session_data['last_activity']
                        if time_diff.total_seconds() > 7200:  # 2 saat
                            expired_sessions.append(user_id)
                
                for user_id in expired_sessions:
                    del bot.data['user_sessions'][user_id]
                    logger.info(f"🧹 Cleaned expired session for user {user_id}")
            
            # Clear performance monitor old data
            try:
                monitor = PerformanceMonitor.get_instance()
                monitor.cleanup_old_data(hours=24)
            except Exception as e:
                logger.warning(f"⚠️ Performance monitor cleanup failed: {e}")
                
    except asyncio.CancelledError:
        logger.info("🛑 Periodic cleanup task cancelled")
    except Exception as e:
        logger.error(f"❌ Periodic cleanup error: {e}")




# ---------------------------------------------------------------------
# ÇALIŞMA MODU KONFİGÜRASYONU
# ---------------------------------------------------------------------
def get_bot_mode() -> str:
    """Bot çalışma modunu belirle"""
    # Oracle ortamında webhook, local'de polling
    if any(env_var in os.environ for env_var in ['ORACLE', 'OCI_', 'OPC_']):
        return "webhook"
    elif os.environ.get('USE_WEBHOOK', '').lower() in ['true', '1', 'yes']:
        return "webhook"
    else:
        return "polling"






# ---------------------------------------------------------------------
# OPTIMIZED MAIN ENTRY POINT - CONFIG TABANLI
# ---------------------------------------------------------------------

async def app_entry():
    """Config tabanlı çift modlu main entry"""
    global app_config, runner, bot, dispatcher
    
    try:
        # ✅ Config yükle
        logger.info("📋 Loading configuration...")
        app_config = await get_config()
        
        # ✅ Config'ten modu oku
        bot_mode = "webhook" if app_config.USE_WEBHOOK else "polling"
        logger.info(f"🚀 Starting bot in {bot_mode.upper()} mode (from config)...")
        logger.info("🤖 Bot polling modunda başlatılıyor-elma...")
        #await dispatcher.start_polling(bot)
        
        # ✅ Lifespan ile bileşenleri başlat
        async with lifespan(app_config):
            
            if app_config.USE_WEBHOOK:
                # ✅ WEBHOOK MODU
                app = await create_app()
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, host=app_config.WEBAPP_HOST, port=app_config.WEBAPP_PORT)
                await site.start()
                logger.info(f"✅ Webhook server started on port {app_config.WEBAPP_PORT}")
                
                # ✅ Bekle
                await shutdown_event.wait()
                
            else:
                # ✅ POLLING MODU - WEBHOOK TEMİZLİĞİ EKLENDİ
                logger.info("🔄 Starting long polling with webhook cleanup...")
                
                # CRITICAL: Webhook'u temizle
                try:
                    await bot.delete_webhook(drop_pending_updates=True)
                    logger.info("✅ Webhook cleared successfully")
                    await asyncio.sleep(2)  # Telegram'ın işlemesi için bekle
                except Exception as e:
                    logger.warning(f"⚠️ Webhook cleanup warning: {e}")
                
                await dispatcher.start_polling(bot)
                
    except Exception as e:
        logger.critical(f"💥 Fatal error: {e}")
        raise
    finally:
        await cleanup_resources()

     


async def create_app() -> web.Application:
    """Ana app creator - lifespan BURADA"""
    global bot, dispatcher, app_config
    
    # ✅ LIFESPAN SADECE BURADA - create_app içinde
    app = web.Application()
    
    # Route'ları önce ekle
    app.router.add_get("/", health_check)
    app.router.add_get("/health", health_check)
    app.router.add_get("/ready", readiness_check)
    
    # ✅ SONRA lifespan ile initialization
    async with lifespan(app_config):
        # Webhook setup
        if app_config.WEBHOOK_HOST:
            webhook_handler = SimpleRequestHandler(
                dispatcher=dispatcher,
                bot=bot,
                secret_token=getattr(app_config, "WEBHOOK_SECRET", None)
            )
            webhook_handler.register(app, path="/webhook/{token}")
        
        # Hooks
        app.on_startup.append(lambda app: on_startup(bot))
        app.on_shutdown.append(lambda app: on_shutdown(bot))
        
        # Aiogram setup
        setup_application(app, dispatcher, bot=bot)
    
    return app
    
# ---------------------------------------------------------------------
# POLLING MODU İÇİN SHUTDOWN DESTEĞİ
# ---------------------------------------------------------------------
async def stop_polling():
    """Polling modunu durdur"""
    global dispatcher, bot
    if dispatcher:
        await dispatcher.stop_polling()
        logger.info("✅ Polling stopped")

# Signal handler
def handle_shutdown(signum, frame) -> None:
    """Handle shutdown signals gracefully."""
    logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
    try:
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(shutdown_event.set)
        
        # Polling modu için ek
        if get_bot_mode() == "polling":
            asyncio.create_task(stop_polling())
            
    except Exception:
        shutdown_event.set()
        
        
signal.signal(signal.SIGTERM, handle_shutdown)
signal.signal(signal.SIGINT, handle_shutdown)

# ---------------------------------------------------------------------
# Utility Functions - yardımcı araçlar
# ---------------------------------------------------------------------


async def notify_admins_about_critical_error(error: Exception) -> None:
    """Notify admins about critical errors."""
    global bot
    if not bot:
        return
        
    message = f"🚨 Kritik Hata: {type(error).__name__}: {str(error)}"
    
    for admin_id in get_admins():
        try:
            await bot.send_message(admin_id, message)
        except Exception as e:
            logger.error(f"❌ Failed to send critical error to admin {admin_id}: {e}")


# SECURE MESSAGE DELETION = güvenli mesaj silme
async def secure_delete_message(bot: Bot, chat_id: int, message_id: int) -> None:
    """Güvenli mesaj silme with error handling"""
    try:
        await bot.delete_message(chat_id, message_id)
        logger.debug(f"✅ Message {message_id} securely deleted")
    except Exception as e:
        logger.warning(f"⚠️ Could not delete message {message_id}: {e}")




# oracle
#def is_oracle_environment() -> bool:
#    """Oracle Cloud environment detection"""
#    return any(env_var in os.environ for env_var in ['ORACLE', 'OCI_', 'OPC_'])

# .db temizlik
async def execute_critical_db_operation(operation_func, *args, **kwargs):
    """Kritik database işlemleri için transaction wrapper"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            result = await operation_func(*args, **kwargs)
            return result
        except aiosqlite.IntegrityError as e:
            logger.error(f"❌ Database integrity error (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                raise
        except aiosqlite.OperationalError as e:
            logger.error(f"❌ Database operational error (attempt {attempt+1}): {e}")
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected database error (attempt {attempt+1}): {e}")
            raise
        
        await asyncio.sleep(retry_delay)
    
    return None


# DATABASE TRANSACTION ROLLBACK  dedicated service'te:
# apikey_manager.py ile aynı
async def register_user_complete(self, user_id: int, user_data: dict) -> bool:
    """Tam kullanıcı kaydı için atomic transaction - GÜNCELLENMİŞ"""
    db = await self.get_db_connection()
    
    try:
        async with db.cursor() as cursor:
            # 1. Kullanıcıyı kaydet
            await cursor.execute(
                """INSERT OR IGNORE INTO users 
                   (user_id, username, first_name, language_code) 
                   VALUES (?, ?, ?, ?)""",
                (user_id, user_data.get('username'), user_data.get('first_name'), user_data.get('language_code', 'en'))
            )
            
            # 2. Varsayılan trade settings
            default_settings = [
                (user_id, 'risk_level', 'medium'),
                (user_id, 'notifications', 'true'),
                (user_id, 'auto_trade', 'false')
            ]
            
            for setting in default_settings:
                await cursor.execute(
                    "INSERT OR REPLACE INTO trade_settings (user_id, setting_key, setting_value) VALUES (?, ?, ?)",
                    setting
                )
            
            # 3. Audit log
            await cursor.execute(
                "INSERT INTO audit_log (user_id, action, details) VALUES (?, ?, ?)",
                (user_id, 'USER_REGISTERED', json.dumps(user_data))
            )
            
            await db.commit()
            logger.info(f"✅ User {user_id} registered successfully")
            return True
            
    except Exception as e:
        await db.rollback()
        logger.error(f"❌ User registration failed for {user_id}: {e}")
        return False
   
# ---------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(app_entry())
    except KeyboardInterrupt:
        logger.info("👋 Application terminated by user")
    except Exception as e:
        logger.critical(f"💥 Fatal error: {e}")
        exit(1)
