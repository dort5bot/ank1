# handlers/apikey_handler.py
"""
Telegram handler for managing API keys, alarms, and trade settings.

Geliştirmeler:
- Comprehensive error handling
- Input validation
- Security improvements
- Better user feedback
- Async/await pattern compliance
Sabit (constant) değerler için büyük harflerle ve alt çizgiyle yazılır: ADMIN_IDS
Normal değişkenler ve listeler için küçük harflerle ve alt çizgiyle yazılır: ADMIN_IDS

| Komut                            | Açıklama                                                                  |
| -------------------------------- | ------------------------------------------------------------------------- |
| `/apikey <API_KEY> <SECRET_KEY>` | Binance API key’lerini kaydeder ve doğrular. Mesaj güvenlik için silinir. |
| `/getapikey`                     | Kayıtlı API key’in maskelenmiş halini ve geçerliliğini gösterir.          |
| `/validatekey`                   | Kayıtlı API key’in Binance ile geçerli olup olmadığını kontrol eder.      |
| `/deletekey`                     | Kullanıcının kayıtlı API key’ini siler.                                   |


"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, Optional, List, Any

from aiogram import Router, types, F
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.exceptions import TelegramBadRequest
from dotenv import set_key

from utils.apikey_manager import (
    APIKeyManager,
    AlarmManager,
    TradeSettingsManager,
)

logger = logging.getLogger(__name__)

router = Router()

# Singletons from utils
_api_manager: APIKeyManager = APIKeyManager.get_instance()
_alarm_manager: AlarmManager = AlarmManager.get_instance()
_trade_manager: TradeSettingsManager = TradeSettingsManager.get_instance()

# Yetkili kullanıcı listesi
#admin_ids = [8291155353, 1234567890]  # liste
ADMIN_IDS = [8291155353,  775252999]


# API key validation regex
API_KEY_REGEX = re.compile(r'^[a-zA-Z0-9]{16,64}$')
SECRET_KEY_REGEX = re.compile(r'^[a-zA-Z0-9]{32,128}$')


def _mask_api_key(api_key: str) -> str:
    """
    Mask the API key for safe display: keep first 4 and last 4 chars visible.
    """
    if not api_key or len(api_key) < 8:
        return "***"
    return api_key[:4] + "*" * (len(api_key) - 8) + api_key[-4:]


def _validate_api_keys(api_key: str, secret_key: str) -> tuple[bool, str]:
    """Validate API key format"""
    if not API_KEY_REGEX.match(api_key):
        return False, "❌ Geçersiz API key formatı"
    
    if not SECRET_KEY_REGEX.match(secret_key):
        return False, "❌ Geçersiz secret key formatı"
    
    return True, "✅ Format doğrulandı"


async def _safe_delete_message(message: Message) -> bool:
    """Safely delete message with error handling"""
    try:
        await message.delete()
        return True
    except TelegramBadRequest as e:
        logger.warning(f"⚠️ Couldn't delete message {message.message_id}: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error deleting message: {e}")
        return False


# -------------------------
# /apikey handler
# -------------------------
@router.message(Command("apikey"))
async def apikey_command(message: Message) -> None:
    logger.info(f"📥 Komut alındı: {message.text}")
    """
    Usage: /apikey <API_KEY> <SECRET_KEY>
    Saves credentials encrypted and deletes the original message for security.
    """
    user_id = message.from_user.id
    args = message.text.split()[1:]  # More reliable parsing

    if len(args) < 2:
        await message.reply(
            "❌ Kullanım: /apikey <API_KEY> <SECRET_KEY>\n\n"
            "⚠️ Bu mesaj API key'inizi içerdiği için güvenlik nedeniyle silinecektir."
        )
        return

    api_key, secret_key = args[0].strip(), args[1].strip()
    
    # Validate API key format
    is_valid, validation_msg = _validate_api_keys(api_key, secret_key)
    if not is_valid:
        await message.reply(validation_msg)
        return

    logger.info("User %s requested to save API keys (masked=%s).", user_id, _mask_api_key(api_key))

    try:
        # Validate credentials with Binance before saving
        temp_client = None
        try:
            from binance import AsyncClient
            temp_client = await AsyncClient.create(api_key, secret_key)
            await temp_client.get_account()
        except Exception as e:
            await message.reply(f"❌ Binance kimlik doğrulama başarısız: {e}")
            return
        finally:
            if temp_client:
                await temp_client.close_connection()

        # Save to DB
        await _api_manager.add_or_update_apikey(user_id, api_key, secret_key)
        logger.info("API key stored for user_id=%s", user_id)

        # Try to delete the original user message for security
        await _safe_delete_message(message)

        # Notify user
        response_msg = await message.answer(
            f"✅ API key başarıyla kaydedildi ve doğrulandı.\n"
            f"🔐 Maskelenmiş Key: `{_mask_api_key(api_key)}`\n"
            f"🛡️ Orijinal mesajınız güvenlik için silindi.",
            parse_mode="Markdown"
        )

        # If user is authorized, also write to .env
        if user_id in ADMIN_IDS:
            env_path = os.path.join(os.getcwd(), ".env")
            try:
                set_key(env_path, "BINANCE_API_KEY", api_key)
                set_key(env_path, "BINANCE_SECRET_KEY", secret_key)
                logger.info("Global .env BINANCE keys updated by authorized user %s", user_id)
                await message.answer("🔑 Global API key de güncellendi.")
            except Exception as e:
                logger.error("Failed to update .env for user %s: %s", user_id, e)
                await message.answer("⚠️ Global .env güncellenemedi (sunucu izni/IO hatası).")

    except ValueError as e:
        logger.warning(f"Validation error for user {user_id}: {e}")
        await message.reply(f"❌ Geçersiz veri: {e}")
    except Exception as e:
        logger.exception("API key kaydedilirken hata (user_id=%s): %s", user_id, e)
        await message.reply("❌ API key kaydedilirken beklenmeyen bir hata oluştu.")


# -------------------------
# /getapikey handler
# -------------------------
@router.message(Command("getapikey"))
async def get_apikey_command(message: Message) -> None:
    """Show masked API key and validation status"""
    user_id = message.from_user.id
    
    try:
        creds: Optional[tuple] = await _api_manager.get_apikey(user_id)
        if not creds:
            await message.reply("❌ Henüz API key kaydetmediniz.")
            return

        api_key, _ = creds
        masked = _mask_api_key(api_key)
        
        # Check if credentials are still valid
        is_valid = await _api_manager.validate_binance_credentials(user_id)
        status_icon = "✅" if is_valid else "❌"
        status_text = "doğrulandı" if is_valid else "geçersiz"
        
        await message.reply(
            f"{status_icon} Kayıtlı API Key: `{masked}`\n"
            f"📊 Durum: {status_text}",
            parse_mode="Markdown"
        )
        
    except Exception as e:
        logger.exception("API key okunamadı (user_id=%s): %s", user_id, e)
        await message.reply("❌ API key okunurken hata oluştu.")


# -------------------------
# /validatekey handler (YENİ)
# -------------------------
@router.message(Command("validatekey"))
async def validate_key_command(message: Message) -> None:
    """Validate current API key with Binance"""
    user_id = message.from_user.id
    
    try:
        await message.reply("🔍 API key doğrulanıyor...")
        
        is_valid = await _api_manager.validate_binance_credentials(user_id)
        
        if is_valid:
            await message.reply("✅ API key geçerli ve Binance ile başarıyla bağlantı kuruldu.")
        else:
            await message.reply(
                "❌ API key geçersiz veya bağlantı başarısız.\n"
                "🔑 Lütfen API key'lerinizi /apikey komutuyla yeniden güncelleyin."
            )
            
    except Exception as e:
        logger.exception("Key validation failed for user %s: %s", user_id, e)
        await message.reply("❌ Doğrulama işlemi sırasında hata oluştu.")


# -------------------------
# /deletekey handler (YENİ)
# -------------------------
@router.message(Command("deletekey"))
async def delete_key_command(message: Message) -> None:
    """Delete stored API key"""
    user_id = message.from_user.id
    
    try:
        success = await _api_manager.delete_apikey(user_id)
        
        if success:
            await message.reply("✅ API key başarıyla silindi.")
        else:
            await message.reply("❌ Silinecek API key bulunamadı.")
            
    except Exception as e:
        logger.exception("Key deletion failed for user %s: %s", user_id, e)
        await message.reply("❌ API key silinirken hata oluştu.")


# -------------------------
# Alarm settings handlers
# -------------------------
@router.message(Command("set_alarm"))
async def set_alarm_command(message: Message) -> None:
    """Set alarm settings with JSON data"""
    user_id = message.from_user.id
    args_text = message.get_args().strip()
    
    if not args_text:
        await message.reply(
            "❌ Lütfen JSON formatında alarm ayarları girin.\n\n"
            "📋 Örnek:\n"
            "```json\n"
            '{"symbol": "BTCUSDT", "threshold": 50000, "condition": "above"}'
            "```",
            parse_mode="Markdown"
        )
        return

    try:
        data: Dict[str, Any] = json.loads(args_text)
        
        # Basic validation
        if not isinstance(data, dict):
            await message.reply("❌ Alarm ayarları bir JSON objesi olmalı.")
            return
            
        if not data:
            await message.reply("❌ Alarm ayarları boş olamaz.")
            return
            
        await _alarm_manager.set_alarm_settings(user_id, data)
        
        logger.info("Alarm ayarları kaydedildi (user_id=%s). Data: %s", user_id, data)
        await message.reply(
            f"✅ Alarm ayarları kaydedildi.\n"
            f"📊 Ayarlar: {list(data.keys())}"
        )
        
    except json.JSONDecodeError as e:
        logger.warning("set_alarm: JSON decode error for user %s: %s", user_id, e)
        await message.reply(
            "❌ Geçersiz JSON formatı.\n"
            "📋 Lütfen doğru JSON syntax'ı kullanın."
        )
    except ValueError as e:
        await message.reply(f"❌ Geçersiz veri: {e}")
    except Exception as e:
        logger.exception("Alarm ayarları kaydedilemedi (user_id=%s): %s", user_id, e)
        await message.reply("❌ Alarm ayarları kaydedilirken beklenmeyen bir hata oluştu.")


@router.message(Command("get_alarm"))
async def get_alarm_command(message: Message) -> None:
    """Get current alarm settings"""
    user_id = message.from_user.id
    
    try:
        settings = await _alarm_manager.get_alarm_settings(user_id)
        if not settings:
            await message.reply("❌ Alarm ayarınız bulunamadı.")
            return

        pretty = json.dumps(settings, indent=2, ensure_ascii=False)
        await message.reply(
            f"🔔 Alarm ayarları:\n<code>{pretty}</code>", 
            parse_mode="HTML"
        )
        
    except Exception as e:
        logger.exception("Alarm ayarları alınamadı (user_id=%s): %s", user_id, e)
        await message.reply("❌ Alarm ayarları okunurken hata oluştu.")


# -------------------------
# Trade settings handlers
# -------------------------
@router.message(Command("set_trade"))
async def set_trade_command(message: Message) -> None:
    """Set trade settings with JSON data"""
    user_id = message.from_user.id
    args_text = message.get_args().strip()
    
    if not args_text:
        await message.reply(
            "❌ Lütfen JSON formatında trade ayarları girin.\n\n"
            "📋 Örnek:\n"
            "```json\n"
            '{"max_trade": 100, "risk_per_trade": 2, "stop_loss": 5}'
            "```",
            parse_mode="Markdown"
        )
        return

    try:
        data: Dict[str, Any] = json.loads(args_text)
        
        # Basic validation
        if not isinstance(data, dict):
            await message.reply("❌ Trade ayarları bir JSON objesi olmalı.")
            return
            
        if not data:
            await message.reply("❌ Trade ayarları boş olamaz.")
            return
            
        await _trade_manager.set_trade_settings(user_id, data)
        
        logger.info("Trade ayarları kaydedildi (user_id=%s). Data: %s", user_id, data)
        await message.reply(
            f"✅ Trade ayarları kaydedildi.\n"
            f"📈 Ayarlar: {list(data.keys())}"
        )
        
    except json.JSONDecodeError as e:
        logger.warning("set_trade: JSON decode error for user %s: %s", user_id, e)
        await message.reply(
            "❌ Geçersiz JSON formatı.\n"
            "📋 Lütfen doğru JSON syntax'ı kullanın."
        )
    except ValueError as e:
        await message.reply(f"❌ Geçersiz trade ayarı: {e}")
    except Exception as e:
        logger.exception("Trade ayarları kaydedilemedi (user_id=%s): %s", user_id, e)
        await message.reply("❌ Trade ayarları kaydedilirken beklenmeyen bir hata oluştu.")


@router.message(Command("get_trade"))
async def get_trade_command(message: Message) -> None:
    """Get current trade settings"""
    user_id = message.from_user.id
    
    try:
        settings = await _trade_manager.get_trade_settings(user_id)
        if not settings:
            await message.reply("❌ Trade ayarınız bulunamadı.")
            return

        pretty = json.dumps(settings, indent=2, ensure_ascii=False)
        await message.reply(
            f"📊 Trade ayarları:\n<code>{pretty}</code>", 
            parse_mode="HTML"
        )
        
    except Exception as e:
        logger.exception("Trade ayarları alınamadı (user_id=%s): %s", user_id, e)
        await message.reply("❌ Trade ayarları okunurken hata oluştu.")


# -------------------------
# /help handler
# -------------------------
@router.message(Command("help"))
async def help_command(message: Message) -> None:
    """Show available commands"""
    help_text = """
🤖 **API Key Yönetim Botu**

**Temel Komutlar:**
🔐 `/apikey <api_key> <secret_key>` - API key kaydet
📋 `/getapikey` - Kayıtlı API key'i göster
✅ `/validatekey` - API key doğrula
🗑️ `/deletekey` - API key sil

**Alarm Ayarları:**
🔔 `/set_alarm {json}` - Alarm ayarlarını kaydet
📊 `/get_alarm` - Alarm ayarlarını göster

**Trade Ayarları:**
📈 `/set_trade {json}` - Trade ayarlarını kaydet
📉 `/get_trade` - Trade ayarlarını göster

**Güvenlik:**
🛡️ API key'leriniz şifrelenerek saklanır
🗑️ Orijinal mesajlar güvenlik için silinir
    """
    
    await message.reply(help_text, parse_mode="Markdown")


# -------------------------
# Cleanup task (periodic)
# -------------------------
async def periodic_cleanup():
    """Periodic cleanup task for cache and old data"""
    import asyncio
    
    while True:
        try:
            # Cleanup old cache entries
            await _api_manager.cleanup_cache(max_size=500)
            
            # Cleanup old alarms (older than 30 days)
            await _alarm_manager.cleanup_old_alarms(days=30)
            
            # Cleanup old API keys (older than 90 days)
            await _trade_manager.cleanup_old_apikeys(days=90)
            
            logger.info("✅ Periodic cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Periodic cleanup failed: {e}")
        
        # Run every hour
        await asyncio.sleep(3600)


# -------------------------
# Startup initialization
# -------------------------
async def init_apikey_handler():
    """Initialize database and start cleanup task"""
    try:
        await _api_manager.init_db()
        logger.info("✅ API Key Handler initialized successfully")
        
        # Start cleanup task
        import asyncio
        asyncio.create_task(periodic_cleanup())
        
    except Exception as e:
        logger.error(f"❌ API Key Handler initialization failed: {e}")
        raise


# -------------------------
# Shutdown cleanup
# -------------------------
async def shutdown_apikey_handler():
    """Cleanup resources on shutdown"""
    try:
        await _api_manager.close_connections()
        logger.info("✅ API Key Handler shutdown completed")
    except Exception as e:
        logger.error(f"❌ API Key Handler shutdown failed: {e}")
        
