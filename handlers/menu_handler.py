# handlers/menu_handler.py


import logging
from aiogram import Router, F, types
from aiogram.filters import Command
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

logger = logging.getLogger(__name__)
router = Router(name="menu_handler")

# --- KLAVYE OLUŞTURUCULAR ---

def get_main_keyboard() -> ReplyKeyboardMarkup:
    builder = ReplyKeyboardBuilder()
    builder.row(KeyboardButton(text="💰 Fiyat"), KeyboardButton(text="📊 Analiz"),KeyboardButton(text="ℹ️ Bilgi"))
    # builder.row(KeyboardButton(text="ℹ️ Bilgi"))
    return builder.as_markup(resize_keyboard=True, placeholder="Bir seçenek seçin...")

def get_price_keyboard() -> ReplyKeyboardMarkup:
    builder = ReplyKeyboardBuilder()
    builder.row(KeyboardButton(text="/p"), KeyboardButton(text="/pwl"),KeyboardButton(text="/pv"))
    builder.row(KeyboardButton(text="/pg"), KeyboardButton(text="/pl"), KeyboardButton(text="⬅️ Ana Menü"))
    # builder.row(KeyboardButton(text="⬅️ Ana Menü"))
    return builder.as_markup(resize_keyboard=True)

def get_analysis_keyboard() -> ReplyKeyboardMarkup:
    builder = ReplyKeyboardBuilder()
    builder.row(KeyboardButton(text="/ap"), KeyboardButton(text="/toi"),KeyboardButton(text="/t"))
    builder.row(KeyboardButton(text="/ttm"), KeyboardButton(text="/tmvx"), KeyboardButton(text="/tv"))
    builder.row(KeyboardButton(text="⬅️ Ana Menü"))
    return builder.as_markup(resize_keyboard=True)

# --- HANDLERLAR ---

@router.message(Command("bot"))
@router.message(F.text == "⬅️ Ana Menü")
async def show_main_menu(message: types.Message):
    await message.answer(
        "🤖 **Ana Menüye Hoş Geldiniz**\nLütfen işlem seçin:",
        reply_markup=get_main_keyboard(),
        parse_mode="Markdown"
    )

@router.message(F.text == "💰 Fiyat")
async def show_price_menu(message: types.Message):
    await message.answer(
        "💰 **Fiyat Menüsü**\n\n/p: Watchlist\n/pv: Hacim\n/pg: Yükselenler\n/pl: Düşenler",
        reply_markup=get_price_keyboard()
    )

@router.message(F.text == "📊 Analiz")
async def show_analysis_menu(message: types.Message):
    await message.answer(
        "📊 **Analiz Menüsü**\n\n/t: Core\n/ap: Alt Power\n/toi: OI Tarama",
        reply_markup=get_analysis_keyboard()
    )

@router.message(F.text == "ℹ️ Bilgi")
async def show_info(message: types.Message):
    info_text = (
        "💡 **Kullanım İpucu**\n\n"
        "Butonlara basarak komutları hızlıca gönderebilirsiniz. "
        "Ayrıca manuel olarak `/p btc` gibi parametre de ekleyebilirsiniz."
    )
    await message.answer(info_text, parse_mode="Markdown")

# Export for handler_loader
__all__ = ['router']