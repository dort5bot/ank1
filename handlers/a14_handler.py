# handlers/a7_handler.py - REVIZE EDILMIS VERSIYON
"""
OPTIMIZED COMMAND HANDLER - TÜM KOMUTLAR İÇİN ORTAK MANTIK

KULLANIM:
/t           → Default 7 coin
/t 5         → Hacimli ilk 5 coin
/t BTC       → Sadece BTC
/t BTC ETH   → BTC ve ETH
/t 10 SOL    → Hacimli 10 coin + SOL

/ts BTC          → BTCUSDT için sentiment analizi
/ts 5            → Hacimli ilk 5 coin sentiment analizi
/ts              → Default watchlist sentiment analizi

TÜM KOMUTLAR AYNI MANTIKLA ÇALIŞIR.

TEK - ZORUNLU BLOK (sadece bunlar)
CORE
REGF
VOL_STATE
STRESS

| Metrik    | Telegram’da anlamı     |
| --------- | ---------------------- |
| CORE      | Long / Short bias      |
| REGF      | Hangi strateji çalışır |
| VOL_STATE | Pozisyon & stop        |
| STRESS    | Risk-off alarm         |

AYRI RAPORDA GÖSTERİLMELİ (🧪 filtre / teyit)
trend
mom
vol
sentp> sntp
complexity



ASLA GÖSTERME (🚫 Telegram’da yeri yok)
Bunlar hesaplanıyor olabilir ama kullanıcıya sunulmamalı
entropy
sentiment

| Grup       | Gösterim     |
| ---------- | ------------ |
| core       | ✅ Tek rapor  |
| regf       | ✅ Tek rapor  |
| vol_state  | ✅ Tek rapor  |
| stress     | ✅ Tek rapor  |
| trend      | ➕ Ayrı rapor |
| mom        | ➕ Ayrı rapor |
| vol        | ➕ Ayrı rapor |
| sentp      | ➕ Ayrı rapor |
| complexity | ➕ Ayrı rapor |
| entropy    | ❌ Gösterme   |
| sentiment  | ❌ Gösterme   |



"""

import logging
import asyncio
import math

from typing import Dict, List, Any, Optional
from aiogram import Router, types
from analysis.a_core import run_pipeline, get_top_volume_symbols

logger = logging.getLogger(__name__)
router = Router(name="command_router")

# ✅ TÜM KOMUTLAR - SADECE SCORES LİSTESİ
COMMANDS = {
    # tekil Başarılılar
    # -----------------------------
    "/tat": ["trend"],
    "/tam": ["mom"],
    "/tav": ["vol"], #ağır
    "/tavs": ["vols"], 
    
    "/tas": ["sntp"],   # DB / süreç ŞART
    "/taz":["strs"],    # DB / süreç ŞART
    
    "/tac": ["core"],
    "/taf": ["regf"],
    "/tar": ["risk"],
    "/tare": ["regim"],
    "/taen": ["entropy"],
    "/taam": ["trend","mom","vol"],
    "/taps": ["trend","mom","vol","regim","entropy","risk"],
    # -----------------------------


    
    # Ne yapmalı
    "/t": ["core","regf","vols"],    #["core","regf","vols","strs"],
    
    # Trend netse: Yön,Güç,Katılım (fake mi değil mi)
    "/tt": ["trend","mom"],  #["trend","mom","sntp"],
    # Kararsız / yatay piyasa
    "/tk": ["mom","vol","cpxy"],
    # Volatil dönem / haber öncesi
    "/tv": ["vol","vols","cpxy"],    #["vol","vols","sntp","cpxy"],
    # detay
    "/tb": ["trend","mom","vol","cpxy"], #"sntp"
    
}

class UnifiedCommandHandler:
    """Tüm komutlar için ortak handler - REVIZE EDILMIS"""
    
    def __init__(self):
        self.commands = COMMANDS
        
        # ✅ DEFAULT WATCHLIST
        self.default_watchlist = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"
        ]
        
        # ✅ MAXIMUM COIN SAYISI
        self.max_coins = 15
        
        logger.info("✅ Unified Command Handler initialized")
    
    async def handle(self, text: str) -> Dict[str, Any]:
        """Tüm komutları işle - TEK MANTIK"""
        parts = text.strip().split()
        if not parts or parts[0] not in self.commands:
            return None
            
        cmd = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        logger.info(f"🔄 Processing: {cmd}, args: {args}")
        
        try:
            # 1. Sembolleri belirle
            symbols = await self._resolve_symbols(args)
            if not symbols:
                return {"error": "Geçersiz sembol veya argüman"}
            
            # 2. Limit kontrolü
            if len(symbols) > self.max_coins:
                logger.warning(f"⚠️ Too many symbols ({len(symbols)}), limiting to {self.max_coins}")
                symbols = symbols[:self.max_coins]
            
            # 3. Required scores'u al
            required_scores = self.commands[cmd]
            
            # 4. Tüm sembolleri paralel analiz et
            symbol_scores = {}
            failed_symbols = []
            volume_based = self._is_volume_based(args)
            
            for symbol in symbols:
                result = await self._analyze_symbol(
                    symbol=symbol,
                    required_scores=required_scores
                )
                
                # YENİ HALİ:
                if result and "error" not in result:
                    scores = self._extract_scores(result, required_scores, symbol)
                    
                    if scores:  # <-- Sadece scores dict boş değilse
                        symbol_scores[symbol] = scores
                        logger.info(f"✅ {symbol} - Analysis complete")
                    else:
                        failed_symbols.append(symbol)
                        logger.warning(f"❌ {symbol} - No real data")

                else:
                    failed_symbols.append(symbol)
                    error_msg = result.get("error", "Unknown error") if result else "No result"
                    logger.warning(f"❌ {symbol} - Analysis failed: {error_msg}")
            
            # 5. Sonuçları düzenle
            if not symbol_scores:
                return {"error": "No real data for any symbol"}
            
            return {
                "command": cmd,
                "command_name": self._get_command_name(cmd),
                "symbols": list(symbol_scores.keys()),
                "symbol_scores": symbol_scores,
                "scores": required_scores,
                "failed_symbols": failed_symbols,
                "volume_based": volume_based,
                "symbol_count": len(symbols),
                "success_count": len(symbol_scores),
            }
            
        except Exception as e:
            logger.error(f"❌ Command failed: {e}", exc_info=True)
            return {"error": f"Processing error: {str(e)}"}
      
    async def _resolve_symbols(self, args: List[str]) -> List[str]:
        # Durum 1: Argüman sayı mı? (/t 5)
        if args and args[0].isdigit():
            n = int(args[0])
            return await get_top_volume_symbols(count=n)
        
        # Durum 2: Argümanlar sembol mü? (/t btc sol)
        if args:
            symbols = []
            for arg in args:
                normalized = self._normalize_symbol(arg)
                if normalized:
                    symbols.append(normalized)
            return symbols
        
        # Durum 3: Boş sorgu (/t)
        return self.default_watchlist    
        
    
    def _is_volume_based(self, args: List[str]) -> bool:
        """Argümanlar hacim bazlı mı?"""
        return bool(args and args[0].isdigit())
    
    def _normalize_symbol(self, symbol_input: str) -> Optional[str]:
        """Sembol normalizasyonu"""
        if not symbol_input or not symbol_input.strip():
            return None
        
        clean = symbol_input.upper().strip()
        
        # USDT ekle (yoksa)
        if not clean.endswith('USDT'):
            # Kısaltma kontrolü
            if clean == 'BTC':
                return 'BTCUSDT'
            elif clean == 'ETH':
                return 'ETHUSDT'
            elif clean == 'BNB':
                return 'BNBUSDT'
            elif clean == 'SOL':
                return 'SOLUSDT'
            elif clean == 'XRP':
                return 'XRPUSDT'
            elif clean == 'ADA':
                return 'ADAUSDT'
            elif clean == 'DOGE':
                return 'DOGEUSDT'
            else:
                return f"{clean}USDT"
        
        return clean
    
    def _get_command_name(self, cmd: str) -> str:
        """Komut için açıklayıcı isim"""
        names = {
            "/t": "CORE ANALYSIS",
            "/ts": "SENTIMENT & FLOW",
            "/tm": "MICROSTRUCTURE",
            "/tt": "TREND",
            "/tv": "VOLATILITY",
            "/tvm": "VOLATILITY MOMENTUM",
            "/ten": "ENTROPY",
            "/tre": "REGIME",
            "/tri": "RISK",
            "/tse": "SENTIMENT",
            "/tl": "LIQUIDITY",
            "/tlr": "LIQUIDITY RISK",
            "/tor": "ORDER FLOW",
            "/tfl": "FLOW DYNAMICS",
            "/tc": "COMPLEXITY",
            "/ta": "REGIME ANALYSIS",
            "/tr": "RISK ANALYSIS",
            "/te": "ENTROPY ANALYSIS",
            "/tcc": "CORE + LIQUIDITY",
            "/tvv": "VOLATILITY SUITE",
        }
        return names.get(cmd, cmd.upper())
    
    async def _analyze_symbol(self, symbol: str, required_scores: List[str]) -> Dict[str, Any]:
        """Core pipeline'ını çağır"""
        try:
            # Timeout ile analiz
            result = await run_pipeline(
                symbol=symbol,
                requested_scores=required_scores,
                interval="1h",
                limit=100
            )
            return result
        except asyncio.TimeoutError:
            logger.error(f"⏰ Timeout for {symbol}")
            return {"error": "Analysis timeout"}
        except Exception as e:
            logger.error(f"❌ Core analysis failed for {symbol}: {e}")
            return {"error": str(e)}
    

    def _extract_scores(self, result: Dict, required_scores: List[str], symbol: str) -> Dict[str, float]:
        """Core'dan gelen skorları çıkar"""
        scores = {}
        
        # DEBUG logging - DETAYLI
        logger.info(f"📊 EXTRACT_SCORES for {symbol}")
        logger.info(f"  Required scores: {required_scores}")
        
        # Tüm skor kaynakları
        all_scores = result.get("scores", {})
        composites = result.get("composites", {})
        macros = result.get("macros", {})
        metrics = result.get("metrics", {})
        
        logger.info(f"  All scores dict: {all_scores}")
        logger.info(f"  Composites dict: {composites}")
        logger.info(f"  Macros dict: {macros}")
        logger.info(f"  Metrics dict: {metrics}")
        
        for score_name in required_scores:
            display_name = score_name.upper()
            raw_value = None
            
            # Sırayla ara
            if score_name in all_scores:
                raw_value = all_scores[score_name]
                logger.info(f"  ✅ {score_name} found in scores: {raw_value} (type: {type(raw_value)})")
            elif score_name in composites:
                raw_value = composites[score_name]
                logger.info(f"  ✅ {score_name} found in composites: {raw_value} (type: {type(raw_value)})")
            elif score_name in macros:
                raw_value = macros[score_name]
                logger.info(f"  ✅ {score_name} found in macros: {raw_value} (type: {type(raw_value)})")
            elif score_name in metrics:
                raw_value = metrics[score_name]
                logger.info(f"  ✅ {score_name} found in metrics: {raw_value} (type: {type(raw_value)})")
            else:
                logger.info(f"  ❌ {score_name} NOT FOUND anywhere")
            
            # Değeri işle
            if raw_value is None:
                scores[display_name] = float('nan')
                logger.info(f"  → {score_name} set to NaN (raw is None)")
            elif isinstance(raw_value, float) and math.isnan(raw_value):
                scores[display_name] = float('nan')
                logger.info(f"  → {score_name} set to NaN (raw is NaN)")
            elif isinstance(raw_value, (int, float)):
                # Clip and round
                clipped = max(-1.0, min(1.0, float(raw_value)))
                scores[display_name] = round(clipped, 3)
                logger.info(f"  → {score_name} final: {scores[display_name]} (clipped from {raw_value})")
            else:
                # Try to convert
                try:
                    val = float(raw_value)
                    clipped = max(-1.0, min(1.0, val))
                    scores[display_name] = round(clipped, 3)
                    logger.info(f"  → {score_name} converted: {scores[display_name]}")
                except:
                    scores[display_name] = float('nan')
                    logger.info(f"  → {score_name} set to NaN (conversion failed)")
        
        logger.info(f"📊 FINAL scores dict for {symbol}: {scores}")
        return scores


# ✅ TEK HANDLER INSTANCE
handler = UnifiedCommandHandler()

# ✅ FORMAT FONKSİYONU
def format_table_response(result: Dict[str, Any]) -> str:
    """Sonuçları formatla"""
    
    # ✅ HATA DURUMU İÇİN ÖZEL MESAJ
    if "error" in result:
        return f"❌ <b>Hata:</b> {result['error']}"
    
    symbol_scores = result["symbol_scores"]
    
    # ✅ EĞER HİÇ SEMBOL YOKSA
    if not symbol_scores:
        if result.get("volume_based"):
            return "❌ <b>Hacim Verisi Alınamadı</b>\n\nBinance'den 24 saatlik hacim verisi alınamadı. Lütfen daha sonra tekrar deneyin."
        else:
            return "❌ <b>Analiz Başarısız</b>\n\nHiçbir sembol için analiz yapılamadı."
    
    scores = result["scores"]
    headers = [s.upper() for s in scores]
    
    # Başlık
    if result.get("volume_based"):
        title = f"📈 <b>{result['command_name']}</b> - Top {result['symbol_count']} Volume Coins"
    else:
        title = f"📊 <b>{result['command_name']}</b> - {result['success_count']} Coins"
    
    # Header satırı
    header_cells = ["Sembol"] + headers
    header_line = "  ".join([f"{cell:10}" for cell in header_cells])
    
    lines = [
        title,
        "─" * (5 + len(headers) * 6),
        f"<b>{header_line}</b>",
        "─" * (5 + len(headers) * 6)
    ]
    
    # Sembolleri sırala - hacim bazlıysa zaten sıralı gelir
    if result.get("volume_based"):
        sorted_symbols = list(symbol_scores.keys())  # Hacim sırasını koru
    else:
        sorted_symbols = sorted(symbol_scores.keys())
    
    for symbol in sorted_symbols:
        scores_dict = symbol_scores[symbol]
        display_symbol = symbol.replace('USDT', '')
        
        # Score hücreleri
        score_cells = [f"{display_symbol:8}"]
        for header in headers:
            value = scores_dict.get(header, float('nan'))
            
            if isinstance(value, float) and math.isnan(value):
                score_cells.append(f"{get_icon(header, None):2} ---")
            else:
                icon = get_icon(header, value)
                formatted = f"{value:+.3f}"
                score_cells.append(f"{icon:2} {formatted:7}")
        
        line = "  ".join(score_cells)
        lines.append(line)
    
    # Özet
    failed_count = len(result.get('failed_symbols', []))
    success_count = result['success_count']
    total_count = result['symbol_count']
    
    summary_lines = [
        "─" * (5 + len(headers) * 6),
        f"<b>Özet:</b> {success_count}/{total_count} başarılı"
    ]
    
    if failed_count > 0:
        failed_display = [s.replace('USDT', '') for s in result.get('failed_symbols', [])]
        if failed_display:
            summary_lines.append(f"<i>Başarısız: {', '.join(failed_display)}</i>")
    
    if result.get("volume_based"):
        summary_lines.append("<i>24 saatlik işlem hacmine göre sıralanmıştır</i>")
    
    lines.extend(summary_lines)
    
    # Help text
    help_text = get_help_text(result["command"])
    if help_text:
        lines.append("")
        lines.append(f"<i>{help_text}</i>")
    
    return "\n".join(lines)


def get_icon(column: str, score: Optional[float]) -> str:
    """Unified color-only indicator (no arrows, no extra icons)"""

    if score is None or math.isnan(score):
        return "—"

    if score >= 0.35:
        return "🟢"
    elif score >= 0.15:
        return "🟡"
    elif score > -0.15:
        return "⚪"
    elif score > -0.35:
        return "🟠"
    else:
        return "🔴"




def get_help_text(cmd: str) -> str:
    """Komut için yardım metni"""
    helps = {
        "/t": ("Ne yapmalı", ["core", "regf", "vols"]),
        "/tt": ("Yön, Güç, Katılım", ["trend", "mom"]),
        "/tk": ("Kararsız / yatay piyasa varsa", ["mom", "vol", "cpxy"]),
        "/tv": ("Volatil dönemde", ["vol", "vols", "cpxy"]),
        "/tb": ("Bilgi / detay", ["trend", "mom", "vol", "cpxy"]),
    }

    if cmd in helps:
        text, tags = helps[cmd]
        return f"{text} | Modüller: {', '.join(tags)}"

    return f"Use: {cmd} [SYMBOL] or {cmd} [NUMBER]"


# ✅ MESSAGE HANDLER
@router.message(lambda msg: msg.text and msg.text.split()[0].lower() in COMMANDS)
async def handle_all_messages(message: types.Message):
    text = message.text or ""
    
    if not text.startswith('/'):
        return
    
    # 1. ADIM: Gelen mesajın ilk kelimesini (komutu) al
    parts = text.split()
    cmd = parts[0].lower() if parts else ""

    # 2. ADIM: Komut bizim COMMANDS listemizde mi? 
    # Değilse sessizce çık (böylece /dar gibi komutlara tepki vermez)
    if cmd not in COMMANDS:
        return
    
    # Loading mesajı (Sadece geçerli bir komutsa gösterilir)
    loading_msg = await message.answer("⏳ Analiz ediliyor...")
    
    try:
        result = await handler.handle(text)
        
        # result None ise veya hata varsa işleme devam et
        if result is None:
            await loading_msg.delete() # Veya hata mesajı
            return
            
        if "error" in result:
            await loading_msg.edit_text(f"⚠️ <b>Hata:</b> {result['error']}")
            return
        
        response = format_table_response(result)
        await loading_msg.edit_text(response, parse_mode="HTML")
        
    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        await loading_msg.edit_text(f"❌ <b>Sistem hatası:</b> {str(e)}")