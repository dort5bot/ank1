# handlers/a14_handler.py - REVIZE EDILMIS VERSIYON
"""
KULLANIM:
/ap  → özel komut, get_alt_power INDEX_BASKET yardımıyla hesaplaması yapacak
/t           → izleme listesi
/t 5         → Hacimli ilk 5 coin
/t BTC       → Sadece BTC
/t BTC ETH   → BTC ve ETH
belki eklenecek > /t 10 SOL    → Hacimli 10 coin + SOL

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
# from analysis.a_core import run_pipeline, calculate_alt_power
# from analysis.a_core import run_full_analysis, get_alt_power, get_top_volume_symbols
from analysis.db_loader import load_latest_snapshots

from analysis.a_core import (
    run_full_analysis, 
    get_alt_power, 
    get_top_volume_symbols,
    WATCHLIST,  # ✅ Core'dan import edin
    INDEX_BASKET  # ✅ İhtiyacınız olursa
)


logger = logging.getLogger(__name__)
router = Router(name="analiz_handler")

# ✅ TÜM KOMUTLAR - SADECE SCORES LİSTESİ
COMMANDS = {
    # tekil Başarılılar
    # -----------------------------
    "/t": ["trend","mom","vol"],
    "/tc": ["core"],
    
    "/tam": ["mom"],
    "/tav": ["vol"], #ağır
    "/tavs": ["vols"], 
    
    "/tas": ["sntp"],   # DB / süreç ŞART
    "/taz":["strs"],    # DB / süreç ŞART
    "/taf": ["regf"],
    "/tar": ["risk"],
    "/tare": ["regim"],
    "/taen": ["entropy"],
    "/taam": ["trend"],
    "/taps": ["trend","mom","vol","regim","entropy","risk"],
    # -----------------------------
    "/ap": "INDEX_MODE", # Özel mod

    # Ne yapmalı
    "/tuz": ["core","regf","vols"],    #["core","regf","vols","strs"],
    
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
        # self.default_watchlist = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"        ]
        self.default_watchlist = WATCHLIST  # Core'daki listeyi kullan
        # VEYA isterseniz core'daki listeyi genişletebilirsiniz:
        # self.default_watchlist = WATCHLIST + ["ARPAUSDT", "ALICEUSDT"]
        
        
        # ✅ MAXIMUM COIN SAYISI
        self.max_coins = 15
        
        # ✅ Core'daki diğer listeleri de kullanabilirsiniz
        # self.index_basket = INDEX_BASKET  # Sadece referans için
        
        logger.info("✅ Unified Command Handler initialized")
    
    # handle metodu artık devasa bir if-else yığını değil. 
    # Sadece komutu tanıyor ve ilgili "uzman" metoda (_handle_table) paslıyor.
    

    async def handle(self, text: str) -> dict:
        parts = text.split()
        if not parts:
            return {"error": "Boş komut"}

        cmd = parts[0].lower()
        args = parts[1:]

        # 1. YARDIM KOMUTU
        if args and args[0] in ["?", "help", "yardım"]:
            return {"type": "HELP", "command": cmd}

        # 2. KOMUT KONTROLÜ (Girinti Düzeltildi)
        if cmd in self.commands:
            # Eğer komut /ap ise özel metodu çağır
            if cmd == "/ap":
                return await self._handle_alt_power(cmd, args)
            
            # Diğer tüm komutlar (/t, /tt, /tv vb.) için tablo mantığı
            return await self._handle_table(cmd, args)

        # 3. TANIMSIZ KOMUT
        return {"error": f"Komut işleme mantığı bulunamadı: {cmd}"}
            

    """
        Tablo tabanlı komutları (Watchlist, Top N, Tekil Coin) yönetir.
        target: Hangi semboller
        cmd: Hangi komut (/t, /tat, /tv vb.)
        hangi metrikleri hesaplar
        core ile iletişim bölümü 
    """

    async def _handle_table(self, cmd: str, args: list) -> dict:
        """
        Tablo tabanlı komutları (Watchlist, Top N, Tekil Coin) yönetir.
        """
        try:
            # 1. Hangi semboller analiz edilecek? ✅
            if not args:
                # /t komutu boşsa default watchlist
                symbols = self.default_watchlist
                volume_based = False
                
            elif args[0].isdigit():
                # /t 5 → Hacimli ilk N coin
                n = int(args[0])
                symbols = await get_top_volume_symbols(count=n)
                volume_based = True
                
            else:
                # /t BTC veya /t BTC ETH SOL
                symbols = []
                for arg in args:
                    normalized = self._normalize_symbol(arg)
                    if normalized:
                        symbols.append(normalized)
                volume_based = False
            
            if not symbols:
                return {"error": "Analiz için sembol bulunamadı"}
            
            # 2. Hangi metrikler hesaplanacak? ✅
            requested_metrics = self.commands.get(cmd, [])
            if not requested_metrics:
                return {"error": f"Komut '{cmd}' için metrik tanımı bulunamadı"}
            
            # 3. Core'u çağır ✅ - TÜM sembolleri bir kerede gönder
            # run_full_analysis() sembol listesi bekler, tek sembol değil
            result = await run_full_analysis(
                symbols=symbols,  # Bu önemli: liste olarak
                metrics=requested_metrics,
                interval="1h",
                limit=100
            )
            
            # 4. Sonuçları işle ✅
            symbol_scores = {}
            failed_symbols = []
            
            # result yapısı: {"market_context": {...}, "results": {symbol1: {...}, symbol2: {...}}}
            all_results = result.get("results", {})
            
            for symbol in symbols:
                symbol_result = all_results.get(symbol)
                
                if not symbol_result or symbol_result.get("status") != "success":
                    failed_symbols.append(symbol)
                    continue
                
                # Skorları çıkar
                scores = self._extract_scores_from_result(symbol_result, requested_metrics, symbol)
                symbol_scores[symbol] = scores
            
            # 5. Sonuçları formatla ✅
            return {
                "type": "TABLE",
                "command": cmd,
                "command_name": self._get_command_name(cmd),
                "symbol_scores": symbol_scores,
                "failed_symbols": failed_symbols,
                "success_count": len(symbol_scores),
                "symbol_count": len(symbols),
                "volume_based": volume_based,
                "scores": requested_metrics,
                "market_context": result.get("market_context", {})  # Market context'i de ekle
            }
            
        except Exception as e:
            logger.error(f"Handler _handle_table hatası: {e}", exc_info=True)
            return {"error": f"Analiz motoru hatası: {str(e)}"}
            


    def _extract_scores_from_result(self, symbol_result: Dict, requested_metrics: List[str], symbol: str) -> Dict[str, float]:
        """Core'dan gelen sonuçtan skorları çıkar"""
        scores = {}
        
        logger.info(f"📊 EXTRACT_SCORES for {symbol}")
        logger.info(f"  Requested metrics: {requested_metrics}")
        
        # Core'un dönüş formatı:
        # {
        #   "symbol": "...",
        #   "status": "success",
        #   "scores": {...},  # COMPOSITES ve MACROS burada
        #   "raw_metrics": {...},  # Ham metrikler burada
        #   "timestamp": "..."
        # }
        
        all_scores = symbol_result.get("scores", {})
        
        logger.info(f"  All scores dict from core: {all_scores}")
        
        for metric_name in requested_metrics:
            display_name = metric_name.upper()
            raw_value = all_scores.get(metric_name)
            
            if raw_value is None:
                logger.info(f"  ❌ {metric_name} not found in scores")
                scores[display_name] = float('nan')
                continue
            
            logger.info(f"  ✅ {metric_name} found: {raw_value} (type: {type(raw_value)})")
            
            # Değeri işle
            try:
                if isinstance(raw_value, (int, float)):
                    if math.isnan(raw_value):
                        scores[display_name] = float('nan')
                    else:
                        # Clip and round
                        clipped = max(-1.0, min(1.0, float(raw_value)))
                        scores[display_name] = round(clipped, 3)
                else:
                    # Try to convert
                    val = float(raw_value)
                    clipped = max(-1.0, min(1.0, val))
                    scores[display_name] = round(clipped, 3)
            except Exception as e:
                logger.error(f"  ⚠️ Error processing {metric_name}: {e}")
                scores[display_name] = float('nan')
        
        logger.info(f"📊 FINAL scores for {symbol}: {scores}")
        return scores
    

    # Alt Power (Index) analizini yönetir
    async def _handle_alt_power(self, cmd: str, args: list) -> dict:
        try:
            from analysis.a_core import get_alt_power
            # a_core.py'daki get_alt_power artık parametresiz çalışabiliyor
            scores = await get_alt_power() 
            return {
                "type": "INDEX_REPORT",
                "command": cmd,
                "data": scores
            }
        except Exception as e:
            return {"error": f"Alt Power hatası: {str(e)}"}
            
      
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
        """Core pipeline'ını standardize edilmiş 'metrics' parametresi ile çağırır."""
        try:
            # Core artık 'metrics' ismini bekliyor
            result = await run_full_analysis(
                symbol=symbol,
                interval="1h",
                limit=100,
                metrics=required_scores  # Burada eşleştirmeyi yaptık
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
def format_table_response(result: dict) -> str:
    """Sonuçları formatla: TABLE ve INDEX_REPORT tiplerini destekler"""

    import math

    # Hata varsa
    if "error" in result:
        return f"❌ <b>Hata:</b> {result['error']}"

    # -----------------------------
    # INDEX_REPORT (Ör. /ap)
    # -----------------------------
    if result.get("type") == "INDEX_REPORT":
        d = result.get("data", {})
        if not d:
            return "❌ <b>Analiz hatası:</b> Veri bulunamadı."

        # Skorlara göre basit renk ikonları
        def get_trend_icon(val): 
            if val is None: return "—"
            return "🟢" if val > 60 else "🔴" if val < 40 else "🟡"

        return (
            f"📊 <b>ALT MARKET POWER</b>\n"
            f"───────────────────\n"
            f"{get_trend_icon(d.get('alt_vs_btc_short'))} <b>Alt vs BTC (Kısa):</b> <code>{d.get('alt_vs_btc_short')}</code>\n"
            f"{get_trend_icon(d.get('alt_short_term'))} <b>Alt Gücü (Kısa):</b> <code>{d.get('alt_short_term')}</code>\n"
            f"{get_trend_icon(d.get('coin_long_term'))} <b>Yapısal Güç (OI):</b> <code>{d.get('coin_long_term')}</code>\n"
            f"───────────────────\n"
            f"<i>Filtre: {len(d.get('INDEX_BASKET', []))} coinlik sepet analizi.</i>"
        )

    # -----------------------------
    # TABLE tipi (Ör. /t, /tv vb.)
    # -----------------------------
    if result.get("type") == "TABLE":
        symbol_scores = result.get("symbol_scores", {})  # <-- güvenli erişim
        if not symbol_scores:
            if result.get("volume_based"):
                return "❌ <b>Hacim Verisi Alınamadı</b>\n\nBinance'den 24 saatlik hacim verisi alınamadı. Lütfen daha sonra tekrar deneyin."
            else:
                return "❌ <b>Analiz Başarısız</b>\n\nHiçbir sembol için analiz yapılamadı."

        scores = result.get("scores", [])
        headers = [s.upper() for s in scores]

        # Başlık
        if result.get("volume_based"):
            title = f"📈 <b>{result.get('command_name')}</b> - Top {result.get('symbol_count', len(symbol_scores))} Volume Coins"
        else:
            title = f"📊 <b>{result.get('command_name')}</b> - {result.get('success_count', len(symbol_scores))} Coins"

        # Header
        header_cells = ["Sembol"] + headers
        header_line = "  ".join([f"{cell:10}" for cell in header_cells])
        lines = [
            title,
            "─" * (5 + len(headers) * 6),
            f"<b>{header_line}</b>",
            "─" * (5 + len(headers) * 6)
        ]

        # Sembolleri sırala
        sorted_symbols = list(symbol_scores.keys()) if result.get("volume_based") else sorted(symbol_scores.keys())

        for symbol in sorted_symbols:
            scores_dict = symbol_scores.get(symbol, {})
            display_symbol = symbol.replace('USDT', '')

            score_cells = [f"{display_symbol:8}"]
            for header in headers:
                value = scores_dict.get(header, float('nan'))

                # Ikon
                icon = get_icon(header, value)
                if isinstance(value, float) and math.isnan(value):
                    score_cells.append(f"{icon:2} ---")
                else:
                    formatted = f"{value:+.3f}"
                    score_cells.append(f"{icon:2} {formatted:7}")

            lines.append("  ".join(score_cells))

        # Özet
        failed_count = len(result.get("failed_symbols", []))
        success_count = result.get("success_count", len(symbol_scores))
        total_count = result.get("symbol_count", len(symbol_scores))

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

        # Yardım metni
        help_text = get_help_text(result.get("command"))
        if help_text:
            lines.append("")
            lines.append(f"<i>{help_text}</i>")

        return "\n".join(lines)

    # -----------------------------
    # Eğer tip bilinmiyorsa
    # -----------------------------
    return "❌ <b>Analiz tipi bilinmiyor</b>"


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