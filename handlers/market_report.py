# handler/market_report.py
"""
import
from handler.market_report import format_table_response


"""

from typing import Optional
import math

# ✅ raporlama bölümü > MERKEZİ YAPILACAK
# ana fonksiyon, heryerden çağrılı
def format_table_response(result: dict) -> str:
    """Sonuçları formatla: TABLE, INDEX_REPORT ve OI_REPORT tiplerini destekler"""

    # Hata varsa
    if "error" in result:
        return f"❌ <b>Hata:</b> {result['error']}"

    # -----------------------------
    # INDEX_REPORT (Ör. /ap)
    # -----------------------------
    """if result.get("type") == "INDEX_REPORT":
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
            f"{get_trend_icon(d.get('alt_vs_btc_short'))} <b>Alt vs BTC (Kısa):</b> <code>{d.get('alt_vs_btc_short'):.2f}</code>\n"
            f"{get_trend_icon(d.get('alt_short_term'))} <b>Alt Gücü (Kısa):</b> <code>{d.get('alt_short_term'):.2f}</code>\n"
            f"{get_trend_icon(d.get('coin_long_term'))} <b>Yapısal Güç (OI):</b> <code>{d.get('coin_long_term'):.2f}</code>\n"

            f"───────────────────\n"
            f"<i>Filtre: {len(d.get('INDEX_BASKET', []))} coinlik sepet analizi.</i>"
        )
        """

    # market_report.py - format_table_response fonksiyonuna ekle
    if result.get("type") == "INDEX_REPORT":
        d = result.get("data", {})
        
        # 1. Ana Alt Power skorları
        lines = [
            f"📊 <b>ALT MARKET POWER</b>",
            f"───────────────────"
        ]
        
        # Skor satırları
        for key in ['alt_vs_btc_short', 'alt_short_term', 'coin_long_term']:
            val = d.get(key)
            icon = "🟢" if val and val > 60 else "🔴" if val and val < 40 else "🟡"
            label = {
                'alt_vs_btc_short': 'Alt vs BTC (Kısa)',
                'alt_short_term': 'Alt Gücü (Kısa)',
                'coin_long_term': 'Yapısal Güç (OI)'
            }[key]
            lines.append(f"{icon} <b>{label}:</b> <code>{val:.2f}</code>")
        
        lines.append(f"───────────────────")
        
        # 2. ETF Akışları
        etf_data = d.get("etf_summary", {})
        if etf_data:
            lines.append(f"📈 <b>ETF AKIŞLARI</b>")
            for asset, info in etf_data.items():
                flow = info.get("flow", 0)
                icon = "🟢" if flow > 0 else "🔴"
                lines.append(f"{icon} {asset}: <code>{flow:+.1f}M$</code> ({info.get('date', 'N/A')})")
            lines.append(f"───────────────────")
        
        # 3. Top Kategoriler
        top_cats = d.get("top_categories", [])
        if top_cats:
            lines.append(f"🏷️ <b>ÖNE ÇIKAN KATEGORİLER</b>")
            for i, cat in enumerate(top_cats, 1):
                change = cat.get("change", 0)
                icon = "📈" if change > 0 else "📉"
                lines.append(f"{i}. <b>{cat['name']}</b> {icon} <code>{change:+.1f}%</code>")
            lines.append(f"───────────────────")
        
        # 4. Market Context
        mkt = d.get("market_context", {})
        if mkt:
            btc_dom = mkt.get("btc_dominance")
            if not math.isnan(btc_dom):
                lines.append(f"🌐 <b>BTC Dominance:</b> <code>{btc_dom:.1f}%</code>")
        
        # 5. Makro regime
        regime = d.get("macro_regime", "Unknown")
        lines.append(f"🎯 <b>Makro Regime:</b> {regime}")
        
        return "\n".join(lines)
        





    # -----------------------------
    # OI_REPORT (Ör. /toi)
    # -----------------------------
    if result.get("type") == "OI_REPORT":
        signals = result.get("signals", [])
        min_oi = result.get("min_oi_change", 3.0)
        
        if not signals:
            return (
                f"📊 <b>OPEN INTEREST TARAMA</b>\n"
                f"────────────────────\n"
                f"<i>Minimum %{min_oi:.1f} OI değişimi ile sinyal bulunamadı.</i>"
            )

        # LİDERLİK DURUMU
        up_oi = len([s for s in signals if s['p_change'] > 0])
        down_oi = len([s for s in signals if s['p_change'] < 0])
        market_sentiment = "🟢 ALICI (Long)" if up_oi > down_oi else "🔴 SATICI (Short)"

        lines = [
            f"📊 <b>MOMENTUM RAPORU</b>",
            f"Piyasa: {market_sentiment} | {up_oi}📈 {down_oi}📉",
            "────────────────────"
        ]

        # Sinyalleri OI değişimine göre sırala
        sorted_signals = sorted(signals, key=lambda x: x.get('oi_change', 0), reverse=True)

        for s in sorted_signals[:12]:
            symbol_raw = s['symbol'].replace('USDT', '')
            tv_link = f"https://www.tradingview.com/chart/?symbol=BINANCE:{symbol_raw}USDT.P"
            symbol_text = f"<a href='{tv_link}'><b>{symbol_raw:6}</b></a>"
            
            oi_ch = s['oi_change']
            p_ch = s['p_change']
            fr = s['fr'] if s['fr'] is not None else 0

            # 1. AKILLI DURUM ETİKETİ
            if p_ch > 0.5 and oi_ch > 5:
                status = "🟢 GÜÇLÜ"
            elif p_ch < -0.5 and oi_ch > 5:
                status = "🔴 BASKI"
            elif abs(p_ch) < 0.5:
                status = "⚡ TOPLAMA"
            else:
                status = "🔸 TAKİP"

            # 2. LONG UYGUNLUK ANALİZİ (FR Mantığı)
            # FR düşük veya negatifse Long için maliyet avantajı vardır.
            if fr > 0.05:
                fr_advice = "❌ <b>PAHALI LONG</b>" # Fonlama maliyeti yüksek
            elif fr < -0.02:
                fr_advice = "💎 <b>SQUEEZE POTANSİYELİ</b>" # Shortlar çok birikmiş, yukarı patlayabilir
            elif fr <= 0.01:
                fr_advice = "✅ <b>LONG UYGUN</b>" # İdeal düşük maliyet
            else:
                fr_advice = "⚖️ <b>NÖTR</b>"

            # SATIR OLUŞTURMA
            line = (
                f"{symbol_text} | OI: %<code>{oi_ch:+.1f}</code> | F: %<code>{p_ch:+.1f}</code>\n"
                f"┗ {status} | FR: {fr_advice} (<code>{fr:.3f}</code>)"
            )
            lines.append(line)
            lines.append("") # Okunabilirlik için boşluk

        lines.append("────────────────────")
        lines.append(f"<b>Toplam:</b> {len(signals)} sinyal | <i>/toi {min_oi}</i>")
        
        return "\n".join(lines)
        


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


# --- yardımcı fonksiyonlar --- kimse çağırmaz, bilmez
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


"""
Sınıfına ETF Veri Çekme Metodu
Momentum raporu hazırlandığı sırada veritabanındaki en son 
ETF durumunu getirmek için bu metodu MarketAnalyzer içine ekle:
"""
    

async def get_latest_etf_summary(self):
        """En son kaydedilen ETF verilerini asset bazlı özetler."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # Her asset için en son ts'ye sahip kaydı getir
            query = """
            SELECT asset, date_str, total_flow 
            FROM etf_flows 
            WHERE ts = (SELECT MAX(ts) FROM etf_flows)
            """
            cursor = await db.execute(query)
            rows = await cursor.fetchall()
            
            summary = []
            for r in rows:
                emoji = "🟢" if r['total_flow'] > 0 else "🔴"
                summary.append(f"{r['asset']}: {emoji} {r['total_flow']}M$ ({r['date_str']})")
            
            return " | ".join(summary) if summary else "ETF Verisi Henüz Yok"
    
"""
check_and_notify Fonksiyonunun Güncellenmesi
Bu fonksiyonu, ETF özetini alacak ve 
bildirim mesajının altına ekleyecek şekilde güncelle:

"""
async def check_and_notify(notifier, analyzer):
    """ETF dipnotu eklenmiş güncel bildirim sistemi."""
    threshold = 8.0
    all_signals = await analyzer.get_momentum_signals(min_oi_change=threshold)
    
    if not all_signals:
        return

    valid_signals = []
    now = time.time()
    
    for s in all_signals:
        symbol = s['symbol'].replace('USDT', '')
        last_time = notifier.last_sent.get(symbol, 0)
        
        if now - last_time >= notifier.cooldown:
            valid_signals.append(s)
            notifier.last_sent[symbol] = now

    if valid_signals:
        # ETF Özetini Al (YENİ)
        etf_summary = await analyzer.get_latest_etf_summary()
        
        result = {
            "type": "OI_REPORT",
            "signals": valid_signals,
            "min_oi_change": threshold,
            "is_auto_alert": True 
        }
        
        formatted_msg = format_table_response(result)
        
        # Mesajı birleştir ve ETF özetini dipnot olarak ekle
        final_msg = (
            f"🔔 <b>MOMENTUM ALARMI</b>\n"
            f"{formatted_msg}\n"
            f"📊 <b>Son ETF Akışları:</b>\n"
            f"<code>{etf_summary}</code>"
        )
        
        await notifier.send_notification(final_msg)
        logger.info(f"📥📢 Alarm ve ETF özeti gönderildi.")