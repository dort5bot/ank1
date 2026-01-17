# market_collector.py - YENİ
"""
bağımsız çlıştır
python -m analysis.market_collector

"""

import os
import time
import logging
import asyncio
import aiohttp
import aiosqlite
from datetime import datetime
from dotenv import load_dotenv

# ETF için bağımlılıklar
import re
from typing import Dict, Any, List, Optional
from bs4 import BeautifulSoup
from curl_cffi.requests import AsyncSession


from analysis.a_core import INDEX_BASKET, WATCHLIST
from handlers.market_report import format_table_response
from utils.notifier import TelegramNotifier

load_dotenv()

logger = logging.getLogger(__name__)

# --- AYARLAR ---
# --- YAPILANDIRMA ---
DB_PATH = "data/market.db"
COINALYZE_API_KEY = os.getenv("COINALYZE_API_KEY")
COLLECT_INTERVAL = 600  # 10 Dakika (Binance & Coinalyze)
CG_CYCLE_LIMIT = 36     # 6 Saatte bir Coingecko (36 * 10dk)
CG_CATEGORY_LIMIT = 20  # kategori için ilk 20 grup
DATA_RETENTION_DAYS = 7 # veri silinme süresi, (gün)
ALLOWED_EXCHANGES = {"Binance", "OKX", "Bybit", "Coinbase Exchange"}


class BTCDataUnavailable(Exception):
    """Kritik: BTC verisi olmadan analiz yapılamaz."""
    pass

class DataManager:
    """Tüm tabloların yönetimini sağlar."""
    def __init__(self, db_path):
        self.db_path = db_path

    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            # Binance & Coinalyze Tabloları
            await db.execute("CREATE TABLE IF NOT EXISTS bi_market (ts INTEGER, symbol TEXT, price REAL, volume REAL, category TEXT, PRIMARY KEY (ts, symbol))")
            await db.execute("CREATE TABLE IF NOT EXISTS cz_derivatives (ts INTEGER, symbol TEXT, open_interest REAL, funding_rate REAL, category TEXT, PRIMARY KEY (ts, symbol))")
            
            # Coingecko Tabloları
            await db.execute("CREATE TABLE IF NOT EXISTS cg_global (ts INTEGER PRIMARY KEY, total_mcap REAL, total_vol REAL, btc_dom REAL, eth_dom REAL)")
            await db.execute("CREATE TABLE IF NOT EXISTS cg_categories (ts INTEGER, category_id TEXT, market_cap REAL, volume REAL, change_24h REAL, PRIMARY KEY (ts, category_id))")
            await db.execute("CREATE TABLE IF NOT EXISTS cg_exchange_bias (ts INTEGER, coin TEXT, exchange TEXT, price REAL, PRIMARY KEY (ts, coin, exchange))")
            
           # ETF Tablosu
            await db.execute("CREATE TABLE IF NOT EXISTS etf_flows (ts INTEGER, asset TEXT, date_str TEXT, total_flow REAL, PRIMARY KEY (ts, asset))")
                
            await db.commit()

    async def save_market_data(self, bi_rows, cz_rows):
        async with aiosqlite.connect(self.db_path) as db:
            if bi_rows: await db.executemany("INSERT OR REPLACE INTO bi_market VALUES (?,?,?,?,?)", bi_rows)
            if cz_rows: await db.executemany("INSERT OR REPLACE INTO cz_derivatives VALUES (?,?,?,?,?)", cz_rows)
            await db.commit()

    async def save_cg_snapshot(self, global_data, category_rows, bias_rows):
        async with aiosqlite.connect(self.db_path) as db:
            if global_data: await db.execute("INSERT OR REPLACE INTO cg_global VALUES (?,?,?,?,?)", global_data)
            if category_rows: await db.executemany("INSERT OR REPLACE INTO cg_categories VALUES (?,?,?,?,?)", category_rows)
            if bias_rows: await db.executemany("INSERT OR REPLACE INTO cg_exchange_bias VALUES (?,?,?,?)", bias_rows)
            await db.commit()

    # ETF kaydetme metodu
    async def save_etf_data(self, etf_rows):
        async with aiosqlite.connect(self.db_path) as db:
            if etf_rows: 
                await db.executemany("INSERT OR REPLACE INTO etf_flows VALUES (?,?,?,?)", etf_rows)
            await db.commit() # Bir tık dışarı (if hizasına) alındı



    async def cleanup(self): # 7 günden eski veriler temizlenir
        limit_ts = int(time.time()) - (DATA_RETENTION_DAYS * 86400)
        async with aiosqlite.connect(self.db_path) as db:
            for table in ["bi_market", "cz_derivatives", "cg_global", "cg_categories", "cg_exchange_bias", "etf_flows"]:
                await db.execute(f"DELETE FROM {table} WHERE ts < ?", (limit_ts,))
            await db.commit()

class MarketCollector:
    """Binance ve Coinalyze verilerini asenkron toplar."""
    def __init__(self, session):
        self.session = session
        self.symbols = list(set(INDEX_BASKET + WATCHLIST + ["BTCUSDT"]))

    async def fetch_binance(self, ts):
        try:
            async with self.session.get("https://api.binance.com/api/v3/ticker/24hr") as r:
                if r.status == 200:
                    tickers = {t['symbol']: t for t in await r.json() if t['symbol'] in self.symbols}
                    return [(ts, s, float(tickers[s]['lastPrice']), float(tickers[s]['quoteVolume']), 'basket') 
                            for s in self.symbols if s in tickers]
        except Exception as e:
            logger.error(f"❌⚠️ Binance Hatası: {e}")
        return []

    async def fetch_coinalyze(self, ts):
        headers = {"api_key": COINALYZE_API_KEY}
        # 1. Fail-fast: Önce BTC'yi kontrol et (Hata alırsa Exception fırlatır)
        btc_res = await self._get_cz_chunk(["BTCUSDT"], headers, ts)
        if not btc_res:
            raise BTCDataUnavailable("BTC verisi alınamadı!")

        results = btc_res
        others = [s for s in self.symbols if s != "BTCUSDT"]
        
        # 2. Chunk bazlı seri işlem (Senin istediğin -1- nolu geliştirme)
        # Chunk size: 2, Delay: 1.2s (Rate limit dostu)
        for i in range(0, len(others), 2):
            chunk = others[i:i+2]
            # logger.info(f"⏳ Veri çekiliyor: {chunk}")
            res = await self._get_cz_chunk(chunk, headers, ts)
            results.extend(res)
            await asyncio.sleep(1.2) 
        return results

    async def _get_cz_chunk(self, chunk, headers, ts):
        """Geliştirilmiş chunk çekici: Hata durumunda 1 kez tekrar dener."""
        c_syms = ",".join(f"{s}_PERP.A" for s in chunk)
        url_oi = f"https://api.coinalyze.net/v1/open-interest?symbols={c_syms}"
        url_fr = f"https://api.coinalyze.net/v1/funding-rate?symbols={c_syms}"
        
        for attempt in range(2): # Basit retry mekanizması
            try:
                async with self.session.get(url_oi, headers=headers) as r1, \
                           self.session.get(url_fr, headers=headers) as r2:
                    
                    if r1.status == 200 and r2.status == 200:
                        oi_raw, fr_raw = await r1.json(), await r2.json()
                        oi_map = {x['symbol']: x['value'] for x in oi_raw}
                        fr_map = {x['symbol']: x['value'] for x in fr_raw}
                        return [(ts, s, oi_map.get(f"{s}_PERP.A"), fr_map.get(f"{s}_PERP.A"), 'basket') for s in chunk]
                    
                    if r1.status == 429 or r2.status == 429:
                        await asyncio.sleep(2 * (attempt + 1)) # Rate limit varsa bekle
            except Exception as e:
                logger.warning(f"⏳⚠️ Coinalyze bağlantı hatası ({chunk}): {e}")
                await asyncio.sleep(1)
        return []
        
class CoingeckoCollector:
    """Coingecko'dan temiz ve filtrelenmiş verileri toplar."""
    def __init__(self, session):
        self.session = session
        self.base_url = "https://api.coingecko.com/api/v3"

    async def fetch_all(self, category_limit=20): # Parametre eklendi
        ts = int(time.time())
        g_data = await self._fetch_global(ts)
        c_rows = await self._fetch_categories(ts, limit=category_limit) # Limiti ilet
        # Sadece BTC'yi değil, önemli gördüğün diğer coinlerin bias'ını da buraya ekleyebilirsin
        b_rows = await self._fetch_bias("bitcoin", ts) 
        return g_data, c_rows, b_rows


    async def _fetch_global(self, ts):
        try:
            async with self.session.get(f"{self.base_url}/global") as r:
                if r.status == 200:
                    d = (await r.json())["data"]
                    return (ts, d["total_market_cap"]["usd"], d["total_volume"]["usd"], d["market_cap_percentage"]["btc"], d["market_cap_percentage"]["eth"])
        except Exception as e:
            logger.warning(f"⏳⚠️ CG Global Hatası: {e}")
        return None


    async def _fetch_categories(self, ts, limit=20): # Varsayılan N=20
        try:
            async with self.session.get(f"{self.base_url}/coins/categories") as r:
                if r.status == 200:
                    data = await r.json()
                    
                    # Piyasa değerine göre ilk N kategoriyi al (Dinamik Ayar)
                    top_data = data[:limit] 
                    logger.info(f"📊 Coingecko'dan en büyük {len(top_data)} kategori işleniyor.")
                    
                    results = []
                    for c in top_data:
                        cat_id = c.get("id")
                        if cat_id:
                            results.append((
                                ts, 
                                cat_id, 
                                c.get("market_cap") or 0.0,
                                c.get("volume_24h") or 0.0, 
                                c.get("market_cap_change_24h") or 0.0
                            ))
                    return results


                elif r.status == 429:
                    logger.warning("⚠️ Coingecko Rate Limit (429) hatası. İstek reddedildi.")
                else:
                    logger.warning(f"⚠️ Coingecko Hatası: Status {r.status}")
                    
        except Exception as e:
            logger.warning(f"⏳⚠️ CG Kategori Hatası: {e}")
        return []
        

    async def _fetch_bias(self, coin_id, ts):
        """Hatalı pariteleri (BTC/SATS vb.) eler, sadece gerçek USD/USDT fiyatlarını alır."""
        try:
            async with self.session.get(f"{self.base_url}/coins/{coin_id}/tickers") as r:
                if r.status == 200:
                    data = await r.json()
                    tickers = data.get("tickers", [])
                    
                    bias_rows = []
                    for t in tickers:
                        exchange_name = t["market"]["name"]
                        # FİLTRE: Sadece izin verilen borsalar ve hedef birimi USD veya USDT olanlar
                        if exchange_name in ALLOWED_EXCHANGES:
                            # target: fiyatın hangi birimde olduğunu belirtir (örn: USD)
                            if t.get("target") in ["USD", "USDT"]:
                                price = t.get("last")
                                if price:
                                    bias_rows.append((ts, coin_id, exchange_name, float(price)))
                    return bias_rows
        except Exception as e:
            logger.warning(f"⏳⚠️ CG Bias Hatası: {e}")
        return []

class ETFDataService:
    """
    Farside üzerinden ETF verilerini çeker ve market_collector veritabanı 
    formatına uygun (timestamp içerikli) hale getirir.
    """
    def __init__(self):
        self.base_urls = {
            "BTC": "https://farside.co.uk/btc/",
            "ETH": "https://farside.co.uk/eth/",
            "SOL": "https://farside.co.uk/sol/"
        }

    def _is_valid_date(self, text: str) -> bool:
        """Metnin tarih formatında olup olmadığını kontrol eder."""
        pattern = r'\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}'
        return bool(re.search(pattern, text))

    def _clean_numeric(self, value: str) -> float:
        """Parantezli ve virgüllü finansal metinleri float'a çevirir."""
        if not value or value in ["-", "0.0", ""]:
            return 0.0
        cleaned = value.replace("(", "-").replace(")", "").replace(",", "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return 0.0

    async def fetch_all_etf_data(self) -> List[tuple]:
        """
        Tüm varlıklar için ETF verilerini çeker ve DB'ye uygun tuple listesi döner.
        Format: (ts, asset, date_str, total_flow)
        """
        results = []
        ts = int(time.time())
        
        async with AsyncSession(impersonate="chrome110") as s:
            for asset, url in self.base_urls.items():
                try:
                    response = await s.get(url, timeout=30)
                    if response.status_code != 200:
                        logger.warning(f"⚠️ ETF Hatası ({asset}): HTTP {response.status_code}")
                        continue
                    
                    soup = BeautifulSoup(response.text, "html.parser")
                    table = soup.find("table", class_="etf")
                    if not table:
                        continue
                        
                    rows = table.find("tbody").find_all("tr")
                    
                    # En güncel (en alttaki) geçerli veriyi bul
                    for row in reversed(rows):
                        cells = row.find_all("td")
                        if not cells: continue
                        
                        date_str = cells[0].get_text(strip=True)
                        total_val = cells[-1].get_text(strip=True)
                        
                        if self._is_valid_date(date_str) and total_val not in ["-", "0.0", "0"]:
                            flow = self._clean_numeric(total_val)
                            results.append((ts, asset, date_str, flow))
                            logger.info(f"✅ ETF Verisi Alındı: {asset} | {date_str} | {flow} $m")
                            break
                            
                except Exception as e:
                    logger.error(f"❌ ETF Servis Hatası ({asset}): {e}")
        
        return results


# --- BILDIRIM BÖLÜMÜ ---
# Momentum sinyallerini yeni tablo yapısıyla (JOIN) en hızlı şekilde çeker
class MarketAnalyzer:
    def __init__(self, db_path):
        self.db_path = db_path

    async def get_momentum_signals(self, min_oi_change=3.0):
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # En son 2 başarılı veri toplama zamanını al
            cursor = await db.execute("SELECT DISTINCT ts FROM cz_derivatives ORDER BY ts DESC LIMIT 2")
            times = await cursor.fetchall()
            if len(times) < 2: return []
            
            latest_ts, prev_ts = times[0]['ts'], times[1]['ts']

            # GELİŞTİRİLMİŞ SORGU: Hem fiyat hem OI değişimini tek seferde hesaplar
            query = """
            SELECT 
                curr.symbol,
                bi.price,
                ((bi.price / bi_prev.price) - 1) * 100 as p_change,
                curr.open_interest as oi,
                ((curr.open_interest / prev.open_interest) - 1) * 100 as oi_change,
                curr.funding_rate as fr
            FROM cz_derivatives curr
            JOIN cz_derivatives prev ON curr.symbol = prev.symbol AND prev.ts = ?
            JOIN bi_market bi ON curr.symbol = bi.symbol AND bi.ts = curr.ts
            JOIN bi_market bi_prev ON curr.symbol = bi_prev.symbol AND bi_prev.ts = ?
            WHERE curr.ts = ? 
              AND curr.open_interest > 0 
              AND prev.open_interest > 0
              AND bi_prev.price > 0
              AND oi_change >= ?
            ORDER BY oi_change DESC
            """
            cursor = await db.execute(query, (prev_ts, prev_ts, latest_ts, min_oi_change))
            return await cursor.fetchall()

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
            
            
async def check_and_notify(notifier, analyzer):
    """Spam engelleme (cooldown) mekanizmalı bildirim sistemi."""
    # %8 ve üzeri OI artışlarını alarm olarak kabul et
    threshold = 8.0
    all_signals = await analyzer.get_momentum_signals(min_oi_change=threshold)
    
    if not all_signals:
        return

    valid_signals = []
    now = time.time()
    
    for s in all_signals:
        symbol = s['symbol'].replace('USDT', '')
        # Cooldown kontrolü (Eski kodundaki notifier.last_sent ve cooldown'u kullanır)
        last_time = notifier.last_sent.get(symbol, 0)
        
        if now - last_time >= notifier.cooldown:
            valid_signals.append(s)
            notifier.last_sent[symbol] = now

    if valid_signals:
        result = {
            "type": "OI_REPORT",
            "signals": valid_signals,
            "min_oi_change": threshold,
            "is_auto_alert": True 
        }
        
        # market_report.py'deki profesyonel tabloyu kullanır
        formatted_msg = format_table_response(result)
        final_msg = f"🔔 <b>MOMENTUM ALARMI</b>\n{formatted_msg}"
        
        await notifier.send_notification(final_msg)
        logger.info(f"📥📢 {len(valid_signals)} coin için alarm gönderildi.")


# ---  python -m analysis.market_collector
async def main_loop():
    db_manager = DataManager(DB_PATH)
    await db_manager.init_db()
    notifier = TelegramNotifier()
    analyzer = MarketAnalyzer(DB_PATH)
    etf_service = ETFDataService() # Servisi başlat
    cycle = 0

    while True:
        async with aiohttp.ClientSession() as session:
            try:
                ts = int(time.time())
                mc = MarketCollector(session)
                cc = CoingeckoCollector(session)

                # 1. Market Verileri (Binance & Coinalyze)
                bi_data = await mc.fetch_binance(ts)
                cz_data = await mc.fetch_coinalyze(ts)
                await db_manager.save_market_data(bi_data, cz_data)

                # 2. Analiz ve Bildirim
                if len(cz_data) > 0:
                    await check_and_notify(notifier, analyzer)

                # 3. Coingecko (Periyodik)
                if cycle % CG_CYCLE_LIMIT == 0:
                    # CG_CATEGORY_LIMIT (20) değerini fetch_all içine gönderiyoruz
                    g, c, b = await cc.fetch_all(category_limit=CG_CATEGORY_LIMIT) 
                    await db_manager.save_cg_snapshot(g, c, b)
                    logger.info(f"✅ Coingecko Snapshot kaydedildi (Top {CG_CATEGORY_LIMIT} Kategori)")
                    
                                                          
                # 4. ETF VERİLERİ (Her 3 Saatte Bir - 18 * 10dk)
                if cycle % 18 == 0: 
                    logger.info("📊 ETF verileri güncelleniyor...")
                    etf_rows = await etf_service.fetch_all_etf_data()
                    if etf_rows:
                        await db_manager.save_etf_data(etf_rows)

                logger.info(f"📥 {datetime.now().strftime('%H:%M')} - Cycle {cycle} tamam.")
                await db_manager.cleanup()
                cycle += 1
                await asyncio.sleep(COLLECT_INTERVAL)

            except BTCDataUnavailable as e:
                logger.info(f"⏳ BTC Bekleniyor: {e}")
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"❌❌ Hata: {e}")
                await asyncio.sleep(10)
                        
if __name__ == "__main__":
    asyncio.run(main_loop())


    