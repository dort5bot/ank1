# analysis/market_collector.py
"""
python analysis/market_collector.py

bağımsız çlıştır
python -m analysis.market_collector

*veri toplama peryodu
10 dakika, anlık momentumu yakalamak ile API limitlerini zorlamamak arasındaki "tatlı nokta"dır

*BTC Kıyaslaması ve Veri Kapsamı
Kodun içerisinde symbols = ALT_BASKET + ["BTCUSDT"]

*veri miktarı db için çerez sayılır
Makul süre: 3 gün (72 saat) analiz için yeterlidir 
ancak trendi görmek için 7 günlük veri sağlıklısıdır. 
Yaklaşık 20.000 satır yapar ki bu DB performansını hiç etkilemez

ts         | symbol     | source    | category | price    | open_interest | funding_rate
"""
import os
import time
import asyncio
import aiohttp
import aiosqlite
from dotenv import load_dotenv

# Merkezi listeyi a_core'dan çekiyoruz
from analysis.a_core import FULL_COLLECT_LIST 

load_dotenv()

DB_PATH = "data/market_snapshot.db"
COINALYZE_API_KEY = os.getenv("COINALYZE_API_KEY")

# --- AYARLAR ---
COLLECT_INTERVAL = 600  
DATA_RETENTION_DAYS = 7 

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        # Tablo oluşturma (category sütunu dahil)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS snapshot (
                ts INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                source TEXT NOT NULL,
                category TEXT DEFAULT 'temp',
                price REAL,
                open_interest REAL,
                funding_rate REAL,
                PRIMARY KEY (ts, symbol, source)
            );
        """)
        await db.commit()

async def cleanup_db():
    """Hibrit temizlik: temp veriler 1 saat, basket veriler 7 gün saklanır"""
    now = int(time.time())
    async with aiosqlite.connect(DB_PATH) as db:
        # 1. Kural: Geçici sorgular (temp) 1 saatlik
        await db.execute("DELETE FROM snapshot WHERE category = 'temp' AND ts < ?", (now - 3600,))
        # 2. Kural: Takip listesi (basket) 7 günlük
        await db.execute("DELETE FROM snapshot WHERE category = 'basket' AND ts < ?", (now - (DATA_RETENTION_DAYS * 86400),))
        await db.commit()

async def fetch_all_data():
    symbols = FULL_COLLECT_LIST
    ts = int(time.time())
    category = "basket" # Collector her zaman 'basket' (kalıcı) olarak toplar
    final_rows = []
    
    async with aiohttp.ClientSession() as session:
        # 1. BINANCE FİYATLAR
        for s in symbols:
            try:
                async with session.get(f"https://api.binance.com/api/v3/ticker/price?symbol={s}", timeout=5) as r:
                    if r.status == 200:
                        data = await r.json()
                        final_rows.append({
                            "ts": ts, "symbol": s, "source": "binance", 
                            "price": float(data['price']),
                            "category": category  
                        })
            except Exception as e:
                print(f"⚠️ Binance Hatası {s}: {e}")

        # 2. COINALYZE (Toplu Çekim)
        c_syms = ",".join([f"{s}_PERP.A" for s in symbols])
        headers = {"api_key": COINALYZE_API_KEY}
        
        try:
            async with session.get(f"https://api.coinalyze.net/v1/open-interest?symbols={c_syms}", headers=headers) as r1, \
                       session.get(f"https://api.coinalyze.net/v1/funding-rate?symbols={c_syms}", headers=headers) as r2:
                
                oi_data = await r1.json() if r1.status == 200 else []
                fr_data = await r2.json() if r2.status == 200 else []

                for s in symbols:
                    oi_val = next((x['value'] for x in oi_data if x['symbol'].startswith(s)), None)
                    fr_val = next((x['value'] for x in fr_data if x['symbol'].startswith(s)), None)
                    
                    if oi_val is not None or fr_val is not None:
                        final_rows.append({
                            "ts": ts, "symbol": s, "source": "coinalyze",
                            "open_interest": oi_val, "funding_rate": fr_val, "category": category
                        })
        except Exception as e:
            print(f"⚠️ Coinalyze Hatası: {e}")

    return final_rows


# SADECE 1 TUR veri toplamalı
# DB init ,fetch_all_data,DB’ye yaz,cleanup 
# ❌ sleep, while True

async def collect_once():
    await init_db()
    rows = await fetch_all_data()

    sql = """INSERT OR REPLACE INTO snapshot 
             (ts, symbol, source, category, price, open_interest, funding_rate) 
             VALUES (?,?,?,?,?,?,?)"""

    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(sql, [
            (r["ts"], r["symbol"], r["source"], r["category"],
             r.get("price"), r.get("open_interest"), r.get("funding_rate"))
            for r in rows
        ])
        await db.commit()

    await cleanup_db()
    return len(rows)



if __name__ == "__main__":
    async def runner():
        while True:
            n = await collect_once()
            print(f"✅ {n} satır yazıldı")
            await asyncio.sleep(COLLECT_INTERVAL)

    asyncio.run(runner())


# --- bağımsız çalışma sonsuz döngüsü var, mainle çakışıyor---
# async def main_loop():
#     await init_db()
#     print(f"🚀 Collector başlatıldı. Periyot: {COLLECT_INTERVAL/60} dk. Saklama: {DATA_RETENTION_DAYS} gün.")
#     
#     while True:
#         start_time = time.time()
#         try:
#             print(f"\n{time.strftime('%H:%M:%S')} - Veri toplanıyor...")
#             rows = await fetch_all_data()
#             
#             # DB Yazma (CATEGORY SÜTUNU EKLENDİ - KRİTİK DÜZELTME)
#             sql = """INSERT OR REPLACE INTO snapshot 
#                      (ts, symbol, source, category, price, open_interest, funding_rate) 
#                      VALUES (?,?,?,?,?,?,?)"""
#             
#             async with aiosqlite.connect(DB_PATH) as db:
#                 await db.executemany(sql, [
#                     (r["ts"], r["symbol"], r["source"], r["category"], 
#                      r.get("price"), r.get("open_interest"), r.get("funding_rate")) 
#                     for r in rows
#                 ])
#                 await db.commit()
#             
#             print(f"✅ {len(rows)} veri 'basket' olarak kaydedildi.")
#             await cleanup_db()
#             
#         except Exception as e:
#             print(f"❌ Döngü Hatası: {e}")
# 
#         elapsed = time.time() - start_time
#         sleep_time = max(0, COLLECT_INTERVAL - elapsed)
#         await asyncio.sleep(sleep_time)
# 
# if __name__ == "__main__":
#     try:
#         asyncio.run(main_loop())
#     except KeyboardInterrupt:
#         print("\n🛑 Collector durduruldu.")