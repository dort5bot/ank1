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

 snapshot içeriği:
ts         | symbol     | source    | category | price    | open_interest | funding_rate

Open Interest (OI) değerinin aniden fırlaması, o coin'e büyük miktarda para girdiğini ve bir volatilite patlamasının yaklaştığını gösterir.
OI Analizi Ne İşe Yarar?
Normalde fiyat ve OI beraber hareket eder. Ancak şu iki durum senin için "altın" değerindedir:

Fiyat Yatay + OI Sert Yukarı: 
Balinalar sessizce pozisyon topluyor. Yakında sert bir kırılım (genelde yukarı) gelebilir.

Fiyat Aşağı + OI Sert Yukarı: 
İnsanlar düşüşe inatla "short" açıyor veya düşüşü satın alıyor. Bu durum genellikle bir "Short Squeeze" (fiyatın aniden yukarı patlaması) ile sonuçlanır.

"""
# analysis/market_collector.py
import os
import time
import asyncio
import aiohttp
import aiosqlite
from dotenv import load_dotenv

# from analysis.a_core import FULL_COLLECT_LIST 
from analysis.a_core import INDEX_BASKET, WATCHLIST

from handlers.market_report import format_table_response

from utils.notifier import TelegramNotifier

load_dotenv()

DB_PATH = "data/market_snapshot.db"
COINALYZE_API_KEY = os.getenv("COINALYZE_API_KEY")

# --- AYARLAR ---
COLLECT_INTERVAL = 600  
DATA_RETENTION_DAYS = 7 

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS snapshot (
                ts INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                source TEXT NOT NULL,
                category TEXT DEFAULT 'temp',
                price REAL,
                open_interest REAL,
                funding_rate REAL,
                volume REAL,
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

# → session parametresi alır
async def fetch_coinalyze_data(session, symbols_ignored_here):
    """
    Öncelik sıralı veri çekme: 
    1. BTC (Kritik)
    2. INDEX_BASKET (Analiz için gerekli)
    3. WATCHLIST (Kişisel takip)
    """
    results = []
    ts = int(time.time())
    headers = {"api_key": COINALYZE_API_KEY}

    # --- 1. ADIM: BTC (Vazgeçilmez) ---
    # BTC her zaman tek başına ve ilk sırada çekilir
    btc_res = await fetch_with_strict_limit(session, ["BTCUSDT"], headers, ts)
    if not btc_res:
        print("❌ KRİTİK: BTC verisi alınamadı! Analiz tutarlılığı için işlem durduruluyor.")
        return [] # Bu periyodu tamamen iptal et (fail-fast)
    results.extend(btc_res)

    # --- 2. ADIM: INDEX_BASKET (Yüksek Öncelik) ---
    # BTC zaten alındığı için listeden çıkarıyoruz
    index_only = [s for s in INDEX_BASKET if s != "BTCUSDT"]
    # Chunk size 3, seri çekim (rate limit koruması)
    index_res = await fetch_in_chunks(session, index_only, headers, ts, chunk_size=3, delay=1.0)
    results.extend(index_res)

    # --- 3. ADIM: WATCHLIST (Normal Öncelik) ---
    # Önceki listelerde olmayanları ayıkla (Mükerrer isteği engeller)
    watch_only = [s for s in WATCHLIST if s not in INDEX_BASKET and s != "BTCUSDT"]
    if watch_only:
        # Daha az kritik olduğu için chunk size biraz daha büyük olabilir
        watch_res = await fetch_in_chunks(session, watch_only, headers, ts, chunk_size=5, delay=1.0)
        results.extend(watch_res)

    return results


async def fetch_in_chunks(session, symbols, headers, ts, chunk_size, delay):
    """Verilen listeyi parçalar halinde ve bekleyerek çeker (Seri İşlem)"""
    chunk_results = []
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        # Mevcut fetch_with_retry mantığı ama daha kısa bekleme süreli
        res = await fetch_with_strict_limit(session, chunk, headers, ts)
        chunk_results.extend(res)
        await asyncio.sleep(delay) # Her chunk arası güvenli bekleme
    return chunk_results
    
async def fetch_with_strict_limit(session, chunk, headers, ts):
    """Kritik veriler için sadece 1 kez kısa bekleyip tekrar dener."""
    c_syms = ",".join(f"{s}_PERP.A" for s in chunk)
    for attempt in range(2): 
        try:
            async with session.get(f"https://api.coinalyze.net/v1/open-interest?symbols={c_syms}", headers=headers) as r1, \
                       session.get(f"https://api.coinalyze.net/v1/funding-rate?symbols={c_syms}", headers=headers) as r2:
                
                if r1.status == 200 and r2.status == 200:
                    oi_raw = await r1.json()
                    fr_raw = await r2.json()
                    oi_lookup = {x['symbol']: x['value'] for x in oi_raw}
                    fr_lookup = {x['symbol']: x['value'] for x in fr_raw}

                    rows = []
                    for s in chunk:
                        c_key = f"{s}_PERP.A"
                        rows.append({
                            "ts": ts, "symbol": s, "source": "coinalyze",
                            "open_interest": oi_lookup.get(c_key),
                            "funding_rate": fr_lookup.get(c_key),
                            "category": "basket"
                        })
                    return rows
                
                if r1.status == 429 or r2.status == 429:
                    await asyncio.sleep(2) # 429 ise kısa bekle ve son kez dene
        except Exception as e:
            print(f"⚠️ Fetch Hatası {chunk}: {e}")
    return []
    


# → TEK session açar
"""async def fetch_all_data():
    # Listeleri burada birleştirin (set kullanımı mükerrer kaydı önler)
    symbols = list(set(INDEX_BASKET + WATCHLIST + ["BTCUSDT"]))
    ts = int(time.time())
    final_rows = []

    async with aiohttp.ClientSession() as session:
        # 1. BINANCE TOPLU FİYAT ÇEKME (Tek İstek!)
        try:
            # Sembol bazlı değil, genel ticker listesini çekiyoruz
            async with session.get("https://api.binance.com/api/v3/ticker/price", timeout=10) as r:
                if r.status == 200:
                    all_tickers = await r.json()
                    # Bizim listemizde olanları sözlüğe çevir (Hızlı erişim için)
                    price_dict = {t['symbol']: float(t['price']) for t in all_tickers if t['symbol'] in symbols}
                    
                    for s in symbols:
                        if s in price_dict:
                            final_rows.append({
                                "ts": ts, "symbol": s, "source": "binance",
                                "price": price_dict[s], "category": "basket"
                            })
        except Exception as e:
            print(f"⚠️ Binance Toplu Fiyat Hatası: {e}")

        # 2. COINALYZE (Aynı session, gruplandırılmış istek)
        coinalyze_rows = await fetch_coinalyze_data(session, symbols)
        final_rows.extend(coinalyze_rows)

    return final_rows
"""

# market_collector.py içindeki fetch_all_data güncellenmiş hali
# price + 24 saatlik kümülatif hacim
async def fetch_all_data():
    symbols = list(set(INDEX_BASKET + WATCHLIST + ["BTCUSDT"]))
    ts = int(time.time())
    final_rows = []

    async with aiohttp.ClientSession() as session:
        # 1. BINANCE TOPLU FİYAT VE HACİM ÇEKME
        try:
            # ticker/24hr hem fiyat (lastPrice) hem hacim (quoteVolume) verir
            async with session.get("https://api.binance.com/api/v3/ticker/24hr", timeout=10) as r:
                if r.status == 200:
                    all_tickers = await r.json()
                    # Sözlük yapısı: { "BTCUSDT": {"price": 50000, "vol": 1000000}, ... }
                    ticker_dict = {
                        t['symbol']: {
                            "price": float(t['lastPrice']), 
                            "volume": float(t['quoteVolume']) # USDT bazlı hacim
                        } 
                        for t in all_tickers if t['symbol'] in symbols
                    }
                    
                    for s in symbols:
                        if s in ticker_dict:
                            final_rows.append({
                                "ts": ts, "symbol": s, "source": "binance",
                                "price": ticker_dict[s]["price"],
                                "volume": ticker_dict[s]["volume"], # ⬅️ DB'ye gidecek
                                "category": "basket"
                            })
        except Exception as e:
            print(f"⚠️ Binance Veri Hatası: {e}")

        # 2. COINALYZE (OI ve Funding çekmeye devam eder)
        coinalyze_rows = await fetch_coinalyze_data(session, symbols)
        final_rows.extend(coinalyze_rows)

    return final_rows


# market_collector.py içindeki collect_once metodu
async def collect_once():
    await init_db()
    rows = await fetch_all_data()

    # volume eklendi
    sql = """INSERT OR REPLACE INTO snapshot 
             (ts, symbol, source, category, price, open_interest, funding_rate, volume) 
             VALUES (?,?,?,?,?,?,?,?)"""

    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(sql, [
            (r["ts"], r["symbol"], r["source"], r["category"],
             r.get("price"), r.get("open_interest"), r.get("funding_rate"), 
             r.get("volume")) # ⬅️ eklendi
            for r in rows
        ])
        await db.commit()

    await cleanup_db()
    return len(rows)


class MarketAnalyzer:
    def __init__(self, db_path):
        self.db_path = db_path

    async def get_momentum_signals(self, min_oi_change=3.0):
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # En son 2 zaman damgasını al
            cursor = await db.execute("SELECT DISTINCT ts FROM snapshot ORDER BY ts DESC LIMIT 2")
            times = await cursor.fetchall()
            
            if len(times) < 2: return []
            latest_ts, prev_ts = times[0]['ts'], times[1]['ts']

            query = """
            SELECT 
                c_oi.symbol,
                b_pr.price,  -- Binance'ten gelen saf fiyat
                ((b_pr.price / b_prev.price) - 1) * 100 as p_change, -- Saf fiyat değişimi
                c_oi.open_interest as oi,
                ((c_oi.open_interest / c_prev.open_interest) - 1) * 100 as oi_change, -- Saf OI değişimi
                c_oi.funding_rate as fr
            FROM snapshot c_oi
            -- 1. Güvenlik: Anlık fiyatı Binance'ten al
            JOIN snapshot b_pr ON c_oi.symbol = b_pr.symbol 
                AND b_pr.ts = c_oi.ts AND b_pr.source = 'binance'
            -- 2. Güvenlik: Önceki OI verisini Coinalyze'dan al (Saf kıyas)
            JOIN snapshot c_prev ON c_oi.symbol = c_prev.symbol 
                AND c_prev.ts = ? AND c_prev.source = 'coinalyze'
            -- 3. Güvenlik: Önceki fiyatı Binance'ten al
            JOIN snapshot b_prev ON c_oi.symbol = b_prev.symbol 
                AND b_prev.ts = ? AND b_prev.source = 'binance'
            
            WHERE c_oi.ts = ? 
              AND c_oi.source = 'coinalyze'
              -- SAF GERÇEKLİK FİLTRELERİ:
              AND c_oi.open_interest IS NOT NULL    -- Anlık OI yoksa hesaplama
              AND c_prev.open_interest IS NOT NULL  -- Önceki OI yoksa hesaplama
              AND b_pr.price IS NOT NULL            -- Anlık fiyat yoksa hesaplama
              AND b_prev.price IS NOT NULL          -- Önceki fiyat yoksa hesaplama
              AND c_prev.open_interest > 0          -- Sıfıra bölünme hatasını engelle
              AND oi_change >= ?                    -- Sadece eşiği geçen gerçek veriler
            ORDER BY oi_change DESC
            """
            cursor = await db.execute(query, (prev_ts, prev_ts, latest_ts, min_oi_change))
            return await cursor.fetchall()
            




async def check_and_notify(notifier, analyzer):
    """
    Hafızalı (cooldown destekli) bildirim kontrolü.
    """
    # 1. Bildirim eşiğine takılan TÜM sinyalleri çek (%8.0+)
    all_signals = await analyzer.get_momentum_signals(min_oi_change=8.0)
    
    if not all_signals:
        return

    # 2. SPAM FİLTRESİ: Sadece süresi dolan (1 saat) coinleri ayıkla
    valid_signals = []
    now = time.time()
    
    for s in all_signals:
        symbol = s['symbol'].replace('USDT', '')
        last_time = notifier.last_sent.get(symbol, 0)
        
        # Eğer cooldown süresi dolmuşsa listeye ekle
        if now - last_time >= notifier.cooldown:
            valid_signals.append(s)
            # Zaman damgasını burada güncelle (Filtreden geçtiği an)
            notifier.last_sent[symbol] = now

    # 3. Eğer filtreden geçen 'yeni' coin varsa raporu gönder
    if valid_signals:
        result = {
            "type": "OI_REPORT",
            "signals": valid_signals,
            "min_oi_change": 8.0,
            "is_auto_alert": True 
        }
        
        # Senin profesyonel formatlayıcın üzerinden mesajı oluşturuyoruz
        formatted_msg = format_table_response(result)
        final_msg = f"🔔 <b>MOMENTUM ALARMI</b>\n{formatted_msg}"
        
        # Telegram'a gönder
        await notifier.send_notification(final_msg)
        print(f"📢 Bildirim gönderildi: {', '.join([s['symbol'] for s in valid_signals])}")

if __name__ == "__main__":
    async def runner():
        print(f"🚀 Market Collector + Alarm Sistemi başlatıldı.")
        print(f"📊 Periyot: {COLLECT_INTERVAL/60} dk | Bildirim Eşiği: %8.0 OI")
        
        # ÖNEMLİ: Nesneleri döngü dışında oluşturuyoruz ki hafıza (cooldown) korunsun
        notifier = TelegramNotifier() 
        analyzer = MarketAnalyzer(DB_PATH) 
        
        while True:
            try:
                # 1. Veri Topla ve Kaydet
                n = await collect_once()
                print(f"{time.strftime('%H:%M:%S')} - ✅ {n} satır veritabanına yazıldı")

                # 2. Konsol Loglama (Daha düşük eşik: %3.0+)
                console_signals = await analyzer.get_momentum_signals(min_oi_change=3.0)
                if console_signals:
                    print(f"\n🔥 MOMENTUM SİNYALLERİ (%3+ OI) 🔥")
                    for s in console_signals:
                        p_str = f"{s['p_change']:+.2f}%"
                        oi_str = f"{s['oi_change']:+.2f}%"
                        print(f"SYMBOL: {s['symbol']:<10} | OI: {oi_str:<8} | PRICE: {p_str:<8}")
                    print("-" * 55)

                # 3. Otomatik Bildirim Kontrolü (%8.0+ ve Cooldown)
                await check_and_notify(notifier, analyzer)

            except Exception as e:
                print(f"{time.strftime('%H:%M:%S')} - ❌ Hata Oluştu: {e}")
            
            # 4. Bekle
            await asyncio.sleep(COLLECT_INTERVAL)

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        print("\n🛑 Collector durduruldu.")
        
