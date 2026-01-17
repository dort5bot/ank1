# analysis/db_loader.py
# collector’un yazdığı tüm alanlar eksiksiz çekilir

# analysis/db_loader.py
import sqlite3
import pandas as pd
import time

DB_PATH = "data/market.db"


def load_latest_market(
    symbols: list[str],
    micro_lookback_hours: int = 24,
    macro_lookback_days: int = 7
):
    """
    Collector ile %100 uyumlu, eksiksiz veri loader.
    - category alanları korunur
    - macro_lookback_days gerçekten kullanılır
    - analiz & feature engineering için hazır
    """

    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join("?" * len(symbols))

    now = int(time.time())
    micro_cutoff = now - micro_lookback_hours * 3600
    macro_cutoff = now - macro_lookback_days * 86400

    try:
        # --------------------------------------------------
        # 1. MICRO DATA (Binance + Coinalyze)
        # --------------------------------------------------
        micro_query = f"""
        SELECT
            bi.ts,
            bi.symbol,
            bi.price,
            bi.volume,
            bi.category        AS bi_category,

            cz.open_interest,
            cz.funding_rate,
            cz.category        AS cz_category
        FROM bi_market bi
        LEFT JOIN cz_derivatives cz
            ON bi.ts = cz.ts
           AND bi.symbol = cz.symbol
        WHERE bi.symbol IN ({placeholders})
          AND bi.ts >= ?
        ORDER BY bi.ts DESC
        """
        df_micro = pd.read_sql(
            micro_query,
            conn,
            params=(*symbols, micro_cutoff)
        )

        # --------------------------------------------------
        # 2. MACRO GLOBAL DATA (Coingecko)
        # --------------------------------------------------
        macro_query = """
        SELECT
            ts,
            total_mcap,
            total_vol,
            btc_dom,
            eth_dom
        FROM cg_global
        WHERE ts >= ?
        ORDER BY ts DESC
        """
        df_macro = pd.read_sql(
            macro_query,
            conn,
            params=(macro_cutoff,)
        )

        # --------------------------------------------------
        # 3. CATEGORY SNAPSHOT (Latest)
        # --------------------------------------------------
        cat_query = """
        SELECT
            ts,
            category_id,
            market_cap,
            volume,
            change_24h
        FROM cg_categories
        WHERE ts = (SELECT MAX(ts) FROM cg_categories)
        """
        df_categories = pd.read_sql(cat_query, conn)

        # --------------------------------------------------
        # 4. EXCHANGE PRICE BIAS (Latest)
        # --------------------------------------------------
        bias_query = """
        SELECT
            ts,
            coin,
            exchange,
            price
        FROM cg_exchange_bias
        WHERE ts = (SELECT MAX(ts) FROM cg_exchange_bias)
        """
        df_bias = pd.read_sql(bias_query, conn)


        # --------------------------------------------------
        # 5. ETF FLOWS (Yeni eklendi)
        # --------------------------------------------------
        # Son 7 günlük (macro_cutoff) ETF hareketlerini getirir
        etf_query = """
        SELECT 
            ts, asset, date_str, total_flow 
        FROM etf_flows 
        WHERE ts >= ? 
        ORDER BY ts DESC
        """
        df_etf = pd.read_sql(etf_query, conn, params=(macro_cutoff,))


        # --------------------------------------------------
        # TIMESTAMP → DATETIME DÖNÜŞÜMÜ  (dönüş sözlüğü)
        # --------------------------------------------------
        for df in (df_micro, df_macro, df_categories, df_bias):
            if not df.empty:
                df["dt"] = pd.to_datetime(df["ts"], unit="s")

        return {
            "micro": df_micro,
            "macro": df_macro,
            "categories": df_categories,
            "bias": df_bias,
            "etf": df_etf  # Sözlüğe eklendi
        }

    finally:
        conn.close()

