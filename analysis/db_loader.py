# analysis/db_loader.py

import sqlite3
import pandas as pd

DB_PATH = "data/market_snapshot.db"

def load_latest_snapshots(symbols: list[str], lookback: int = 20) -> pd.DataFrame:
    """
    symbols: ["BTCUSDT", "ETHUSDT", ...]
    lookback: kaç snapshot geri
    """
    conn = sqlite3.connect(DB_PATH)

    placeholders = ",".join("?" * len(symbols))
    query = f"""
    SELECT *
    FROM snapshot
    WHERE symbol IN ({placeholders})
    ORDER BY ts DESC
    """

    df = pd.read_sql(query, conn, params=symbols)
    conn.close()

    # her symbol + source için son N kayıt
    df = (
        df.sort_values("ts", ascending=False)
          .groupby(["symbol", "source"])
          .head(lookback)
    )

    return df
