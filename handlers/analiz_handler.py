# handlers/analiz_handler.py
import asyncio
import math
from typing import List, Optional, Dict, Any
from datetime import datetime

# Aggregator'ı kullanıyoruz
from analysis.analysis_core import get_aggregator, AnalysisAggregator
from utils.context_logger import get_logger

logger = get_logger(__name__) if hasattr(__import__('utils.context_logger'), 'get_logger') else __import__('logging').getLogger(__name__)

# -----------------------
# Formatter yardımcıları
# -----------------------
def format_volume(v: float) -> str:
    """
    Otomatik hacim formatlama: 1234 -> '1.23K', 1_500_000 -> '1.50M'
    """
    try:
        v = float(v)
    except Exception:
        return str(v)
    if v >= 1_000_000_000:
        return f"{v/1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v/1_000:.2f}K"
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}"

def format_price(p: float) -> str:
    """
    Akıllı fiyat formatlama:
      - >= 1 -> 2 desimal
      - 0.1 - 1 -> 4 desimal
      - 0.01 - 0.1 -> 6 desimal
      - < 0.01 -> 8 desimal
    """
    try:
        p = float(p)
    except Exception:
        return str(p)
    if p >= 1:
        return f"{p:.2f}"
    if p >= 0.1:
        return f"{p:.4f}"
    if p >= 0.01:
        return f"{p:.6f}"
    return f"{p:.8f}"

def is_usdt_pair(symbol: str) -> bool:
    return symbol.upper().endswith("USDT")

# -----------------------
# Report builder
# -----------------------
def build_line(symbol: str, score: float, signal: str, regime: Optional[str], corr: Optional[float], extra: Dict[str,Any]=None) -> str:
    arrow = "↑" if score > 0.55 else ("↓" if score < 0.45 else "→")
    corr_s = f"{corr:.2f}" if isinstance(corr,(int,float)) else "—"
    regime_label = regime or "unknown"
    return f"{symbol}:  {arrow}  α={score:.2f} | Rejim={regime_label} | corr={corr_s}"

def format_summary_table(rows: List[Dict[str,Any]]) -> str:
    lines = ["| Sembol | Skor α | Sinyal | Rejim | Hacim | Fiyat |",
             "|---|---:|---|---|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['symbol']} | {r['score']:.2f} | {r['signal']} | {r.get('regime','-')} | {format_volume(r.get('volume',0))} | {format_price(r.get('price',0))} |")
    return "\n".join(lines)

# -----------------------
# Market scan logic
# -----------------------
async def market_scan(
    symbols: List[str],
    mode: str = "top30",
    top_n: Optional[int] = None,
    only_usdt: bool = True,
    priority: Optional[str] = None
) -> Dict[str,Any]:
    """
    symbols: sembol listesi (örn. ['BTCUSDT','ETHUSDT'] veya [] ise config cüzdanından vs)
    only_usdt: True ise sadece USDT pair'leri tarar
    """
    agg: AnalysisAggregator = await get_aggregator()

    # Filter USDT pairs if requested
    if only_usdt:
        symbols = [s for s in symbols if is_usdt_pair(s)]

    if top_n:
        symbols = symbols[:top_n]

    # Paralel çalış: semboller için run_all çağır
    tasks = [agg.run_all(symbol=symbol, priority=priority) for symbol in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    out_rows = []
    for i, res in enumerate(results):
        symbol = symbols[i]
        if isinstance(res, Exception):
            logger.error(f"Scan failed for {symbol}: {res}")
            continue

        # res: AggregatedResult; içinden composite score çek (varsa)
        score = 0.5
        signal = "neutral"
        regime = None
        price = None
        volume = None
        # Basit extraction: module sonuçları içinden trend_strength veya trend_moment'e bak
        try:
            for ar in res.results:
                if ar.module_name.lower().startswith("trend"):
                    mod_score = ar.data.get("score") if isinstance(ar.data, dict) else None
                    if mod_score is not None:
                        score = mod_score
                        signal = ar.data.get("signal", signal)
                # price/volume için componentlere bak
                if ar.data and isinstance(ar.data, dict):
                    if "price" in ar.data:
                        price = ar.data["price"]
                    if "volume" in ar.data:
                        volume = ar.data["volume"]
        except Exception as e:
            logger.debug(f"Result parsing error for {symbol}: {e}")

        out_rows.append({
            "symbol": symbol,
            "score": score,
            "signal": signal,
            "regime": regime,
            "corr": None,
            "price": price or 0,
            "volume": volume or 0
        })

    # Sort by score desc if mode top30
    if mode and mode.startswith("top"):
        out_rows.sort(key=lambda r: r["score"], reverse=True)

    return {
        "mode": mode,
        "timestamp": datetime.utcnow().isoformat(),
        "rows": out_rows,
        "report_md": format_summary_table(out_rows)
    }

# -----------------------
# CLI-like command entrypoints
# -----------------------
async def cmd_t_default(config_symbols: List[str]):
    """/t -> config içindeki coinleri tara (varsayılan)"""
    return await market_scan(config_symbols, mode="default", only_usdt=True)

async def cmd_t_all(all_symbols: List[str]):
    """/t all -> tüm USDT çiftlerini tara"""
    return await market_scan(all_symbols, mode="all", only_usdt=True)

async def cmd_t_topN(all_symbols: List[str], n: int = 10):
    """/t 10 -> hacme veya score'a göre ilk N'yi tara (burada score temel)"""
    return await market_scan(all_symbols, mode=f"top{n}", top_n=n, only_usdt=True)

async def cmd_t_symbol(symbol: str):
    """/t BTC -> sadece BTC analizi (varsayılan 4 saat)"""
    return await market_scan([symbol], mode="single", only_usdt=True)

# Example usage (async):
# result = await cmd_t_topN(all_symbols, 10)
# print(result['report_md'])
