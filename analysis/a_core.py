# a_core.py (updated) - fetch endpoints per-metric and fully parallel pipeline
"""
fetch_data_for_pipeline şunları yapar:
required_endpoints = ["klines"] (sadece klines)
fetcher.fetch_all_for_symbol("BTCUSDT", ["klines"])
Binance'den klines verisini çeker
_klines_to_dataframe ile DataFrame'e çevirir:

calculate_metrics> classical.py'deki ema

“Yükselme ihtimali” skoru (Composite Alpha)
ALPHA_SCORE =
 + 0.30 * trend
 + 0.20 * core
 + 0.15 * mom
 + 0.15 * sentiment
 + 0.10 * flow
 - 0.10 * risk
 
 Yorum
Trend + Core ana motor
Volmom = erken ivme
Sentiment + Flow = yakıt
Risk = frene basan el
Bu skor handler’da değil core’da üretilmeli



🔥 beyin: (alphax + core + risk)

🧠 alphax
trend + mom + sentiment + flow - risk
→ “Yükselme / düşme isteği var mı?”

🧱 core
trend + vol + regim + risk
→ “Bu istek yapısal olarak sağlıklı mı?”

🚨 risk
→ “Her şey güzel ama patlar mı?”

✅ LONG için ideal senaryo
alphax > +0.35
core   > +0.20
risk   < 0.30


❌ Sahte yükseliş (çok kritik!)
alphax > +0.40
core   < 0
risk   > 0.50


Trend-follow uygun mu?
complexity < 0.4
vol        < 0.5
regim      > 0

Chop / range ortamı
📌 Bu kombinasyon olmadan trend sinyali kullanmak kör uçuş olur.
complexity > 0.6
regim      < 0


Trade açılabilir mi?
microstructure > 0
liqrisk        < 0.3

-özet--
| Amaç                          | Gerekli Kombinasyon        |
| ----------------------------- | -------------------------- |
| **Ana yön kararı**            | `alphax + core + risk`     |
| **Trend ortamı mı?**          | `complexity + regim + vol` |
| **Scalp teyidi**              | `trend + mom + order`   |
| **Sentiment tuzağı filtresi** | `sentflow + trend + risk`  |
| **Trade izni**                | `microstructure + liqrisk` |

| Composite | Cevapladığı Soru          |
| --------- | ------------------------- |
| trend     | *Yön var mı?*             |
| mom    | *hız:Yön hızlanıyor mu?*      |
| vol       | *rejim:Ortam ne kadar oynak?*   |
| sentiment | *ağırlık:Pozisyonlanma ne diyor?* |
| risk      | *Bu iş patlar mı?*        |



"""

from __future__ import annotations
import asyncio
import logging
import math
import ast
import concurrent.futures
import os
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Set
from functools import partial

from analysis.metricresolver import get_default_resolver
from utils.binance_api.binance_a import BinanceAggregator

import pandas as pd
import numpy as np
from datetime import datetime

# ------------------------------------------------------------
# Logger
# ------------------------------------------------------------
logger = logging.getLogger("analysis.core")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(h)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

# ------------------------------------------------------------
# Globals & Constants
# ------------------------------------------------------------
DEFAULT_DATA_MODEL = "pandas"

# ------------------------------------------------------------
# COMPOSITES / MACROS maps 
# ------------------------------------------------------------

COMPOSITES = {
    # ✅ binance api ile başarılı
    "trend": { # Sadece directional bias
        "depends": ["ema", "macd", "rsi", "stochastic_oscillator"],
        "formula": "0.30*ema + 0.30*macd + 0.20*rsi + 0.20*stochastic_oscillator",
    },
    "mom": { # Hareket var mı ve güçleniyor mu
        "depends": ["roc", "adx", "atr"],
        "formula": "0.45*roc + 0.35*adx - 0.20*atr",
    },
    "vol": { # Bu piyasa trend taşır mı
        "depends": ["historical_volatility", "garch_1_1", "hurst_exponent"],
        "formula": "0.40*historical_volatility + 0.35*garch_1_1 + 0.25*(1 - hurst_exponent)",
    },
    
    # ✅ ⚠️ Hesaplanabilir, yaklaşık bilgi veriri, gerçek anlamlılık sınırlı.
    "sentiment": {
        "depends": ["funding_rate", "funding_premium", "oi_trend"],
        "formula": "0.35*funding_rate + 0.25*funding_premium + 0.40*oi_trend",
    },
    "risk": {
        "depends": ["volatility_risk", "liquidity_depth_risk", "price_impact_risk"],
        "formula": "0.40*volatility_risk + 0.35*liquidity_depth_risk + 0.25*price_impact_risk",
    },
    "regim": {
        "depends": ["advance_decline_line", "volume_leadership", "performance_dispersion"],
        "formula": "0.45*advance_decline_line + 0.35*volume_leadership + 0.20*performance_dispersion",
    },
    "entropy": {# entropy_fractal
        "depends": ["entropy_index", "fractal_dimension_index_fdi", "hurst_exponent", "variance_ratio_test"],
        "formula": "0.35*entropy_index + 0.25*fractal_dimension_index_fdi - 0.25*hurst_exponent - 0.15*variance_ratio_test",
    },
    
    # ⚠️ anlamsız metrikler, yetersiz veri nedeniyle
    "liqu": {
        "depends": ["liquidity_density","microprice_deviation"],
        "formula": "0.5*liquidity_density+0.5*microprice_deviation",
    },
    "liqrisk": {
        "depends": ["liquidity_density", "liquidity_gaps"], # ❌ "market_impact", "cascade_risk"
        "formula": "0.5*liquidity_density - 0.2*liquidity_gaps",
    },
    
    # ❌ hesaplanamaz binance api ile
    "order": {
        "depends": ["ofi", "cvd", "microprice_deviation"], # "taker_dominance_ratio"
        "formula": "0.45*ofi + 0.35*cvd  + 0.20*microprice_deviation",
    },
    "flow": {
        "depends": ["etf_net_flow", "exchange_netflow", "stablecoin_flow"],
        "formula": "0.4*etf_net_flow - 0.3*exchange_netflow + 0.3*stablecoin_flow",
    },

}	
# Binance API ile doğrudan elde edilemeyenler
# market_impact, depth_elasticity, taker_dominance_ratio (ham veriden türetilir ama direkt verilmez)
# ⚠️ garch_1_1, hurst_exponent, fdi, variance_ratio_test, fractal_dimension_index_fdi




MACROS = {	
    "core": {
        "depends": ["trend", "vol", "regim", "risk"],
        "formula": "0.35*trend + 0.25*vol + 0.25*regim + 0.15*risk",
    },
    "alphax":{
        "depends": ["trend","mom","sentiment","flow","risk"],
        "formula": "0.35*trend + 0.25*mom + 0.20*sentiment + 0.15*flow - 0.15*risk", 
    },
    "sentri": { # Sinyal > 0 long, = 0 bekle, <0 short
        "depends": ["sentiment", "entropy", "regim", "liquidity_density", "risk"],
        "formula": "0.24*sentiment + 0.18*entropy + 0.2*regim+ 0.2*liquidity_density - 0.18*risk",
    },
    
    "coreliq": {
        "depends": ["trend", "vol", "regim", "risk", "liqu"],
        "formula": "0.27*trend + 0.20*vol + 0.20*regim + 0.18*risk + 0.15*liqu",
    },
    "complexity": {
        "depends": ["entropy", "vol"],
        "formula": "0.6*entropy + 0.4*vol",
    },
    "sentflow": {
        "depends": ["sentiment", "flow"],
        "formula": "0.55*sentiment + 0.45*flow",
    },
    "microstructure": {
        "depends": ["liqu", "liqrisk", "order"],
        "formula": "0.4*liqu+ 0.35*liqrisk + 0.25*order",
    },
}

# ------------------------------------------------------------
# Formula compile helpers (unchanged)
# ------------------------------------------------------------
_ALLOWED_NODES = {
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load, ast.Add,
    ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Name,
    ast.Constant, ast.Mod, ast.FloorDiv, ast.Call
}

_formula_compile_cache: Dict[str, Any] = {}
_formula_compile_lock = asyncio.Lock()

def _validate_ast(node: ast.AST) -> None:
    for n in ast.walk(node):
        if type(n) not in _ALLOWED_NODES:
            raise ValueError(f"Disallowed AST node: {type(n).__name__}")

async def _get_compiled_formula(formula: str):
    if not formula:
        return None
    if formula in _formula_compile_cache:
        return _formula_compile_cache[formula]
    async with _formula_compile_lock:
        if formula in _formula_compile_cache:
            return _formula_compile_cache[formula]
        try:
            tree = ast.parse(formula, mode="eval")
            _validate_ast(tree)
            code = compile(tree, "<formula>", "eval")
            _formula_compile_cache[formula] = code
            return code
        except Exception as e:
            logger.warning(f"Formula compile failed: {formula} -> {e}")
            _formula_compile_cache[formula] = None
            return None

def evaluate_compiled_formula(code_obj, ctx: Dict[str, float]) -> float:
    if code_obj is None: return 0.0
    # ctx içindeki nan değerlerini 0.0 ile temizle
    safe_ctx = {k: (v if not math.isnan(v) else 0.0) for k, v in ctx.items()}
    try:
        val = eval(code_obj, {"__builtins__": {}}, safe_ctx)
        return float(val) if val is not None else 0.0
    except Exception:
        return 0.0


# ------------------------------------------------------------
# Utilities from original file: resolve_scores_to_metrics, extract_final_value, etc.
# Keep them the same as original. (Copy/paste from your original a_core.py)
def resolve_scores_to_metrics(requested_scores: List[str], COMPOSITES: Dict = None, MACROS: Dict = None) -> Dict[str, List[str]]:
    COMPOSITES = COMPOSITES or {}
    MACROS = MACROS or {}
    out = {}
    for score_name in requested_scores:
        metrics = []
        if score_name in COMPOSITES:
            metrics.extend(COMPOSITES[score_name].get("depends", []))
        elif score_name in MACROS:
            for dep in MACROS[score_name].get("depends", []):
                if dep in COMPOSITES:
                    metrics.extend(COMPOSITES[dep].get("depends", []))
        out[score_name] = sorted(set(metrics))
    return out

# (extract_final_value function unchanged - paste from original)

def extract_final_value(raw_result: Any, metric_name: str) -> float:
    if raw_result is None:
        return float("nan")
    
    try:
        # 1. Zaten float/int ise direkt döndür
        if isinstance(raw_result, (int, float, np.number)):
            return float(raw_result)  # ← BU KESİNLİKLE ÇALIŞACAK
        
        # 2. Pandas Series
        if isinstance(raw_result, pd.Series):
            if raw_result.empty:
                return float("nan")
            return float(raw_result.iat[-1])
        
        # 3. DataFrame
        if isinstance(raw_result, pd.DataFrame):
            if raw_result.empty:
                return float("nan")
            for col in ['value', 'score', metric_name, 'result']:
                if col in raw_result.columns:
                    try:
                        return float(raw_result[col].iat[-1])
                    except Exception:
                        continue
            try:
                return float(raw_result.iat[-1, 0])
            except Exception:
                return float("nan")
        if isinstance(raw_result, (list, tuple, np.ndarray)):
            try:
                if isinstance(raw_result, np.ndarray):
                    if raw_result.size == 0:
                        return float("nan")
                    return float(np.asarray(raw_result).flat[-1])
                else:
                    if len(raw_result) == 0:
                        return float("nan")
                    return float(raw_result[-1])
            except Exception:
                return float("nan")
        if isinstance(raw_result, dict):
            for key in ('value', 'score', metric_name, 'result', 'data'):
                if key in raw_result:
                    try:
                        return float(raw_result[key])
                    except Exception:
                        continue
            for v in raw_result.values():
                if isinstance(v, (int, float, np.number)):
                    return float(v)
            return float("nan")
        if isinstance(raw_result, (int, float, np.number)):
            return float(raw_result)
        if isinstance(raw_result, str):
            try:
                return float(raw_result)
            except Exception:
                return float("nan")
        return float(str(raw_result))
    except Exception:
        return float("nan")


# ------------------------------------------------------------
# ThreadPool executor (shared) - Global Seviye
# ------------------------------------------------------------

_CPU = os.cpu_count() or 2
_DEFAULT_MAX_WORKERS = min(max(4, _CPU * 2), 20)
_global_executor = concurrent.futures.ThreadPoolExecutor(max_workers=_DEFAULT_MAX_WORKERS)


_CPU_EXECUTOR = concurrent.futures.ProcessPoolExecutor(
    max_workers=os.cpu_count()
)
_IO_EXECUTOR = _global_executor


def run_sync_metric(
    fn: Callable,
    inp,
    params: Dict,
    metric_name: str
) -> Tuple[str, float]:
    """
    Global seviyeye taşınan senkron metrik çalıştırıcı.
    Artık pickle edilebilir (ProcessPool için uygun).
    """
    try:
        # Not: extract_final_value fonksiyonunun da globalde tanımlı olduğundan emin olun
        raw = fn(inp, **params)
        val = extract_final_value(raw, metric_name)
        
        # NaN kontrolü ve kırpma
        if math.isnan(val):
            return metric_name, float("nan")
            
        return metric_name, float(np.clip(val, -1, 1))
    except Exception as e:
        # Alt processlerdeki hataları ana sürece bildirmek için log veya hata dönüyoruz
        return metric_name, float("nan")

async def run_async_metric(
    fn: Callable,
    inp,
    params: Dict,
    metric_name: str
) -> Tuple[str, float]:
    """Asenkron metrik çalıştırıcı (Global scope)"""
    try:
        raw = await fn(inp, **params)
        val = extract_final_value(raw, metric_name)
        
        if math.isnan(val):
            return metric_name, float("nan")
            
        return metric_name, float(np.clip(val, -1, 1))
    except Exception as e:
        return metric_name, float("nan")
        
# ------------------------------------------------------------
# Data preparation (unchanged)
# ------------------------------------------------------------

def prepare_data(data: pd.DataFrame, def_info: Dict) -> Any:
    if data is None or data.empty:
        return pd.DataFrame()

    data_model = def_info.get("data_model", "pandas")
    required_cols = def_info.get("required_columns", []) or []

    if required_cols:
        selected_cols = {}

        for col in required_cols:
            # 1️⃣ Direkt varsa
            if col in data.columns:
                selected_cols[col] = data[col]
                continue

            # 2️⃣ suffix’li kolonları ara (close__klines gibi)
            matches = [c for c in data.columns if c.startswith(col + "__")]
            if matches:
                # ilk bulunanı al (klines genelde tek olur)
                selected_cols[col] = data[matches[0]]

        if not selected_cols:
            return pd.DataFrame()

        selected = pd.DataFrame(selected_cols, index=data.index)
    else:
        selected = data

    if data_model == "numpy":
        try:
            return selected.to_numpy()
        except Exception:
            return selected

    if data_model == "polars":
        try:
            import polars as pl
            return pl.from_pandas(selected)
        except Exception:
            return selected

    return selected



# ------------------------------------------------------------
# Metric execution (unchanged)
# ------------------------------------------------------------
# sadece debug, debugsuz olanı altta
# max_workers: int = None
# Parallel, NaN-safe, CPU-aware metric execution engine.

async def calculate_metrics(
    data: pd.DataFrame,
    metric_defs: Dict[str, Dict],
    max_workers: int = None
) -> Dict[str, float]:
    logger.debug(f"calculate_metrics called with {len(metric_defs)} metrics")

    if data is None or data.empty or not metric_defs:
        logger.warning("No data or metric definitions")
        return {}

    loop = asyncio.get_running_loop()
    results: Dict[str, float] = {}
    tasks: List[asyncio.Future] = []

    # Executor'lar (Bunların yukarıda tanımlandığını varsayıyoruz)
    CPU_EXECUTOR = _CPU_EXECUTOR
    IO_EXECUTOR = _global_executor

    for name, def_info in metric_defs.items():
        func = def_info.get("function")
        params = def_info.get("default_params", {}) or {}
        exec_type = def_info.get("execution_type", "sync")
        metadata = def_info.get("metadata", {}) or {}

        if func is None:
            results[name] = float("nan")
            continue

        try:
            input_data = prepare_data(data, def_info)
        except Exception as e:
            logger.debug(f"prepare_data failed: {name} → {e}")
            results[name] = float("nan")
            continue

        # Minimum bar kontrolü
        min_bars = metadata.get("min_bars", 1)
        if hasattr(input_data, "__len__") and len(input_data) < min_bars:
            results[name] = float("nan")
            continue

        # --- ASYNC GÖREVLER ---
        if exec_type == "async":
            tasks.append(
                asyncio.create_task(
                    run_async_metric(func, input_data, params, name)
                )
            )
            continue

        # --- SYNC GÖREVLER (Executor ile) ---
        category = metadata.get("category", "")
        # Kategoriye göre doğru executor seçimi
        executor = CPU_EXECUTOR if category in ("advanced", "volatility") else IO_EXECUTOR

        # run_in_executor artık global 'run_sync_metric' fonksiyonunu sorunsuzca pickle edebilir
        future = loop.run_in_executor(
            executor,
            run_sync_metric,
            func,
            input_data,
            params,
            name
        )
        tasks.append(future)

    # --- SONUÇLARI TOPLA ---
    if tasks:
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for item in gathered:
            if isinstance(item, tuple) and len(item) == 2:
                k, v = item
                results[k] = v
            elif isinstance(item, Exception):
                # Hataları görünür hale getirin
                logger.error(f"Kritik Metrik Hatası: {item}")

    return results
    
# ------------------------------------------------------------
# ------------------------------------------------------------
# Composite/Macro calculation (unchanged)
# ------------------------------------------------------------

# debug
async def calculate_formula_scores(
    source_values: Dict[str, float],
    definitions: Dict[str, dict]
) -> Dict[str, float]:

    out = {}
    formula_map: Dict[str, Any] = {}

    logger.debug(f"calculate_formula_scores SOURCE VALUES: {source_values}")


    # 1️⃣ Compile (cached)
    for name, info in definitions.items():
        
        # DEBUG: VOL için özel log
        if name == "vol":
            deps = info.get("depends", [])
            logger.debug(f"DEBUG VOL calculation - deps: {deps}")
            logger.debug(
                f"DEBUG VOL values - atr: {source_values.get('atr')}, "
                f"hist_vol: {source_values.get('historical_volatility')}, "
                f"garch: {source_values.get('garch_1_1')}, "
                f"hurst: {source_values.get('hurst_exponent')}"
            )
        code = await _get_compiled_formula(info.get("formula"))
        formula_map[name] = (code, info.get("depends", []))

    # 2️⃣ Evaluate (NaN-robust)
    for name, (code, deps) in formula_map.items():
        if not code or not deps:
            out[name] = float("nan")
            continue

        values = {}
        valid_weights = 0.0

        for dep in deps:
            v = source_values.get(dep, float("nan"))
            if not math.isnan(v):
                values[dep] = float(v)
                valid_weights += 1
            else:
                values[dep] = 0.0  # NaN kırıcı

        if valid_weights == 0:
            out[name] = float("nan")
            continue

        # regime factor (safe)
        if "hurst_exponent" in values:
            h = values["hurst_exponent"]
            if not math.isnan(h):
                values["regime_factor"] = max(-1.0, min(1.0, (h - 0.5) * 2.0))

        out[name] = evaluate_compiled_formula(code, values)

    return out





# ------------------------------------------------------------
# Data fetcher: fetch multiple endpoints per-symbol in parallel
# ------------------------------------------------------------
class BinanceDataFetcher:
    """Wrapper to fetch multiple endpoints for a symbol in parallel and merge into one DataFrame."""
    def __init__(self):
        self.aggregator = None

    async def initialize(self):
        if self.aggregator is None:
            self.aggregator = await BinanceAggregator.get_instance()
        return self.aggregator

    def normalize_depth(self, data, symbol, top_n=20):
        """
        RAW depth verisini KORU, sadece formatını düzenle.
        risk.py'nin beklediği [side, price, size] formatına çevir.
        """
        bids = data.get("bids", [])[:top_n]
        asks = data.get("asks", [])[:top_n]

        if not bids or not asks:
            return pd.DataFrame()

        rows = []
        
        # Bids (highest to lowest)
        for i, (price_str, size_str) in enumerate(bids):
            try:
                rows.append({
                    'level': i,
                    'side': 'bid',
                    'price': float(price_str),
                    'size': float(size_str),
                    'symbol': symbol,
                    'timestamp': pd.Timestamp.utcnow()
                })
            except (ValueError, TypeError):
                continue
        
        # Asks (lowest to highest)
        for i, (price_str, size_str) in enumerate(asks):
            try:
                rows.append({
                    'level': i + len(bids),
                    'side': 'ask',
                    'price': float(price_str),
                    'size': float(size_str),
                    'symbol': symbol,
                    'timestamp': pd.Timestamp.utcnow()
                })
            except (ValueError, TypeError):
                continue
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Doğru sıralama
        df_bids = df[df['side'] == 'bid'].sort_values('price', ascending=False)
        df_asks = df[df['side'] == 'ask'].sort_values('price', ascending=True)
        
        result = pd.concat([df_bids, df_asks], ignore_index=True)
        result.set_index('timestamp', inplace=True)
        
        return result
    
    
    # ================================
    # 🔥 Yeni sistem — tek doğru endpoint çağrısı
    # ================================
    async def fetch_endpoint_with_params(self, symbol: str, endpoint_name: str, params: Dict) -> pd.DataFrame:
        
        await self.initialize()
        try:
            data = await self.aggregator.get_public_data(
                endpoint_name=endpoint_name,
                **params
            )
            
            if endpoint_name == "klines":
                return _klines_to_dataframe(data, symbol)
                
            elif endpoint_name == "depth":   # 🔥 BURADA DEĞİŞTİRDİK
                # Artık RAW depth verisini formatlayarak döndürüyoruz
                return self.normalize_depth(data, symbol, top_n=20)
             
            else:
                return _endpoint_to_dataframe(data, symbol, endpoint_name)
        except Exception as e:
            logger.warning(f"fetch_endpoint failed: {symbol} {endpoint_name} -> {e}")
            return pd.DataFrame()
        
    
    # ================================
    # 🔥 Tüm endpointleri paralel çek
    # ================================

    async def fetch_all_for_symbol(self, symbol: str, endpoint_params: Dict[str, Dict]):
        tasks = {
            ep: asyncio.create_task(
                self.fetch_endpoint_with_params(symbol, ep, params)
            )
            for ep, params in endpoint_params.items()
        }

        results: Dict[str, pd.DataFrame] = {}
        for ep, task in tasks.items():
            try:
                df = await task
                results[ep] = df if not df.empty else pd.DataFrame()
            except Exception as e:
                logger.error(f"Failed to fetch {ep} for {symbol}: {e}")
                results[ep] = pd.DataFrame()

        # Merge all DataFrames
        merged = None
        for ep, df in results.items():
            if df is None or df.empty:
                continue

            df_copy = df.copy()
            cols = [c for c in df_copy.columns if c != "symbol"]
            rename_map = {c: f"{c}__{ep}" for c in cols}
            df_copy = df_copy.rename(columns=rename_map)

            # timestamp varsa DatetimeIndex yap
            if "timestamp" in df_copy.columns:
                df_copy["timestamp"] = pd.to_datetime(
                    df_copy["timestamp"], unit="ms", errors="coerce", utc=True
                )
                df_copy = df_copy.set_index("timestamp")
            else:
                # timestamp yoksa index reset ve tek seviyeye düşür
                df_copy = df_copy.reset_index()
                df_copy.index.name = "timestamp"

            # Merge için tüm DataFrame'leri tek seviyeli index yap
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy.index = pd.Index(df_copy.index, name="timestamp")

            if merged is None:
                merged = df_copy
            else:
                # Artık farklı seviyeler hatası olmayacak
                merged = pd.merge(
                    merged, df_copy, left_index=True, right_index=True, how="outer"
                )

                # Fazla symbol kolonlarını temizle
                sym_cols = [c for c in merged.columns if c == "symbol" or c.endswith("__symbol")]
                if len(sym_cols) > 1:
                    for c in sym_cols[1:]:
                        merged.drop(columns=[c], inplace=True, errors="ignore")

        if merged is not None:
            merged.sort_index(inplace=True)
        else:
            merged = pd.DataFrame()

        return merged


# ================================
# 🔥 fetch_data_for_pipeline — endpoint param üretme & fetch yönetimi
# ================================
def filter_healthy_symbols(results):
    healthy = {}

    for sym, data in results.items():
        s = data["scores"]

        if s["LIQRISK"] > 0.5:
            continue
        if s["ENTROPY"] > 0.8:
            continue
        if s["REGIM"] < -0.3:
            continue
        if s["VOL"] > 0.6 and s["TREND"] <= 0:
            continue

        healthy[sym] = data

    return healthy


async def get_top_volume_symbols(count: int = 10):
    """
    En yüksek hacimli sembolleri filtreler ve getirir.
    Performans için önce ilk 30-40 tanesini ayırır, 
    sonra içinden istenen n tanesini döndürür.
    """
    try:
        from utils.binance_api.binance_a import BinanceAggregator

        # 1. Tüm 24s ticker verilerini çek
        aggregator = await BinanceAggregator.get_instance()
        
        all_tickers = await aggregator.get_public_data(
            endpoint_name="ticker_24hr"
        )
        
        if not all_tickers:
            return ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # fallback

        # 2. Sadece USDT çiftlerini ve 'sağlıklı' olanları filtrele
        # (UP, DOWN, BULL, BEAR gibi kaldıraçlı tokenları eliyoruz)      
        excluded_keywords = ["UP", "DOWN", "BULL", "BEAR"]
        stable_coins = ["USDC", "FDUSD", "USD1", "TUSD", "DAI", "USDP", "EUR", "AEUR", "PAXG"]
        valid_pairs = []

        for ticker in all_tickers:
            symbol = ticker.get('symbol', '')
            if not symbol:
                continue
                
            # USDT ile bitiyor mu?
            # Kaldıraçlı token içermiyor mu?
            # Diğer stable coin'leri içermiyor mu?
            if (symbol.endswith('USDT') and 
                not any(k in symbol for k in excluded_keywords) and
                not any(s in symbol for s in stable_coins)):
                valid_pairs.append(ticker)

        # 3. Hacme (quoteVolume) göre büyükten küçüğe sırala
        # quoteVolume = USDT cinsinden toplam hacim
        sorted_pairs = sorted(
            valid_pairs, 
            key=lambda x: float(x.get('quoteVolume', 0)), 
            reverse=True
        )

        # 4. İlk 40 tanesini "Güvenli Havuz" olarak belirle (Performans Sınırı)
        safe_pool = sorted_pairs[:40] if len(sorted_pairs) > 40 else sorted_pairs

        # 5. Kullanıcının istediği 'count' kadarını bu 40 içinden al
        final_count = min(count, len(safe_pool))
        final_symbols = [t['symbol'] for t in safe_pool[:final_count]]

        # Eğer hiç sembol kalmadıysa fallback
        if not final_symbols:
            return ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
            
        return final_symbols

    except Exception as e:
        logger.error(f"❌ get_top_volume_symbols hatası: {e}")
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Hata anında fallback





   
async def fetch_data_for_pipeline(symbol, metric_defs, interval="1h", limit=500):
    # Eğer limit belirtilmezse Binance varsayılan olarak az veri gönderebilir
    # Metriklerin "ısınması" için en az 100 bar çekmelisiniz

    is_single = isinstance(symbol, str)
    symbols = [symbol] if is_single else symbol

    fetcher = BinanceDataFetcher()
    await fetcher.initialize()

    # 1) Tüm metriklerden endpoint → param factory çıkar
    endpoint_factories = {}
    for mdef in metric_defs.values():
        for ep, factory in mdef.get("endpoint_params", {}).items():
            endpoint_factories[ep] = factory

    # 2) Her symbol için parametreleri oluştur
    tasks = {}
    for sym in symbols:
        params_for_symbol = {
            ep: factory(sym, interval, limit)
            for ep, factory in endpoint_factories.items()
        }

        tasks[sym] = asyncio.create_task(
            fetcher.fetch_all_for_symbol(sym, params_for_symbol)
        )

    # 3) Sonuçları topla
    results = {}
    for sym, task in tasks.items():
        try:
            df = await task
            results[sym] = df
        except Exception as e:
            logger.error(f"Failed fetch for {sym}: {e}")
            results[sym] = pd.DataFrame()

    return results[symbol] if is_single else results


def _klines_to_dataframe(klines: List, symbol: str) -> pd.DataFrame:
    if not klines:
        logger.warning(f"Klines empty for {symbol}")
        return pd.DataFrame()
    
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
    df.set_index('timestamp', inplace=True)
    
    # HEP SINIR NUMERIC KOLONLAR
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # taker_buy_base ve taker_buy_quote da numeric olmalı
    if 'taker_buy_base' in df.columns:
        df['taker_buy_base'] = pd.to_numeric(df['taker_buy_base'], errors='coerce')
    if 'taker_buy_quote' in df.columns:
        df['taker_buy_quote'] = pd.to_numeric(df['taker_buy_quote'], errors='coerce')
    
    # trades ve close_time integer
    if 'trades' in df.columns:
        df['trades'] = pd.to_numeric(df['trades'], errors='coerce').fillna(0).astype(int)
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce').fillna(0).astype(int)
    
    df['symbol'] = symbol
    
    # DEBUG: Tip kontrolü
    logger.debug(f"DataFrame dtypes after conversion:\n{df.dtypes}")
    
    return df


    
def _endpoint_to_dataframe(data: Any, symbol: str, endpoint_name: str) -> pd.DataFrame:
    """
    Generic normalizer for endpoints other than klines.
    Tries to infer timestamp column -> index; otherwise returns table with 'value__endpoint' if scalar list given.
    """
    if data is None:
        return pd.DataFrame()
    # If data already a DataFrame-like
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce', utc=True)
                df.set_index('timestamp', inplace=True)
            except Exception:
                pass
        df['symbol'] = symbol
        return df

    
    # If list of dicts
    if isinstance(data, list):
        try:
            df = pd.DataFrame(data)

            # -------------------------------
            # 🔥 BINANCE OPEN INTEREST PATCH
            # -------------------------------
            if endpoint_name == "open_interest_hist":
                if "sumOpenInterest" in df.columns:
                    df["open_interest"] = pd.to_numeric(
                        df["sumOpenInterest"], errors="coerce"
                    )
            # -------------------------------

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(
                    df['timestamp'], unit='ms', errors='coerce', utc=True
                )
                df.set_index('timestamp', inplace=True)

            df['symbol'] = symbol
            return df

        except Exception:
            pass



    # If a dict with keys -> try to create series
    if isinstance(data, dict):
        try:
            # flatten scalar dicts to DataFrame with single timestamp=now
            s = pd.Series(data)
            df = pd.DataFrame([s])
            df.index = pd.to_datetime([pd.Timestamp.utcnow()])
            df['symbol'] = symbol
            return df
        except Exception:
            pass

    # If scalar or unknown -> return small DF with "value"
    try:
        return pd.DataFrame([{ 'value': data, 'symbol': symbol }], index=[pd.Timestamp.utcnow()])
    except Exception:
        return pd.DataFrame()

# ------------------------------------------------------------
# Verify metric definitions (unchanged)
# ------------------------------------------------------------
def verify_metric_definitions(metric_defs: Dict[str, Dict]) -> Dict[str, bool]:
    out = {}
    for name, d in metric_defs.items():
        ok = isinstance(d, dict) and d.get("function") is not None and d.get("execution_type") in ("sync", "async")
        out[name] = bool(ok)
        if not ok:
            logger.debug(f"Metric definition invalid: {name}")
    return out

def get_metric_metadata(metric_defs: Dict) -> Dict[str, Dict]:
    meta = {}
    for name, d in metric_defs.items():
        if not d:
            continue
        meta[name] = {
            "data_model": d.get("data_model", "unknown"),
            "execution_type": d.get("execution_type", "unknown"),
            "category": d.get("metadata", {}).get("category", "unknown"),
            "module": d.get("metadata", {}).get("module_name", "unknown"),
        }
    return meta

# ------------------------------------------------------------
# Single-symbol pipeline (updated order: resolve defs -> fetch endpoints -> calc)
# ------------------------------------------------------------

async def _run_single_pipeline(
    symbol: str,
    requested_scores: List[str],
    raw_df: Optional[pd.DataFrame] = None,
    interval: str = "1h",
    limit: int = 500
) -> Dict[str, Any]:

    logger.info(f"Pipeline start: {symbol}")

    # -------------------------------------------------
    # 1) Resolve required metrics
    # -------------------------------------------------
    score_to_metrics = resolve_scores_to_metrics(requested_scores, COMPOSITES, MACROS)
    all_required_metrics = sorted(
        {m for metrics in score_to_metrics.values() for m in metrics}
    )

    resolver = get_default_resolver()
    metric_defs = resolver.resolve_multiple_definitions(all_required_metrics)


    logger.debug(f"All required metrics: {all_required_metrics}")
    logger.debug(f"Metric defs keys: {list(metric_defs.keys())}")


    # validate & filter invalid metric defs
    valid_map = verify_metric_definitions(metric_defs)
    
    logger.debug(f"Valid metrics: {[k for k, v in valid_map.items() if v]}")
    logger.debug(f"Invalid metrics: {[k for k, v in valid_map.items() if not v]}")
    
    
    metric_defs = {k: v for k, v in metric_defs.items() if valid_map.get(k)}

    if not metric_defs:
        return {"error": "No valid metric definitions", "symbol": symbol}

    # -------------------------------------------------
    # 2) Collect required endpoints (metadata only)
    # -------------------------------------------------
    required_endpoints_set: Set[str] = set()

    for m_info in metric_defs.values():
        for ep in m_info.get("required_endpoints", []) or []:
            required_endpoints_set.add(ep)

    # Heuristic: OHLCV ihtiyacı varsa klines garanti
    if "klines" not in required_endpoints_set:
        for m_info in metric_defs.values():
            req_cols = m_info.get("required_columns", []) or []
            if any(c in ("open", "high", "low", "close", "volume", "returns") for c in req_cols):
                required_endpoints_set.add("klines")
                break

    required_endpoints = sorted(required_endpoints_set)

    # -------------------------------------------------
    # 3) Fetch data (merged)
    # -------------------------------------------------
    if raw_df is None:
        try:
            merged_df = await fetch_data_for_pipeline(
                symbol, metric_defs, interval, limit
            )
        except Exception as e:
            logger.error(f"Fetch failed for {symbol}: {e}")
            return {"error": f"Data fetch failed: {e}", "symbol": symbol}
    else:
        merged_df = raw_df

    if merged_df is None or merged_df.empty:
        return {"error": "No data", "symbol": symbol}

    # ❌ GLOBAL COLUMN NORMALIZATION YOK
    # prepare_data + required_columns tek doğru yol

    # -------------------------------------------------
    # 4) Calculate metrics
    # -------------------------------------------------
    metric_results = await calculate_metrics(
        merged_df,
        metric_defs,
        max_workers=_DEFAULT_MAX_WORKERS
    )

    # -------------------------------------------------
    # 5) Composite & macro scores
    # -------------------------------------------------
    composite_scores = await calculate_formula_scores(metric_results, COMPOSITES)
    macro_scores = await calculate_formula_scores(composite_scores, MACROS)

    # -------------------------------------------------
    # 6) Final score assembly
    # -------------------------------------------------
    final_scores: Dict[str, float] = {}

    for s in requested_scores:
        if s in composite_scores:
            final_scores[s] = composite_scores[s]
        elif s in macro_scores:
            final_scores[s] = macro_scores[s]
        elif s in metric_results:
            final_scores[s] = metric_results[s]
        else:
            final_scores[s] = float("nan")

    result = {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "scores": final_scores,
        "metrics": metric_results,
        "composites": composite_scores,
        "macros": macro_scores,
        "metadata": {
            "metrics_count": len(metric_results),
            "valid_metrics": list(metric_results.keys()),
            "metric_defs_summary": get_metric_metadata(metric_defs),
            "required_endpoints": required_endpoints,
        },
    }

    logger.info(f"Pipeline done: {symbol}")
    return result


# ------------------------------------------------------------
# Public run_pipeline (unchanged behavior, supports single/batch)
# ------------------------------------------------------------
async def run_pipeline(symbol: Union[str, List[str]], requested_scores: List[str], raw_data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None, **kwargs) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    if isinstance(symbol, list):
        raw_map = raw_data if isinstance(raw_data, dict) else {}
        tasks = [asyncio.create_task(_run_single_pipeline(sym, requested_scores, raw_map.get(sym), **kwargs)) for sym in symbol]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return {sym: res for sym, res in zip(symbol, results)}
    else:
        return await _run_single_pipeline(symbol, requested_scores, raw_data, **kwargs)

def run_pipeline_sync(symbol: Union[str, List[str]], requested_scores: List[str], raw_data: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]] = None, timeout: int = 30, **kwargs) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    try:
        return asyncio.run(run_pipeline(symbol, requested_scores, raw_data, **kwargs))
    except Exception as e:
        logger.error(f"run_pipeline_sync failed: {e}")
        return {"error": str(e), "symbol": symbol}

# ------------------------------------------------------------
# Debug helpers etc. (unchanged)
# ------------------------------------------------------------
async def debug_metric_calculation(metric_name: str, data: pd.DataFrame) -> Dict:
    resolver = get_default_resolver()
    def_info = resolver.resolve_metric_definition(metric_name)
    info = {
        "metric_name": metric_name,
        "data_model": def_info.get("data_model"),
        "execution_type": def_info.get("execution_type"),
        "required_columns": def_info.get("required_columns", []),
        "available_columns": list(data.columns),
        "normalization": def_info.get("normalization", {}),
        "category": def_info.get("metadata", {}).get("category"),
    }
    func = def_info.get("function")
    if func:
        params = def_info.get("default_params", {})
        try:
            if def_info.get("execution_type") == "async":
                raw = await func(data, **params)
            else:
                raw = func(data, **params)
            info["raw_result_type"] = type(raw).__name__
            info["final_value"] = extract_final_value(raw, metric_name)
        except Exception as e:
            info["error"] = str(e)
    return info

def get_system_status() -> Dict:
    resolver = get_default_resolver()
    all_metrics = resolver.get_available_metrics()
    sample = {}
    for m in ["ema", "rsi", "macd"]:
        try:
            d = resolver.resolve_metric_definition(m)
            sample[m] = {"data_model": d.get("data_model"), "execution_type": d.get("execution_type")}
        except Exception:
            pass
    return {
        "total_metrics": len(all_metrics),
        "sample_metrics": sample,
        "default_data_model": DEFAULT_DATA_MODEL,
        "executor_workers": _DEFAULT_MAX_WORKERS
    }


# -------
async def test_open_interest_hist_raw():
    from utils.binance_api.binance_a import BinanceAggregator

    agg = await BinanceAggregator.get_instance()

    print("👉 open_interest_hist RAW TEST START")

    data = await agg.get_public_data(
        endpoint_name="open_interest_hist",
        symbol="BTCUSDT",
        period="1h",
        limit=5
    )

    print("👉 RESPONSE TYPE:", type(data))
    print("👉 RESPONSE:", data)
