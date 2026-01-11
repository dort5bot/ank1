# a_core.py
# python -m analysis.a_core

from __future__ import annotations
import asyncio
import logging
import math
import ast
import concurrent.futures
import os
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Set
from datetime import datetime
from functools import partial

import pandas as pd
import numpy as np

from analysis.metricresolver import get_default_resolver
from utils.binance_api.binance_a import BinanceAggregator

# ------------------------------------------------------------
# 1. LOGGING & CONSTANTS
# ------------------------------------------------------------
logger = logging.getLogger("analysis.core")
logging.basicConfig(level=logging.INFO)

INDEX_BASKET = [
    "ETHUSDT", "SOLUSDT", "BNBUSDT", "PEPEUSDT", "WIFUSDT", "DOGEUSDT",
    "FETUSDT", "NEARUSDT", "TAOUSDT", "SUIUSDT", "APTUSDT", "OPUSDT",
    "ARBUSDT", "LINKUSDT", "AVAXUSDT", "ONDOUSDT", "PENDLEUSDT", "XRPUSDT"
]

WATCHLIST = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "PEPEUSDT", "FETUSDT", "SUSDT", "ARPAUSDT"]

# Önemli: Collector'ın neyi toplayacağını bilmesi için birleştirilmiş liste
FULL_COLLECT_LIST = list(set(INDEX_BASKET + WATCHLIST + ["BTCUSDT"]))

# ------------------------------------------------------------
# 2. CONFIGURATION (COMPOSITES & MACROS)
# ------------------------------------------------------------
COMPOSITES = {
    
    # ✅ Başrılı-anlamlı
    # "trend","mom","vol",
    "trend": { # Piyasa long mu short mu
        "depends": ["ema", "macd", "rsi", "stochastic_oscillator"],
        "formula": "0.30*ema + 0.30*macd + 0.20*rsi + 0.20*stochastic_oscillator",
    },
    "mom": { # Hareket var mı: Trend başlar mı / devam eder
        "depends": ["roc", "adx", "atr"],
        "formula": "0.45*roc + 0.35*adx - 0.20*atr",
    },
    "vol": { # Piyasada Trend taşınır mı, chop mu
        "depends": ["historical_volatility", "garch_1_1", "hurst_exponent"],
        "formula": "0.40*historical_volatility + 0.35*garch_1_1 + 0.25*(1 - hurst_exponent)",
    },
    "vols": { #→ pozisyon & stop vol’un davranışı 
        "depends": ["historical_volatility", "vol_of_vol", "garch_1_1"],
        "formula": "0.4*historical_volatility + 0.35*vol_of_vol + 0.25*garch_1_1",
    },
    
    
    "sntp": { #→ move gerçek mi? sentiment’in güçlü hali: Trend parayla mı Fake mi 
    # DB / süreç ŞART
        "depends": ["oi_trend", "oi_growth_rate", "oi_price_correlation"],
        "formula": "0.4*oi_trend + 0.35*oi_growth_rate + 0.25*oi_price_correlation",
    },
    "strs": { #stress→ risk-off alarmı risk’in daha akıllısı: piyasa sıkışıyor mu: trade kapat, haber öncesi
    # DB / süreç ŞART
        "depends": ["funding_stress_risk","open_interest_shock_risk","spread_risk"],
        "formula": "0.4*funding_stress_risk + 0.35*open_interest_shock_risk + 0.25*spread_risk",
    },


    # ⚠️✅⚠️ Hesaplanabilir, yaklaşık bilgi verir, gerçek anlamlılık sınırlı.
    # "regim","entropy","risk"
    "sentiment": { # YARDIMCI Herkes aynı tarafta mı, Binance ile yaklaşık bilgi verir, çöp değil
        "depends": ["funding_rate", "funding_premium", "oi_trend"],
        "formula": "0.35*funding_rate + 0.25*funding_premium + 0.40*oi_trend",
    },
    "entropy": {# YARDIMCI. entropy_fractal
        "depends": ["entropy_index", "fractal_dimension_index_fdi", "hurst_exponent", "variance_ratio_test"],
        "formula": "0.35*entropy_index + 0.25*fractal_dimension_index_fdi - 0.25*hurst_exponent - 0.15*variance_ratio_test",
    },
    
    "risk": { # ⚠️ POZİSYON FİLTRESİ
        "depends": ["volatility_risk", "liquidity_depth_risk", "price_impact_risk"],
        "formula": "0.40*volatility_risk + 0.35*liquidity_depth_risk + 0.25*price_impact_risk",
    },
    "regim": { # ⚠️ ZAYIF
        "depends": ["advance_decline_line", "volume_leadership", "performance_dispersion"],
        "formula": "0.45*advance_decline_line + 0.35*volume_leadership + 0.20*performance_dispersion",
    },
    
    
    # ❌ BOŞA EMEK, SİL  ek veri yoksa dur) ⚠️ anlamsız metrikler, yetersiz veri nedeniyle
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
    },}	
# Binance API ile doğrudan elde edilemeyenler
# market_impact, depth_elasticity, taker_dominance_ratio (ham veriden türetilir ama direkt verilmez)
# ⚠️ garch_1_1, hurst_exponent, fdi, variance_ratio_test, fractal_dimension_index_fdi

MACROS = {	
    "core": { #→ karar: ANA METRİK: pusula: Trade bias / pozisyon yönü için ideal
        "depends": ["trend", "mom", "vol", "risk"],
        "formula": "0.35*trend + 0.25*mom + 0.25*vol - 0.15*risk",
    },
    "regf": { #→ strateji seçimi: regime_filter
        "depends": ["regim", "entropy", "hurst_exponent"],
        "formula": "0.45*regim + 0.35*entropy + 0.20*(1 - hurst_exponent)",
    },
    "cpxy": { #→ filtre: Piyasa karmaşık mı, Strateji seçimi
        "depends": ["entropy", "vol"],
        "formula": "0.6*entropy + 0.4*vol",
    },  

    
    # ⚠️ SİL -  sonraki  makrolar çöp uğraşma
    "coreliq": {
        "depends": ["trend", "vol", "regim", "risk", "liqu"],
        "formula": "0.27*trend + 0.20*vol + 0.20*regim + 0.18*risk + 0.15*liqu",
    },
    "alphax":{
        "depends": ["trend","mom","sentiment","flow","risk"],
        "formula": "0.35*trend + 0.25*mom + 0.20*sentiment + 0.15*flow - 0.15*risk", 
    },
    "sentri": { 
        "depends": ["sentiment", "entropy", "regim", "liquidity_density", "risk"],
        "formula": "0.24*sentiment + 0.18*entropy + 0.2*regim+ 0.2*liquidity_density - 0.18*risk",
    },
    "sentflow": {
        "depends": ["sentiment", "flow"],
        "formula": "0.55*sentiment + 0.45*flow",
    },
    "microstructure": {
        "depends": ["liqu", "liqrisk", "order"],
        "formula": "0.4*liqu+ 0.35*liqrisk + 0.25*order",
    },}

# ------------------------------------------------------------
# temizleyici ve ölçekleyici fonksiyonlar

class CoreAnalyzer:
    def __init__(self, resolver, config):
        self.resolver = resolver
        self.config = config

    def _fix_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """ESKİ ÇÖZÜM: Suffixleri temizler ve indikatörlerin hata almasını önler."""
        if df is None or df.empty: return df
        # Sütunlardaki __klines gibi ekleri temizle (BNB hatasının ana sebebi)
        df.columns = [c.split('__')[0] for c in df.columns]
        return df

    def _normalize_signal(self, name: str, value: float, df: pd.DataFrame) -> float:
        """ESKİ ÇÖZÜM: BNB fiyatı 600$ olsa bile skoru -1 ile 1 arasına çeker."""
        try:
            price = df['close'].iloc[-1]
            # Fiyat bazlıları oranla (İlkel kalmasını engelleyen kısım burasıydı)
            if name in ['ema', 'macd', 'bollinger']:
                return np.tanh((price - value) / value * 10)
            # RSI/Stoch gibi 0-100 arası olanları merkeze çek
            if 0 <= value <= 100:
                return (value - 50) / 50
            return np.clip(value, -1, 1)
        except:
            return 0.0

    async def analyze_symbol(self, symbol: str, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Senin ana mantığına sadık kalınan analiz süreci."""
        try:
            # 1. Adım: Veriyi tamir et (Eski yöntem)
            klines = self._fix_dataframe(data.get('klines'))
            
            # 2. Adım: Metrikleri tek tek hesapla ve anında normalize et
            metrics = {}
            defs = self.resolver.get_all_definitions()
            
            for name, func in defs.items():
                try:
                    raw_val = func(klines)
                    # Buradaki normalizasyon BNB'nin kilitlenmesini çözer
                    metrics[name] = self._normalize_signal(name, raw_val, klines)
                except Exception as e:
                    logger.warning(f"{symbol} - {name} hesaplanamadı: {e}")
                    metrics[name] = 0.0

            # 3. Adım: Trend skoru (Kompozit hesaplama)
            # Senin 1 aydır kullandığın ağırlıkları buraya gir:
            trend = (
                0.30 * metrics.get('ema', 0) + 
                0.30 * metrics.get('macd', 0) + 
                0.20 * metrics.get('rsi', 0) + 
                0.20 * metrics.get('stochastic_oscillator', 0)
            )

            return {
                "symbol": symbol,
                "trend": np.clip(trend, -1, 1),
                "metrics": metrics,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Kritik hata: {symbol} analiz edilemedi: {e}")
            return {"symbol": symbol, "status": "error"}
            

# ------------------------------------------------------------
# 3. UTILITIES & FORMULA ENGINE
# ------------------------------------------------------------
class FormulaEngine:
    _ALLOWED_NODES = {
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Num, ast.Load, ast.Add,
        ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Name,
        ast.Constant, ast.Mod, ast.FloorDiv, ast.Call
    }
    _cache: Dict[str, Any] = {}
    _lock = asyncio.Lock()

    @classmethod
    async def evaluate(cls, formula: str, context: Dict[str, float]) -> float:
        code = await cls._get_compiled(formula)
        if not code: return 0.0
        safe_ctx = {k: (v if not math.isnan(v) else 0.0) for k, v in context.items()}
        try:
            return float(eval(code, {"__builtins__": {}}, safe_ctx))
        except: return 0.0

    @classmethod
    async def _get_compiled(cls, formula: str):
        if formula in cls._cache: return cls._cache[formula]
        async with cls._lock:
            try:
                tree = ast.parse(formula, mode="eval")
                for n in ast.walk(tree):
                    if type(n) not in cls._ALLOWED_NODES: raise ValueError("Unsafe formula")
                cls._cache[formula] = compile(tree, "<formula>", "eval")
                return cls._cache[formula]
            except: return None

# ------------------------------------------------------------
# 4. DATA FETCHING & PROCESSING
# ------------------------------------------------------------
class BinanceDataFetcher:
    def __init__(self):
        self.aggregator = None
        # Filtreleme kriterlerini sınıf düzeyinde tutmak yönetimi kolaylaştırır
        self.excluded_keywords = ["UP", "DOWN", "BULL", "BEAR"]
        self.stable_coins = ["USDC", "FDUSD", "TUSD", "DAI", "USDP", "EUR", "PAXG"]

    async def get_aggregator(self):
        if not self.aggregator:
            self.aggregator = await BinanceAggregator.get_instance()
        return self.aggregator

    async def get_top_volume_symbols(self, count: int = 10) -> List[str]:
        """
        Market genelindeki hacimli sembolleri filtreleyerek getirir.
        """
        try:
            agg = await self.get_aggregator()
            all_tickers = await agg.get_public_data(endpoint_name="ticker_24hr")
            
            if not all_tickers:
                return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

            valid_pairs = []
            for ticker in all_tickers:
                symbol = ticker.get('symbol', '')
                # Filtreleme mantığı
                if (symbol.endswith('USDT') and 
                    not any(k in symbol for k in self.excluded_keywords) and
                    not any(s in symbol for s in self.stable_coins)):
                    valid_pairs.append(ticker)

            # Hacme göre sırala (quoteVolume: USDT bazlı hacim)
            sorted_pairs = sorted(
                valid_pairs, 
                key=lambda x: float(x.get('quoteVolume', 0)), 
                reverse=True
            )

            return [t['symbol'] for t in sorted_pairs[:min(count, 40)]]

        except Exception as e:
            logger.error(f"❌ get_top_volume_symbols hatası: {e}")
            return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    async def fetch_symbol_data(self, symbol: str, endpoint_params: Dict) -> pd.DataFrame:
        agg = await self.get_aggregator()
        
        async def _fetch(ep, params):
            try:
                raw = await agg.get_public_data(endpoint_name=ep, **params)
                if ep == "klines": return self._klines_to_df(raw, symbol)
                return self._generic_to_df(raw, symbol, ep)
            except Exception as e:
                logger.error(f"Fetch error {symbol} {ep}: {e}")
                return pd.DataFrame()

        tasks = [_fetch(ep, p) for ep, p in endpoint_params.items()]
        dfs = await asyncio.gather(*tasks)
        
        merged_df = pd.DataFrame()
        for df in dfs:
            if df.empty: continue
            merged_df = df if merged_df.empty else merged_df.combine_first(df)
        return merged_df

    # a_core.py içindeki BinanceDataFetcher sınıfının metodunu bu şekilde güncelleyin:

    async def fetch_multi_market_data(self, symbols: List[str], interval="1h", limit=100):
        """
        Rate limit (IP engeli) yememek için sembolleri 
        küçük gecikmelerle (throttle) çeker.
        """
        results = []
        
        for symbol in symbols:
            try:
                # Her sembol için klines parametrelerini hazırla
                p = {"klines": {"symbol": symbol, "interval": interval, "limit": limit}}
                
                # Tek bir sembolün verisini çek
                df = await self.fetch_symbol_data(symbol, p)
                
                if not df.empty:
                    results.append(df)
                
                # 🔥 KRİTİK DÜZELTME: Her istekten sonra 200ms bekle.
                # Bu sayede saniyede en fazla 5 istek gider (Binance limiti saniyede 10-50 arasıdır).
                # Botun kitlenmesini önleyen ana parça burasıdır.
                await asyncio.sleep(0.2)
                
            except Exception as e:
                logger.error(f"⚠️ {symbol} verisi çekilirken hata oluştu: {e}")
                continue

        # Sonuçları birleştir
        if results:
            return pd.concat(results)
        else:
            logger.warning("❌ Hiçbir sembolden veri çekilemedi!")
            return pd.DataFrame()
            


    def _klines_to_df(self, klines, symbol):
        df = pd.DataFrame(klines, columns=['ts','o','h','l','c','v','ct','qv','tr','tbb','tbq','i'])
        df['ts'] = df['ts'] // 1000 
        df['timestamp'] = pd.to_datetime(df['ts'], unit='s', utc=True)
        df.set_index('timestamp', inplace=True)
        cols = {'o':'open', 'h':'high', 'l':'low', 'c':'close', 'v':'volume'}
        return df.rename(columns=cols).apply(pd.to_numeric, errors='coerce').assign(symbol=symbol)

    def _generic_to_df(self, data, symbol, ep):
        df = pd.DataFrame(data)
        if 'timestamp' in df.columns:
            df['ts'] = df['timestamp'] // 1000
            df['timestamp'] = pd.to_datetime(df['ts'], unit='s', utc=True)
            df.set_index('timestamp', inplace=True)
        return df.add_suffix(f"__{ep}").assign(symbol=symbol)
     
# ------------------------------------------------------------
# 5. CORE ANALYSIS ENGINE - ana işlemci, 
# diğerleri class olarak yazılır, buraya eklenir
# ------------------------------------------------------------
class CoreAnalysisEngine:
    """
    Core analiz motoru.
    - Metric resolve
    - Data fetch
    - Raw metric hesaplama
    - Normalizasyon
    - Composite & Macro skorlar
    """

    # =========================================================
    # 1️⃣ INIT / LIFECYCLE
    # =========================================================
    def __init__(self):
        self.fetcher = BinanceDataFetcher()
        self.market_engine = MarketContextEngine()
        self.resolver = get_default_resolver()

        self.cpu_executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=os.cpu_count()
        )
        self.io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=20
        )

    # Kaynakları düzgün kapat.
    """async def close(self):
        if self.fetcher:
            await self.fetcher.close()

        self.cpu_executor.shutdown(wait=False)
        self.io_executor.shutdown(wait=False)
        
        self.cpu_executor.shutdown(wait=False, cancel_futures=True)
        self.io_executor.shutdown(wait=False, cancel_futures=True)
    """

    async def close(self):
        """Graceful shutdown for CoreAnalysisEngine"""
        # Fetcher
        try:
            if self.fetcher:
                await self.fetcher.close()
                self.fetcher = None
        except Exception as e:
            logger.warning(f"Fetcher close failed: {e}")

        # CPU executor
        try:
            if self.cpu_executor:
                self.cpu_executor.shutdown(
                    wait=False,
                    cancel_futures=True
                )
                self.cpu_executor = None
        except Exception as e:
            logger.warning(f"CPU executor shutdown failed: {e}")

        # IO executor
        try:
            if self.io_executor:
                self.io_executor.shutdown(
                    wait=False,
                    cancel_futures=True
                )
                self.io_executor = None
        except Exception as e:
            logger.warning(f"IO executor shutdown failed: {e}")

        
  

    # =========================================================
    # 2️⃣ STATIC HELPERS (STATE YOK)
    # =========================================================
    @staticmethod
    def normalize_metric_value(
        name: str,
        value: float,
        df: pd.DataFrame
    ) -> float:
        """Metric tipine göre normalize et."""
        try:
            if name in ("ema", "macd", "sma"):
                if "close" in df.columns and not df.empty and value:
                    price = df["close"].iloc[-1]
                    return float(np.tanh((price - value) / value * 10))

            elif name in ("rsi", "stochastic_oscillator", "adx"):
                return float((value - 50) / 50)

            elif name in ("historical_volatility", "atr", "bollinger_width"):
                return float(np.clip(value / 0.1, -1, 1))

            elif name in ("funding_rate", "funding_premium"):
                return float(np.clip(value * 100, -1, 1))

            return float(np.clip(value, -1, 1))
        except Exception:
            return 0.0

    @staticmethod
    def prepare_data(df: pd.DataFrame, mdef: Dict) -> pd.DataFrame:
        """Metric için gerekli kolonları hazırla."""
        required_cols = mdef.get("required_columns", [])
        if not required_cols:
            return df

        prepared = {}
        for col in required_cols:
            if col in df.columns:
                prepared[col] = df[col]
            else:
                matches = [c for c in df.columns if c.startswith(f"{col}__")]
                if matches:
                    prepared[col] = df[matches[0]]

        return pd.DataFrame(prepared) if prepared else df

    # =========================================================
    # 3️⃣ MARKET CONTEXT (AP)
    # =========================================================
    async def get_alt_power(self, basket: List[str] = INDEX_BASKET):
        """Alt Power (AP) hesapla."""
        from analysis.db_loader import load_latest_snapshots

        symbols = list(set(basket + ["BTCUSDT"]))
        loop = asyncio.get_running_loop()

        df = await loop.run_in_executor(
            None, load_latest_snapshots, symbols, 50
        )

        if df.empty:
            return {
                "alt_vs_btc_short": 50.0,
                "alt_short_term": 50.0,
                "coin_long_term": 50.0,
                "error": "Veri yok"
            }

        return self.market_engine.calculate_alt_power(df, basket)

    # =========================================================
    # 4️⃣ ANA PUBLIC API
    # =========================================================

    async def run_full_analysis(
        self,
        symbols,
        metrics=None,
        interval="1h",
        limit=100
    ):
        """
        TEK ve GERÇEK analiz yolu.
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        results = {}

        ap_task = asyncio.create_task(self.get_alt_power())

        for symbol in symbols:
            results[symbol] = await self._analyze_symbol(
                symbol, metrics, interval, limit
            )

        market_context = await ap_task

        return {
            "market_context": market_context,
            "results": results
        }



    # =========================================================
    # 5️⃣ INTERNAL PIPELINE
    # =========================================================
    async def _analyze_symbol(
        self,
        symbol: str,
        metrics,
        interval: str,
        limit: int
    ) -> Dict[str, Any]:
        try:
            score_map = resolve_scores_to_metrics(metrics, COMPOSITES, MACROS)
            flat_metrics = list({m for v in score_map.values() for m in v})

            if not flat_metrics:
                flat_metrics = ["ema", "macd", "rsi", "stochastic_oscillator"]

            metric_defs = self.resolver.resolve_multiple_definitions(flat_metrics)

            endpoint_params = {}
            for mdef in metric_defs.values():
                for ep, factory in mdef.get("endpoint_params", {}).items():
                    endpoint_params[ep] = factory(symbol, interval, limit)

            if not endpoint_params:
                endpoint_params = {
                    "klines": {"symbol": symbol, "interval": interval, "limit": limit}
                }

            df = await self.fetcher.fetch_symbol_data(symbol, endpoint_params)
            if df.empty:
                return {"symbol": symbol, "status": "failed", "error": "No data"}

            # ---- RAW METRICS ----
            raw_metrics = {}
            for name, mdef in metric_defs.items():
                try:
                    func = mdef["function"]
                    prep_df = self.prepare_data(df, mdef)

                    raw = func(prep_df, **mdef.get("default_params", {}))
                    val = extract_final_value(raw, name)

                    raw_metrics[name] = self.normalize_metric_value(
                        name, val, prep_df
                    )
                except Exception:
                    raw_metrics[name] = 0.0

            # ---- COMPOSITES ----
            composites = {}
            for comp in metrics or []:
                if comp in COMPOSITES:
                    composites[comp] = await FormulaEngine.evaluate(
                        COMPOSITES[comp]["formula"],
                        raw_metrics
                    )

            # ---- MACROS ----
            macros = {}
            context = {**raw_metrics, **composites}
            for name, info in MACROS.items():
                macros[name] = await FormulaEngine.evaluate(
                    info["formula"], context
                )

            return {
                "symbol": symbol,
                "status": "success",
                "scores": {**composites, **macros},
                "raw_metrics": raw_metrics,
                "timestamp": datetime.utcnow().isoformat(),
                "endpoints_used": list(endpoint_params.keys())
            }

        except Exception as e:
            return {"symbol": symbol, "status": "failed", "error": str(e)}



# ------------------------------------------------------------
# 6. 'ap' işlemlerini (Alt Power) yöneten motor.
    """
    Eski koddaki 'ap' işlemlerini (Alt Power) yöneten motor.
    Sepet bazlı (Alt vs BTC, OI Trend vb.) analizleri yapar.
    API'den veri çekmez
    veritabanını okuyan db_loader.py modülünü kullan
    
    Metrik	Yeni Öneri	Neden?
    Hacim (Volume)	Hacim ağırlıklı ortalama.	Shitcoinlerin spekülatif hareketleri endeksi bozmaz.
    Zaman Ağırlığı	5dk, 1sa ve 4sa ağırlıklı.	Kısa süreli "fake" fitilleri eler, trendi yakalar.
    Normalizasyon	Daraltılmış/Dinamik aralık.	Mikabot gibi hassas sonuçlar için -0.005 yerine -0.003 gibi daha dar eşikler tepkiyi artırır.
    OI Analizi	Son 1 saatlik ivme.	Akümülasyonu ve ani para girişini daha iyi ölçer.
    
    Piyasayı çok daha hassas okuyan, volatiliteye duyarlı (ATR tabanlı) ve hacim ağırlıklı kod
    - Durgun Piyasada "Yalancı" Sinyalleri Önler:
    - Sert Piyasada Skorun "Kör" Olmasını Engeller:
    - Mikabot Benzeri "Gerçekçi" Sonuç: Mikabot gibi profesyonel araçlar 
    "göreceli" güç ölçer. Yani bir veriyi sadece rakam olarak değil, "son 24 saatin normaline göre ne kadar saptığına" bakarak puanlar.
    - Ters Funding Skalası: Funding yükseldikçe (longlar maliyetlendikçe) yapısal riski artırıp skoru hafifçe aşağı çektik (daha gerçekçi bir piyasa baskısı okuması).
    
    """

# 6. 'ap' işlemlerini (Alt Power) yöneten motor.
class MarketContextEngine:
    
    @staticmethod
    def scale_0_100(value: float, min_val: float, max_val: float) -> float:
        """Dinamik veya sabit aralıklarla normalizasyon yapar."""
        if min_val == max_val: return 50.0
        # Değerlerin yönünü koruyarak (ters skala gerekirse min > max olabilir)
        norm = (value - min_val) / (max_val - min_val)
        return float(np.clip(norm * 100, 0, 100))

    def calculate_alt_power(self, df_raw: pd.DataFrame, INDEX_REPORT: List[str]) -> Dict[str, float]:
        """
        Gelişmiş AP Hesaplama: 
        1. Hacim Ağırlıklı (VWAP Mantığı)
        2. Volatiliteye Duyarlı (ATR Bazlı Ölçekleme)
        3. Çoklu Zaman Dilimi Ağırlıklandırması
        """
        if df_raw.empty:
            return {"alt_vs_btc_short": 50.0, "alt_short_term": 50.0, "coin_long_term": 50.0}

        # --- 1. VERİ TEMİZLEME & HAZIRLIK ---
        df = df_raw.copy()
        df_clean = df.groupby(['ts', 'symbol']).agg({
            'price': 'last',
            'open_interest': 'last',
            'funding_rate': 'last',
            'volume': 'sum'
        }).reset_index().sort_values('ts')

        # Pivot Tablolar
        prices = df_clean.pivot(index="ts", columns="symbol", values="price").ffill()
        volumes = df_clean.pivot(index="ts", columns="symbol", values="volume").ffill()
        
        available_alts = [s for s in INDEX_REPORT if s in prices.columns]
        if not available_alts or len(prices) < 12: # En az 1 saatlik veri (5dk bar ile)
            return {"alt_vs_btc_short": 50.0, "alt_short_term": 50.0, "coin_long_term": 50.0}

        # --- 2. PİYASA OYNAKLIĞI (ATR) HESABI ---
        # Son 24 barın (2 saat) standart sapmasını volatilite göstergesi olarak kullanalım
        # Bu, skorumuzun 'yapışmasını' engelleyen dinamik filtredir.
        market_volatility = prices[available_alts].pct_change().std().mean() 
        # Eğer piyasa çok ölüyse minimum bir eşik belirleyelim (Örn: %0.1)
        dynamic_threshold = max(market_volatility, 0.001)

        # --- 3. ALT vs BTC (Hacim Ağırlıklı) ---
        btc_ret = prices["BTCUSDT"].pct_change(1).iloc[-1] if "BTCUSDT" in prices.columns else 0
        alt_rets = prices[available_alts].pct_change(1).iloc[-1]
        alt_vols = volumes[available_alts].iloc[-1]
        
        # Büyük altcoinlerin etkisi daha yüksek
        weighted_alt_ret = (alt_rets * alt_vols).sum() / (alt_vols.sum() + 1e-9)
        
        # Dinamik ölçek: Volatilitenin 0.5 katı fark normal karşılanır
        v_btc = self.scale_0_100(weighted_alt_ret - btc_ret, -dynamic_threshold * 0.5, dynamic_threshold * 0.5)

        # --- 4. ALT MOMENTUM (Zaman Ağırlıklı) ---
        # Kısa (5dk), Orta (1sa), Uzun (4sa) değişimlerin ortalaması
        ret_5m = prices[available_alts].pct_change(1).iloc[-1].mean()
        ret_1h = prices[available_alts].pct_change(12).iloc[-1].mean() if len(prices) >= 12 else ret_5m
        ret_4h = prices[available_alts].pct_change(48).iloc[-1].mean() if len(prices) >= 48 else ret_1h
        
        # Öneri: %50 kısa, %30 orta, %20 uzun
        combined_mom = (ret_5m * 0.5) + (ret_1h * 0.3) + (ret_4h * 0.2)
        
        # Dinamik ölçek: Momentum eşiği volatiliteye göre esner
        v_short = self.scale_0_100(combined_mom, -dynamic_threshold * 2, dynamic_threshold * 2)

        # --- 5. YAPISAL GÜÇ (OI & Funding) ---
        try:
            oi_pivot = df_clean.pivot(index="ts", columns="symbol", values="open_interest").ffill()
            # OI İvmesi (Son 1 saatlik değişim)
            oi_change = (oi_pivot[available_alts].iloc[-1] / oi_pivot[available_alts].iloc[-12]) - 1
            
            # OI için sabit eşik daha sağlıklıdır çünkü OI fiyattan bağımsız şişebilir
            oi_score = self.scale_0_100(oi_change.mean(), -0.05, 0.05)
            
            # Funding (Düşük/Negatif funding yükseliş için daha sağlıklı 'duvar' sağlar)
            fund_avg = df_clean[df_clean['symbol'].isin(available_alts)].groupby('symbol')['funding_rate'].last().mean()
            # 0.01 pozitif funding (pahalı), -0.01 negatif funding (ucuz)
            fund_score = self.scale_0_100(fund_avg, 0.015, -0.005) 
            
            v_long = (oi_score * 0.7) + (fund_score * 0.3)
        except:
            v_long = 50.0

        return {
            "alt_vs_btc_short": round(float(v_btc), 1),
            "alt_short_term": round(float(v_short), 1),
            "coin_long_term": round(float(v_long), 1)
        }
        



# ------------------------------------------------------------
# 7. Handler burayı çağıracak
async def process_pipeline(
    target: Any, 
    cmd: str = "/t", 
    metrics: List[str] = None
) -> List[Dict]:
    """
    Handler'ın beklediği eski arayüzü, yeni modern motora bağlar.
    """
    # 1. symbols listesini belirle
    symbols = []
    if target is None:
        symbols = WATCHLIST
    elif target == "INDEX_BASKET":
        symbols = INDEX_BASKET
    elif isinstance(target, int):
        
        # HACİMLİ N COİN BİLGİSİ
        # Sınıfı çağırıyoruz ve içindeki metodu kullanıyoruz
        symbols = await _engine.fetcher.get_top_volume_symbols(target)
        # -----------------

    else:
        if isinstance(target, str):
            symbols = [target if target.endswith("USDT") else f"{target}USDT"]
        else:
            symbols = [s if s.endswith("USDT") else f"{s}USDT" for s in target]


    # 2. Hangi metrikleri hesaplayacağız?
    # metrics parametresi handler'dan geliyor, onu kullan!
    if metrics is None or not metrics:
        # Eğer handler metrik göndermedi, default değer kullan
        metrics = ["core", "regf", "vols"]
    
    # 3. Motoru Çalıştır - SADECE İSTENEN METRİKLERİ HESAPLA
    raw_results = await _engine.run_full_analysis(symbols, metrics=metrics)

    # 4. Sonuçları formatla - BASİT VERSİYON
    results = []
    
    # Motorun dönüş formatını anla
    if isinstance(raw_results, dict) and "symbol" in raw_results:
        # Tek sembol dönüşü: {symbol: "...", scores: {...}}
        processed = {raw_results["symbol"]: raw_results}
    else:
        # Çoklu sembol dönüşü: {symbol1: {...}, symbol2: {...}}
        processed = raw_results

    for symbol in symbols:
        data = processed.get(symbol, {})
        
        if "error" in data or not data:
            results.append({"symbol": symbol, "error": "Analiz başarısız"})
            continue
            
        scores = data.get("scores", {})
        
        # Handler'ın istediği her metrik için değer al
        result_item = {"symbol": symbol}
        for metric in metrics:
            # Core'daki scores dict'inden doğrudan al
            # İsimler aynı olduğu için eşleme gerekmiyor!
            result_item[metric] = scores.get(metric, 0.0)
        
        results.append(result_item)
        
    return results
    
 
 
# ------------------------------------------------------------
# 8. HELPERS (Globalized for Pickleability)
# ------------------------------------------------------------
def prepare_data(df, mdef):
    req_cols = mdef.get("required_columns", [])
    if not req_cols: return df
    # Basit eşleştirme: close -> close__klines veya close
    available = {}
    for c in req_cols:
        if c in df.columns: available[c] = df[c]
        else:
            match = [col for col in df.columns if col.startswith(f"{c}__")]
            if match: available[c] = df[match[0]]
    return pd.DataFrame(available)

def extract_final_value(raw, name):
    if isinstance(raw, (int, float, np.number)): return float(raw)
    if isinstance(raw, pd.Series): return float(raw.iloc[-1]) if not raw.empty else float('nan')
    if isinstance(raw, dict): return float(raw.get('value', raw.get(name, 0)))
    return 0.0

def resolve_scores_to_metrics(metrics, comp_map, macro_map):
    out = {}
    for m in metrics:
        if m in comp_map: out[m] = comp_map[m]["depends"]
        elif m in macro_map:
            deps = []
            for d in macro_map[m]["depends"]:
                deps.extend(comp_map.get(d, {}).get("depends", [d]))
            out[m] = list(set(deps))
        else: out[m] = [m]
    return out

# ------------------------------------------------------------
# EXPORTED INTERFACE (Dışarıya açılan kapı)
# ------------------------------------------------------------
_engine = CoreAnalysisEngine()

# ---Dışarıdan kolay erişim için sarmalayıcı (wrapper) metod ---
async def get_alt_power():
    return await _engine.get_alt_power()

# async def get_top_volume_symbols(count: int = 10) -> List[str]:
    # return await _engine.fetcher.get_top_volume_symbols(count)


async def get_top_volume_symbols(count: int = 10):
    fetcher = BinanceDataFetcher()
    return await fetcher.get_top_volume_symbols(count)

   
async def run_full_analysis(
    symbols,
    metrics=None,
    interval="1h",
    limit=100,
):
    return await _engine.run_full_analysis(
        symbols, metrics, interval, limit
    )

# a_core.py'nin son kısmında (export edilen fonksiyonların yanına):
WATCHLIST = WATCHLIST  # Zaten tanımlı, sadece export ediyoruz
INDEX_BASKET = INDEX_BASKET
FULL_COLLECT_LIST = FULL_COLLECT_LIST

   
if __name__ == "__main__":
    async def test_top_volume_analysis():
        """Hacimli coinleri analiz eden test
        # Varsayılan (5 coin)
        python -m analysis.a_core

        # Hızlı test (sadece BTC)
        python -m analysis.a_core quick
                
                
        """
        print("Hacimli Coin Analizi Testi\n" + "="*50)
        
        # 1. Hacimli coinleri al
        count = 7  # İstediğin sayı
        top_symbols = await get_top_volume_symbols(count)
        print(f"En hacimli {count} coin: {top_symbols}")
        print()
        
        # 2. Bu coinleri analiz et
        metrics = ["trend", "mom", "vol", "risk", "core"]
        # metrics = ["core"]
        print(f"Analiz edilecek metrikler: {metrics}")
        print()
        
        # 3. Analizi çalıştır
        results = await run_full_analysis(
            symbols=top_symbols,
            metrics=metrics,
            interval="1h",
            limit=100
        )
        
        # 4. Sonuçları tablo şeklinde göster
        print("\n📊 ANALİZ SONUÇLARI")
        print("-" * 80)
        print(f"{'Sembol':<12} {'Trend':<8} {'Mom':<8} {'Vol':<8} {'Risk':<8} {'Core':<8}")
        print("-" * 80)
        
        for symbol in top_symbols:
            symbol_data = results.get("results", {}).get(symbol, {})
            scores = symbol_data.get("scores", {})
            
            print(f"{symbol:<12} "
                  f"{scores.get('trend', 0):<8.3f} "
                  f"{scores.get('mom', 0):<8.3f} "
                  f"{scores.get('vol', 0):<8.3f} "
                  f"{scores.get('risk', 0):<8.3f} "
                  f"{scores.get('core', 0):<8.3f}")
        
        # 5. Market context'i göster
        print("\n🌐 MARKET DURUMU")
        print("-" * 80)
        market_context = results.get("market_context", {})
        for key, value in market_context.items():
            if isinstance(value, (int, float)):
                print(f"{key:<25}: {value:.2f}")
            else:
                print(f"{key:<25}: {value}")
        
        # 6. En yüksek core skorlu coin
        print("\n🏆 EN İYİ PERFORMANS")
        best_symbol = None
        best_score = -1
        
        for symbol in top_symbols:
            symbol_data = results.get("results", {}).get(symbol, {})
            core_score = symbol_data.get("scores", {}).get("core", -1)
            if core_score > best_score:
                best_score = core_score
                best_symbol = symbol
        
        if best_symbol:
            print(f"En yüksek core skoru: {best_symbol} ({best_score:.3f})")
    
    async def quick_test():
        """Hızlı test - tek coin"""
        print("Hızlı Test: BTCUSDT")
        result = await run_full_analysis("BTCUSDT", metrics=["trend", "core"])
        print(result)
    
    # Testleri çalıştır
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "quick":
            asyncio.run(quick_test())
        elif sys.argv[1] == "top":
            count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            asyncio.run(test_top_volume_analysis())
    else:
        # Varsayılan: kapsamlı test
        asyncio.run(test_top_volume_analysis())