# a_core.py
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

WATCHLIST = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "PEPEUSDT", "FETUSDT", "SUSDT", "ARPAUSDT"]

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

    """async def fetch_multi_market_data(self, symbols: List[str], interval="1h", limit=100):
        tasks = []
        for s in symbols:
            p = {"klines": {"symbol": s, "interval": interval, "limit": limit}}
            tasks.append(self.fetch_symbol_data(s, p))
        
        results = await asyncio.gather(*tasks)
        return pd.concat(results) if results else pd.DataFrame()
    """
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
    def __init__(self):
        self.fetcher = BinanceDataFetcher()
        self.market_engine = MarketContextEngine()
        self.resolver = get_default_resolver()
        self.cpu_executor = concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count())
        self.io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)

    async def get_alt_power(self, basket: List[str] = INDEX_BASKET):
        """Sadece AP sonucunu döner. Veriyi DB'den çeker."""
        from analysis.db_loader import load_latest_snapshots # Import burada veya üstte
        
        # 1. Gerekli tüm sembolleri (Sepet + BTC) belirle
        symbols_to_load = list(set(basket + ["BTCUSDT"]))
        
        # 2. DB'den son 50 snapshot'ı çek (funding ve OI burada var)
        # Bu fonksiyon senkron olduğu için bir executor içinde çalıştırmak daha sağlıklıdır
        loop = asyncio.get_running_loop()
        df_raw = await loop.run_in_executor(None, load_latest_snapshots, symbols_to_load, 50)
        
        if df_raw.empty:
            logger.warning("⚠️ Alt Power için DB'den veri gelmedi!")
            return {"alt_vs_btc_short": 50.0, "alt_short_term": 50.0, "coin_long_term": 50.0, "error": "Veri yok"}

        return self.market_engine.calculate_alt_power(df_raw, basket)
        
    async def run_full_analysis(self, symbols: Union[str, List[str]], metrics: List[str] = None):
        """Hem AP hem de teknik analiz sonuçlarını birleştirir."""
        symbols = [symbols] if isinstance(symbols, str) else symbols
        
        # 1. Market Context (AP) paralel çalışsın
        ap_task = asyncio.create_task(self.get_alt_power())
        
        # 2. Teknik Analizler
        analysis_tasks = [self._analyze_single(s, metrics or ["trend"], "1h", 100) for s in symbols]
        analyses = await asyncio.gather(*analysis_tasks)
        
        ap_results = await ap_task
        
        return {
            "market_context": ap_results,
            "results": {s: r for s, r in zip(symbols, analyses)}
        }
        
    async def _analyze_single(self, symbol: str, metrics: List[str], interval: str, limit: int):
        try:
            # 1. Resolve & Fetch
            score_map = resolve_scores_to_metrics(metrics, COMPOSITES, MACROS)
            flat_metrics = list({m for sub in score_map.values() for m in sub})
            metric_defs = self.resolver.resolve_multiple_definitions(flat_metrics)
            
            ep_params = {}
            for mdef in metric_defs.values():
                for ep, factory in mdef.get("endpoint_params", {}).items():
                    ep_params[ep] = factory(symbol, interval, limit)

            df = await self.fetcher.fetch_symbol_data(symbol, ep_params)
            if df.empty: return {"symbol": symbol, "error": "No data"}

            # 2. Calculate Raw Metrics
            calc_tasks = []
            for name, mdef in metric_defs.items():
                calc_tasks.append(self._calculate_metric(name, mdef, df))
            
            metric_results = dict(await asyncio.gather(*calc_tasks))

            # 3. Calculate Composites & Macros
            comp_scores = await self._calculate_formulas(metric_results, COMPOSITES)
            macro_scores = await self._calculate_formulas(comp_scores, MACROS)

            return {
                "symbol": symbol,
                "timestamp": datetime.utcnow().isoformat(),
                "scores": {**comp_scores, **macro_scores},
                "raw_metrics": metric_results
            }
        except Exception as e:
            logger.error(f"Pipeline error for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

    async def _calculate_metric(self, name, mdef, df):
        func = mdef.get("function")
        params = mdef.get("default_params", {})
        exec_type = mdef.get("execution_type", "sync")
        
        # Data preparation (Sadece gerekli kolonlar)
        prep_df = prepare_data(df, mdef)
        
        try:
            if exec_type == "async":
                raw = await func(prep_df, **params)
            else:
                loop = asyncio.get_running_loop()
                executor = self.cpu_executor if mdef.get("metadata", {}).get("category") == "advanced" else self.io_executor
                raw = await loop.run_in_executor(executor, partial(func, prep_df, **params))
            
            val = extract_final_value(raw, name)
            return name, float(np.clip(val, -1, 1)) if not math.isnan(val) else 0.0
        except:
            return name, 0.0

    async def _calculate_formulas(self, source, definitions):
        results = {}
        for name, info in definitions.items():
            formula = info.get("formula")
            results[name] = await FormulaEngine.evaluate(formula, source)
        return results


# ------------------------------------------------------------
# 6. 'ap' işlemlerini (Alt Power) yöneten motor.
class MarketContextEngine:
    """
    Eski koddaki 'ap' işlemlerini (Alt Power) yöneten motor.
    Sepet bazlı (Alt vs BTC, OI Trend vb.) analizleri yapar.
    API'den veri çekmez
    veritabanını okuyan db_loader.py modülünü kullan
    """
    
    @staticmethod
    def scale_0_100(value: float, min_val: float, max_val: float) -> float:
        """Sabit ölçekli normalizasyon - İşlev korundu."""
        if min_val == max_val: return 50.0
        norm = (value - min_val) / (max_val - min_val)
        return float(np.clip(norm * 100, 0, 100))

    def calculate_alt_power(self, df_raw: pd.DataFrame, INDEX_REPORT: List[str]) -> Dict[str, float]:
        """
        Paylaştığınız 'calculate_alt_power' fonksiyonunun modernize edilmiş, 
        hata toleransı artırılmış hali.
        """
        if df_raw.empty:
            return {"alt_vs_btc_short": 50.0, "alt_short_term": 50.0, "coin_long_term": 50.0}

        # --- 1. ZAMAN SENKRONİZASYONU ---
        df = df_raw.copy()
        # Saniyeleri dakikaya yuvarla (İşlev korundu)
        df['ts'] = (df['ts'] // 60) * 60 
        
        # Veriyi temizle ve grupla
        df_clean = df.groupby(['ts', 'symbol']).agg({
            'price': 'last',           # max yerine last daha güvenli fiyattır
            'open_interest': 'max',
            'funding_rate': 'last'
        }).reset_index().sort_values('ts')

        unique_ts = df_clean['ts'].unique()
        if len(unique_ts) < 2:
            return {"alt_vs_btc_short": 50.0, "alt_short_term": 50.0, "coin_long_term": 50.0, "status": "pending"}

        # --- 2. PIVOT TABLOLAR (Vektörel Hesaplama İçin) ---
        prices_pivot = df_clean.pivot(index="ts", columns="symbol", values="price").ffill()
        
        # --- 3. HESAPLAMALAR ---
        
        # A. Alt vs BTC (Short)
        v_btc = 50.0
        if "BTCUSDT" in prices_pivot.columns:
            returns = prices_pivot.pct_change(1).iloc[-1]
            btc_ret = returns["BTCUSDT"]
            available_alts = [s for s in INDEX_REPORT if s in returns.index]
            if available_alts:
                avg_alt_ret = returns[available_alts].mean()
                v_btc = self.scale_0_100(avg_alt_ret - btc_ret, -0.005, 0.005)

        # B. Alt Momentum (Short Term Strength)
        v_short = 50.0
        if len(prices_pivot) >= 5:
            returns_5 = prices_pivot.pct_change(5).iloc[-1]
            available_alts = [s for s in INDEX_REPORT if s in returns_5.index]
            if available_alts:
                v_short = self.scale_0_100(returns_5[available_alts].mean(), -0.02, 0.02)

        # C. Long Term Strength (OI & Funding)
        v_long = 50.0
        try:
            oi_pivot = df_clean.pivot(index="ts", columns="symbol", values="open_interest").ffill()
            available_alts = [s for s in INDEX_REPORT if s in oi_pivot.columns]
            
            if available_alts and len(oi_pivot) >= 2:
                # OI Değişimi
                oi_change = (oi_pivot[available_alts].iloc[-1] / oi_pivot[available_alts].iloc[0]) - 1
                oi_score = self.scale_0_100(oi_change.mean(), -0.05, 0.05)
                
                # Funding
                fund_avg = df_clean[df_clean['symbol'].isin(available_alts)].groupby('symbol')['funding_rate'].last().mean()
                fund_score = self.scale_0_100(fund_avg, 0.05, -0.01) # Ters skala
                
                v_long = (oi_score * 0.7) + (fund_score * 0.3)
        except Exception:
            pass

        return {
            "alt_vs_btc_short": round(float(v_btc), 1),
            "alt_short_term": round(float(v_short), 1),
            "coin_long_term": round(float(v_long), 1),
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

async def get_top_volume_symbols(count: int = 10) -> List[str]:
    return await _engine.fetcher.get_top_volume_symbols(count)
    
async def run_full_analysis(symbols, metrics=None):
    return await _engine.run_full_analysis(symbols, metrics)

if __name__ == "__main__":
    # Test kullanım
    async def test():
        res = await run_full_analysis("BTCUSDT", metrics=["trend"])
        print(res)
    asyncio.run(test())