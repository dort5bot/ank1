# analysis/corr_lead.py
"""
Korelasyon & Lead-Lag (Liderlik Analizi) Modülü
Modül: corr_lead.py
Config: c_corr.py
Endpoint: /api/v3/klines (multi-symbol), /fapi/v1/markPriceKlines, /api/v3/ticker/price
API Türü: Spot + Futures Public

Metrikler:
- Klasik: Pearson Corr, Beta, Rolling Covariance, Partial Correlation, Rolling Lead-Lag Delta
- Profesyonel: Granger Causality Test, Dynamic Time Warping (DTW), Canonical Correlation, Vector AutoReg (VAR)

Amaç: Coin'ler arası liderlik & yön takibi
Çıktı: Correlation Lead-Lag Matrix
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import asyncio
from scipy import stats
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import warnings
warnings.filterwarnings('ignore')

from analysis.analysis_base_module import BaseAnalysisModule
from utils.binance_api.binance_a import BinanceAggregator
#from analysis.config.c_corr import CONFIG


class CorrelationLeadLagModule(BaseAnalysisModule):
    """Korelasyon ve Lead-Lag Analiz Modülü"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.module_name = "correlation_lead_lag"
        self.version = "1.0.0"
        self.binance = BinanceAggregator()
        
        # Metrik ağırlıkları
        self.weights = {
            "pearson_corr": 0.15,
            "beta": 0.15,
            "rolling_cov": 0.10,
            "partial_corr": 0.10,
            "lead_lag_delta": 0.20,
            "granger_causality": 0.15,
            "dtw_distance": 0.10,
            "var_impulse": 0.05
        }
        
        # Threshold değerleri
        self.thresholds = {
            "high_correlation": 0.7,
            "medium_correlation": 0.3,
            "significant_lead": 0.1,
            "strong_causality": 0.05
        }
        
    async def initialize(self):
        """Modül başlatma"""
        await super().initialize()
        self.logger.info(f"{self.module_name} modülü başlatıldı")
    
    async def fetch_price_data(self, symbols: List[str], interval: str = "1h", limit: int = 100) -> Dict[str, pd.DataFrame]:
        """Çoklu sembol için fiyat verilerini getir"""
        try:
            tasks = []
            for symbol in symbols:
                # Spot ve futures verilerini paralel al
                spot_task = self.binance.get_klines(
                    symbol=symbol, 
                    interval=interval, 
                    limit=limit
                )
                futures_task = self.binance.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
                tasks.extend([spot_task, futures_task])
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            price_data = {}
            for i, symbol in enumerate(symbols):
                spot_data = results[i*2]
                futures_data = results[i*2 + 1]
                
                if not isinstance(spot_data, Exception) and len(spot_data) > 0:
                    closes = [float(x[4]) for x in spot_data]  # Close price
                    price_data[f"{symbol}_spot"] = pd.Series(closes)
                
                if not isinstance(futures_data, Exception) and len(futures_data) > 0:
                    closes = [float(x[4]) for x in futures_data]
                    price_data[f"{symbol}_futures"] = pd.Series(closes)
            
            return price_data
            
        except Exception as e:
            self.logger.error(f"Veri çekme hatası: {e}")
            return {}
    
    def calculate_pearson_correlation(self, series1: pd.Series, series2: pd.Series) -> float:
        """Pearson korelasyon katsayısı"""
        if len(series1) < 2 or len(series2) < 2:
            return 0.0
        
        # NaN değerleri temizle ve ortak indekse göre hizala
        df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        if len(df) < 2:
            return 0.0
            
        corr, _ = pearsonr(df['s1'], df['s2'])
        return corr if not np.isnan(corr) else 0.0
    
    def calculate_beta(self, series1: pd.Series, series2: pd.Series) -> float:
        """Beta katsayısı (seri2'nin seri1'e göre beta'sı)"""
        if len(series1) < 2 or len(series2) < 2:
            return 0.0
            
        df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        if len(df) < 2:
            return 0.0
            
        # Getiri hesapla
        returns1 = df['s1'].pct_change().dropna()
        returns2 = df['s2'].pct_change().dropna()
        
        if len(returns1) < 2 or len(returns2) < 2:
            return 0.0
            
        # Kovaryans ve varyans
        covariance = np.cov(returns1, returns2)[0, 1]
        variance = np.var(returns1)
        
        return covariance / variance if variance != 0 else 0.0
    
    def calculate_rolling_covariance(self, series1: pd.Series, series2: pd.Series, window: int = 20) -> float:
        """Rolling kovaryans ortalaması"""
        if len(series1) < window or len(series2) < window:
            return 0.0
            
        df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        if len(df) < window:
            return 0.0
            
        rolling_cov = df['s1'].rolling(window=window).cov(df['s2']).dropna()
        return rolling_cov.mean() if len(rolling_cov) > 0 else 0.0
    
    def calculate_partial_correlation(self, series1: pd.Series, series2: pd.Series, 
                                    control_series: pd.Series = None) -> float:
        """Kısmi korelasyon (control_series kontrol değişkeni)"""
        if len(series1) < 3 or len(series2) < 3:
            return 0.0
            
        df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        if len(df) < 3:
            return 0.0
            
        if control_series is not None:
            # Kontrol değişkeni ile kısmi korelasyon
            control_aligned = control_series.reindex(df.index).dropna()
            if len(control_aligned) < 3:
                return self.calculate_pearson_correlation(series1, series2)
            
            df_control = pd.DataFrame({
                's1': df['s1'],
                's2': df['s2'],
                'control': control_aligned
            }).dropna()
            
            if len(df_control) < 3:
                return self.calculate_pearson_correlation(series1, series2)
                
            # Residual-based partial correlation
            from sklearn.linear_model import LinearRegression
            
            # s1 ~ control
            X_control = df_control[['control']].values.reshape(-1, 1)
            y_s1 = df_control['s1'].values
            model_s1 = LinearRegression().fit(X_control, y_s1)
            resid_s1 = y_s1 - model_s1.predict(X_control)
            
            # s2 ~ control  
            y_s2 = df_control['s2'].values
            model_s2 = LinearRegression().fit(X_control, y_s2)
            resid_s2 = y_s2 - model_s2.predict(X_control)
            
            # Residuals arası korelasyon
            corr, _ = pearsonr(resid_s1, resid_s2)
            return corr if not np.isnan(corr) else 0.0
        else:
            return self.calculate_pearson_correlation(series1, series2)
    
    def calculate_lead_lag_delta(self, series1: pd.Series, series2: pd.Series, max_lag: int = 10) -> Dict[str, float]:
        """Lead-lag delta hesaplama (seri1'in seri2'ye göre liderliği)"""
        if len(series1) < max_lag * 2 or len(series2) < max_lag * 2:
            return {"delta": 0.0, "optimal_lag": 0, "max_correlation": 0.0}
            
        df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        if len(df) < max_lag * 2:
            return {"delta": 0.0, "optimal_lag": 0, "max_correlation": 0.0}
        
        correlations = []
        lags = range(-max_lag, max_lag + 1)
        
        for lag in lags:
            if lag < 0:
                # seri1 lagged, seri2 leads
                s1_lagged = df['s1'].shift(-lag)
                s2_original = df['s2']
            elif lag > 0:
                # seri1 leads, seri2 lagged
                s1_original = df['s1']
                s2_lagged = df['s2'].shift(lag)
            else:
                # no lag
                s1_original = df['s1']
                s2_original = df['s2']
            
            if lag < 0:
                aligned_df = pd.DataFrame({'s1': s1_lagged, 's2': s2_original}).dropna()
            elif lag > 0:
                aligned_df = pd.DataFrame({'s1': s1_original, 's2': s2_lagged}).dropna()
            else:
                aligned_df = df.copy()
            
            if len(aligned_df) < max_lag:
                correlations.append(0.0)
                continue
                
            corr = self.calculate_pearson_correlation(aligned_df['s1'], aligned_df['s2'])
            correlations.append(corr)
        
        if not correlations:
            return {"delta": 0.0, "optimal_lag": 0, "max_correlation": 0.0}
            
        max_corr_idx = np.argmax(np.abs(correlations))
        optimal_lag = lags[max_corr_idx]
        max_correlation = correlations[max_corr_idx]
        
        # Normalize lead-lag delta (-1 to 1)
        delta = optimal_lag / max_lag if max_lag != 0 else 0.0
        delta = max(min(delta, 1.0), -1.0)
        
        return {
            "delta": delta,
            "optimal_lag": optimal_lag,
            "max_correlation": max_correlation
        }
    
    def granger_causality_test(self, series1: pd.Series, series2: pd.Series, 
                             maxlag: int = 5) -> Dict[str, float]:
        """Granger nedensellik testi"""
        if len(series1) < maxlag * 2 or len(series2) < maxlag * 2:
            return {"p_value": 1.0, "f_statistic": 0.0, "causality_strength": 0.0}
            
        df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        if len(df) < maxlag * 2:
            return {"p_value": 1.0, "f_statistic": 0.0, "causality_strength": 0.0}
        
        try:
            # Test s2 -> s1 causation
            test_data = df[['s1', 's2']].values
            gc_test = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
            
            # En iyi lag için p-value'yu al
            best_p_value = 1.0
            best_f_stat = 0.0
            
            for lag in range(1, maxlag + 1):
                try:
                    f_stat = gc_test[lag][0]['ssr_ftest'][0]
                    p_value = gc_test[lag][0]['ssr_ftest'][1]
                    
                    if p_value < best_p_value:
                        best_p_value = p_value
                        best_f_stat = f_stat
                except (KeyError, IndexError):
                    continue
            
            # Nedensellik gücü (1 - p_value)
            causality_strength = 1.0 - best_p_value if best_p_value <= 1.0 else 0.0
            
            return {
                "p_value": best_p_value,
                "f_statistic": best_f_stat,
                "causality_strength": causality_strength
            }
            
        except Exception as e:
            self.logger.warning(f"Granger test hatası: {e}")
            return {"p_value": 1.0, "f_statistic": 0.0, "causality_strength": 0.0}
    
    def dynamic_time_warping(self, series1: pd.Series, series2: pd.Series) -> Dict[str, float]:
        """Dynamic Time Warping mesafesi"""
        if len(series1) < 2 or len(series2) < 2:
            return {"dtw_distance": 1.0, "normalized_distance": 1.0}
            
        df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        if len(df) < 2:
            return {"dtw_distance": 1.0, "normalized_distance": 1.0}
        
        try:
            # Z-score normalization
            s1_norm = (df['s1'] - df['s1'].mean()) / df['s1'].std()
            s2_norm = (df['s2'] - df['s2'].mean()) / df['s2'].std()
            
            distance, path = fastdtw(s1_norm.values, s2_norm.values, dist=euclidean)
            
            # Normalize distance (0-1 arası)
            max_possible_distance = len(s1_norm) * np.sqrt(2)  # Maksimum olası mesafe
            normalized_distance = distance / max_possible_distance if max_possible_distance > 0 else 1.0
            normalized_distance = min(normalized_distance, 1.0)
            
            return {
                "dtw_distance": distance,
                "normalized_distance": normalized_distance
            }
            
        except Exception as e:
            self.logger.warning(f"DTW hatası: {e}")
            return {"dtw_distance": 1.0, "normalized_distance": 1.0}
    
    def vector_autoregression_analysis(self, series1: pd.Series, series2: pd.Series, 
                                     max_lags: int = 5) -> Dict[str, float]:
        """VAR modeli ile impulse response analizi"""
        if len(series1) < max_lags * 3 or len(series2) < max_lags * 3:
            return {"var_strength": 0.0, "impulse_response": 0.0}
            
        df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        if len(df) < max_lags * 3:
            return {"var_strength": 0.0, "impulse_response": 0.0}
        
        try:
            # Getiri serileri
            returns = df.pct_change().dropna()
            if len(returns) < max_lags * 2:
                return {"var_strength": 0.0, "impulse_response": 0.0}
            
            # VAR modeli
            model = VAR(returns)
            results = model.fit(maxlags=max_lags, ic='aic')
            
            # Model uyum kalitesi (R-squared ortalaması)
            r_squared = np.mean([results.rsquared for results in results.equations])
            
            # Impulse response strength
            irf = results.irf(10)
            impulse_strength = np.mean(np.abs(irf.irfs))
            
            return {
                "var_strength": min(r_squared, 1.0),
                "impulse_response": min(impulse_strength, 1.0)
            }
            
        except Exception as e:
            self.logger.warning(f"VAR analiz hatası: {e}")
            return {"var_strength": 0.0, "impulse_response": 0.0}
    
    async def compute_pair_metrics(self, symbol1: str, symbol2: str, 
                                 price_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """İki sembol arası metrikleri hesapla"""
        try:
            # Sembol verilerini al
            series1 = price_data.get(symbol1)
            series2 = price_data.get(symbol2)
            
            if series1 is None or series2 is None or len(series1) < 20 or len(series2) < 20:
                return {}
            
            # Tüm metrikleri hesapla
            pearson_corr = self.calculate_pearson_correlation(series1, series2)
            beta = self.calculate_beta(series1, series2)
            rolling_cov = self.calculate_rolling_covariance(series1, series2)
            partial_corr = self.calculate_partial_correlation(series1, series2)
            
            lead_lag_result = self.calculate_lead_lag_delta(series1, series2)
            lead_lag_delta = lead_lag_result["delta"]
            
            granger_result = self.granger_causality_test(series1, series2)
            granger_strength = granger_result["causality_strength"]
            
            dtw_result = self.dynamic_time_warping(series1, series2)
            dtw_distance = 1.0 - dtw_result["normalized_distance"]  # Benzerlik için
            
            var_result = self.vector_autoregression_analysis(series1, series2)
            var_strength = var_result["var_strength"]
            
            # Bileşen skorları
            components = {
                "pearson_corr": abs(pearson_corr),  # Mutlak değer
                "beta": min(abs(beta), 2.0) / 2.0,  # Normalize
                "rolling_cov": min(abs(rolling_cov) * 1000, 1.0),  # Scale
                "partial_corr": abs(partial_corr),
                "lead_lag_delta": (lead_lag_delta + 1) / 2,  # -1,1 -> 0,1
                "granger_causality": granger_strength,
                "dtw_distance": dtw_distance,
                "var_impulse": var_strength
            }
            
            # Toplam skor
            total_score = sum(components[metric] * weight 
                            for metric, weight in self.weights.items())
            
            # Sinyal belirleme
            if total_score > 0.7:
                signal = "strong_relationship"
            elif total_score > 0.5:
                signal = "moderate_relationship" 
            elif total_score > 0.3:
                signal = "weak_relationship"
            else:
                signal = "no_relationship"
            
            # Liderlik yönü
            leadership = "symbol1_leads" if lead_lag_delta > 0.1 else \
                        "symbol2_leads" if lead_lag_delta < -0.1 else "no_clear_lead"
            
            return {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "score": total_score,
                "signal": signal,
                "leadership": leadership,
                "components": components,
                "explain": {
                    "correlation_strength": abs(pearson_corr),
                    "causality_direction": leadership,
                    "time_alignment": dtw_distance,
                    "market_beta": beta
                },
                "raw_metrics": {
                    "pearson_correlation": pearson_corr,
                    "beta_coefficient": beta,
                    "optimal_lead_lag": lead_lag_result["optimal_lag"],
                    "granger_p_value": granger_result["p_value"],
                    "dtw_raw_distance": dtw_result["dtw_distance"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Çift metrik hesaplama hatası {symbol1}-{symbol2}: {e}")
            return {}
    
    async def compute_metrics(self, symbols: List[str], **kwargs) -> Dict[str, Any]:
        """
        Ana metrik hesaplama fonksiyonu
        
        Args:
            symbols: Analiz edilecek sembol listesi
            **kwargs: Ek parametreler (interval, limit, vb.)
        """
        try:
            interval = kwargs.get('interval', '1h')
            limit = kwargs.get('limit', 100)
            
            # Fiyat verilerini al
            price_data = await self.fetch_price_data(symbols, interval, limit)
            if not price_data:
                return self._create_error_response("Fiyat verisi alınamadı")
            
            # Tüm çift kombinasyonları için metrikleri hesapla
            symbol_pairs = []
            for i, sym1 in enumerate(symbols):
                for sym2 in symbols[i+1:]:
                    # Spot ve futures kombinasyonları
                    for type1 in ['_spot', '_futures']:
                        for type2 in ['_spot', '_futures']:
                            symbol1 = f"{sym1}{type1}"
                            symbol2 = f"{sym2}{type2}"
                            if symbol1 in price_data and symbol2 in price_data:
                                symbol_pairs.append((symbol1, symbol2))
            
            # Paralel hesaplama
            tasks = [
                self.compute_pair_metrics(sym1, sym2, price_data) 
                for sym1, sym2 in symbol_pairs
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Geçerli sonuçları filtrele
            valid_results = [r for r in results if isinstance(r, dict) and r]
            
            # Korelasyon matrisi oluştur
            correlation_matrix = self._create_correlation_matrix(valid_results, symbols)
            leadership_matrix = self._create_leadership_matrix(valid_results, symbols)
            
            # Ana skor (ortalama ilişki gücü)
            avg_score = np.mean([r.get('score', 0) for r in valid_results]) if valid_results else 0.0
            
            return {
                "score": avg_score,
                "signal": "high_connectivity" if avg_score > 0.6 else 
                         "medium_connectivity" if avg_score > 0.4 else "low_connectivity",
                "components": {
                    "correlation_matrix": correlation_matrix,
                    "leadership_matrix": leadership_matrix,
                    "pair_analyses": valid_results
                },
                "explain": {
                    "total_pairs_analyzed": len(valid_results),
                    "average_correlation": np.mean([abs(r.get('raw_metrics', {}).get('pearson_correlation', 0)) 
                                                  for r in valid_results]) if valid_results else 0.0,
                    "dominant_leader": self._find_dominant_leader(valid_results, symbols),
                    "market_regime": "highly_correlated" if avg_score > 0.6 else "moderately_correlated"
                },
                "metadata": {
                    "symbols_analyzed": symbols,
                    "interval": interval,
                    "data_points": limit,
                    "calculation_time": self.get_timestamp()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Metrik hesaplama hatası: {e}")
            return self._create_error_response(str(e))
    
    def _create_correlation_matrix(self, pair_results: List[Dict], symbols: List[str]) -> Dict[str, Any]:
        """Korelasyon matrisi oluştur"""
        matrix = {}
        for symbol in symbols:
            matrix[symbol] = {}
            for other_symbol in symbols:
                if symbol == other_symbol:
                    matrix[symbol][other_symbol] = 1.0
                else:
                    # İlgili çifti bul
                    correlation = 0.0
                    for result in pair_results:
                        if (result.get('symbol1', '').startswith(symbol) and 
                            result.get('symbol2', '').startswith(other_symbol)):
                            correlation = result.get('raw_metrics', {}).get('pearson_correlation', 0.0)
                            break
                        elif (result.get('symbol2', '').startswith(symbol) and 
                              result.get('symbol1', '').startswith(other_symbol)):
                            correlation = result.get('raw_metrics', {}).get('pearson_correlation', 0.0)
                            break
                    matrix[symbol][other_symbol] = correlation
        return matrix
    
    def _create_leadership_matrix(self, pair_results: List[Dict], symbols: List[str]) -> Dict[str, Any]:
        """Liderlik matrisi oluştur"""
        matrix = {}
        for symbol in symbols:
            matrix[symbol] = {}
            for other_symbol in symbols:
                if symbol == other_symbol:
                    matrix[symbol][other_symbol] = "self"
                else:
                    leadership = "unknown"
                    for result in pair_results:
                        sym1 = result.get('symbol1', '')
                        sym2 = result.get('symbol2', '')
                        
                        if sym1.startswith(symbol) and sym2.startswith(other_symbol):
                            leadership = result.get('leadership', 'unknown')
                            break
                        elif sym2.startswith(symbol) and sym1.startswith(other_symbol):
                            # Ters çevir
                            orig_leadership = result.get('leadership', 'unknown')
                            if orig_leadership == "symbol1_leads":
                                leadership = "symbol2_leads"
                            elif orig_leadership == "symbol2_leads":
                                leadership = "symbol1_leads"
                            else:
                                leadership = orig_leadership
                            break
                    matrix[symbol][other_symbol] = leadership
        return matrix
    
    def _find_dominant_leader(self, pair_results: List[Dict], symbols: List[str]) -> str:
        """En dominant lider sembolü bul"""
        leadership_scores = {symbol: 0 for symbol in symbols}
        
        for result in pair_results:
            leadership = result.get('leadership', '')
            sym1 = result.get('symbol1', '').split('_')[0]
            sym2 = result.get('symbol2', '').split('_')[0]
            
            if leadership == "symbol1_leads" and sym1 in leadership_scores:
                leadership_scores[sym1] += 1
            elif leadership == "symbol2_leads" and sym2 in leadership_scores:
                leadership_scores[sym2] += 1
        
        if leadership_scores:
            return max(leadership_scores.items(), key=lambda x: x[1])[0]
        return "unknown"
    
    async def aggregate_output(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Çıktıyı toplu hale getir"""
        return {
            "module": self.module_name,
            "version": self.version,
            "timestamp": self.get_timestamp(),
            "analysis": results
        }
    
    async def generate_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analiz raporu oluştur"""
        analysis_data = analysis_results.get('analysis', {})
        
        return {
            "report_type": "correlation_lead_lag_analysis",
            "summary": {
                "overall_market_connectivity": analysis_data.get('score', 0),
                "market_regime": analysis_data.get('explain', {}).get('market_regime', 'unknown'),
                "dominant_leader": analysis_data.get('explain', {}).get('dominant_leader', 'unknown')
            },
            "key_insights": self._generate_insights(analysis_data),
            "recommendations": self._generate_recommendations(analysis_data),
            "detailed_analysis": analysis_data
        }
    
    def _generate_insights(self, analysis_data: Dict) -> List[str]:
        """Analiz içgörüleri oluştur"""
        insights = []
        
        avg_corr = analysis_data.get('explain', {}).get('average_correlation', 0)
        if avg_corr > 0.7:
            insights.append("Piyasalar yüksek korelasyon içinde - sistematik risk yüksek")
        elif avg_corr > 0.4:
            insights.append("Orta düzey korelasyon - çeşitlendirme fırsatları mevcut")
        else:
            insights.append("Düşük korelasyon - bağımsız hareket eden varlıklar")
        
        dominant_leader = analysis_data.get('explain', {}).get('dominant_leader', 'unknown')
        if dominant_leader != 'unknown':
            insights.append(f"{dominant_leader} piyasa lideri konumunda")
        
        return insights
    
    def _generate_recommendations(self, analysis_data: Dict) -> List[str]:
        """Öneriler oluştur"""
        recommendations = []
        
        score = analysis_data.get('score', 0)
        if score > 0.7:
            recommendations.append("Yüksek korelasyon nedeniyle çeşitlendirme az etkili olabilir")
            recommendations.append("Sistematik risk yönetimine odaklanın")
        elif score > 0.4:
            recommendations.append("Seçici çeşitlendirme ile risk yönetimi yapılabilir")
            recommendations.append("Lider takip stratejileri değerlendirilebilir")
        else:
            recommendations.append("Düşük korelasyon çeşitlendirme için uygun")
            recommendations.append("Bireysel teknik analiz daha etkili olabilir")
        
        return recommendations


# Factory fonksiyonu
def create_module(config: Dict[str, Any] = None) -> CorrelationLeadLagModule:
    """Modül factory fonksiyonu"""
    return CorrelationLeadLagModule(config)