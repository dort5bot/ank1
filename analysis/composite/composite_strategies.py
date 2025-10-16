# analysis/composite/composite_strategies.py
# Strategy Pattern ile Skorlar
"""
Bileşik Skor Stratejileri
Her bileşik skor için özel strateji sınıfları
"""


"""
Bileşik Skor Stratejileri
Her bileşik skor için özel strateji sınıfları
"""
import logging
from typing import Dict, Any, List, Tuple
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseCompositeStrategy(ABC):
    """Bileşik skor stratejileri için base sınıf"""
    
    def __init__(self):
        self.config = {}
        self.weights = {}
        self.thresholds = {}
    
    def configure(self, config: Dict):
        """Stratejiyi config ile yapılandır"""
        self.config = config
        self.weights = config.get('weights', {})
        self.thresholds = config.get('thresholds', {})
    
    @property
    @abstractmethod
    def required_modules(self) -> List[str]:
        """Bu strateji için gerekli modül listesi"""
        pass
    
    @abstractmethod
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Bileşik skoru hesapla"""
        pass
    
    def _extract_scores(self, module_results: Dict) -> Dict[str, float]:
        """Modül sonuçlarından skorları çıkar"""
        scores = {}
        for module_name in self.required_modules:
            if module_results.get(module_name) and 'score' in module_results[module_name]:
                scores[module_name] = module_results[module_name]['score']
            else:
                scores[module_name] = 0.5  # Neutral fallback
                logger.warning(f"Missing score for {module_name}, using neutral 0.5")
        
        return scores
    
    def _calculate_confidence(self, module_results: Dict) -> float:
        """Skor güvenilirliğini hesapla"""
        valid_scores = 0
        total_weight = 0
        
        for module_name, weight in self.weights.items():
            if (module_results.get(module_name) and 
                module_results[module_name].get('score') is not None):
                valid_scores += weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return valid_scores / total_weight
    
    def _get_signal(self, score: float, signal_type: str = "trend") -> str:
        """Skora göre sinyal belirle"""
        if signal_type == "trend":
            return self._get_trend_signal(score)
        elif signal_type == "risk":
            return self._get_risk_signal(score)
        else:
            return self._get_generic_signal(score)
    
    def _get_trend_signal(self, score: float) -> str:
        """Trend sinyali"""
        if score >= self.thresholds.get('strong_bullish', 0.7):
            return "strong_bullish"
        elif score >= self.thresholds.get('weak_bullish', 0.55):
            return "weak_bullish" 
        elif score <= self.thresholds.get('weak_bearish', 0.3):
            return "weak_bearish"
        elif score <= self.thresholds.get('strong_bearish', 0.1):
            return "strong_bearish"
        else:
            return "neutral"
    
    def _get_risk_signal(self, score: float) -> str:
        """Risk sinyali (ters skalada)"""
        if score >= 0.7:
            return "high_risk"
        elif score >= 0.5:
            return "medium_risk"
        else:
            return "low_risk"
    
    def _get_generic_signal(self, score: float) -> str:
        """Genel sinyal"""
        if score >= 0.6:
            return "bullish"
        elif score <= 0.4:
            return "bearish"
        else:
            return "neutral"

class TrendStrengthStrategy(BaseCompositeStrategy):
    """
    Trend Strength Score Stratejisi
    A (Trend) %35 + B (Regime) %25 + C (Sentiment) %20 + D (Order Flow) %20
    """
    
    @property
    def required_modules(self) -> List[str]:
        return ["trend_moment", "regime_anomal", "deriv_sentim", "order_micros"]
    
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Trend Strength Score hesapla"""
        try:
            # Modül skorlarını al
            scores = self._extract_scores(module_results)
            
            # Ağırlıklı ortalama hesapla
            total_score = 0.0
            total_weight = 0.0
            component_details = {}
            
            for module_name, weight in self.weights.items():
                if module_name in scores:
                    module_score = scores[module_name]
                    total_score += module_score * weight
                    total_weight += weight
                    
                    component_details[module_name] = {
                        'score': module_score,
                        'weight': weight,
                        'contribution': module_score * weight
                    }
            
            if total_weight == 0:
                final_score = 0.5
            else:
                final_score = total_score / total_weight
            
            # Güvenilirlik hesapla
            confidence = self._calculate_confidence(module_results)
            
            # Sinyal belirle
            signal = self._get_signal(final_score, "trend")
            
            # Trend yönü ve gücü analizi
            trend_analysis = self._analyze_trend_components(component_details)
            
            return {
                'score': round(final_score, 4),
                'confidence': round(confidence, 4),
                'signal': signal,
                'components': component_details,
                'trend_analysis': trend_analysis,
                'interpretation': self._interpret_trend_strength(final_score, signal),
                'timestamp': module_results.get('timestamp')
            }
            
        except Exception as e:
            logger.error(f"Trend strength calculation failed: {e}")
            return {
                'score': 0.5,
                'confidence': 0.0,
                'signal': 'error',
                'components': {},
                'error': str(e)
            }
    
    def _analyze_trend_components(self, components: Dict) -> Dict[str, Any]:
        """Trend bileşenlerini detaylı analiz et"""
        bullish_components = 0
        total_components = len(components)
        dominant_component = max(components.items(), key=lambda x: x[1]['contribution'])
        
        for comp_name, comp_data in components.items():
            if comp_data['score'] > 0.6:
                bullish_components += 1
            elif comp_data['score'] < 0.4:
                bullish_components -= 1
        
        trend_alignment = bullish_components / total_components if total_components > 0 else 0
        
        return {
            'bullish_components': bullish_components,
            'total_components': total_components,
            'trend_alignment': trend_alignment,
            'dominant_component': dominant_component[0],
            'dominant_contribution': dominant_component[1]['contribution']
        }
    
    def _interpret_trend_strength(self, score: float, signal: str) -> Dict[str, str]:
        """Trend skorunu yorumla"""
        interpretations = {
            'strong_bullish': {
                'summary': 'Çok güçlü yükseliş trendi',
                'action': 'Trend takip stratejileri uygulanabilir',
                'risk': 'Düşük - trend net'
            },
            'weak_bullish': {
                'summary': 'Zayıf yükseliş eğilimi', 
                'action': 'Doğrulama beklenmeli',
                'risk': 'Orta - trend zayıf'
            },
            'neutral': {
                'summary': 'Belirsiz trend',
                'action': 'Yan bant stratejileri uygun',
                'risk': 'Yüksek - yön belirsiz'
            },
            'weak_bearish': {
                'summary': 'Zayıf düşüş eğilimi',
                'action': 'Korunma pozisyonları',
                'risk': 'Orta - trend zayıf'
            },
            'strong_bearish': {
                'summary': 'Güçlü düşüş trendi',
                'action': 'Kısa pozisyon veya korunma',
                'risk': 'Düşük - trend net'
            }
        }
        
        return interpretations.get(signal, {
            'summary': 'Belirsiz trend durumu',
            'action': 'Temkinli olun',
            'risk': 'Yüksek'
        })

class RiskExposureStrategy(BaseCompositeStrategy):
    """Risk Exposure Score Stratejisi"""
    
    @property
    def required_modules(self) -> List[str]:
        return ["regime_anomal", "onchain", "risk_expos"]
    
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        # Implementasyon benzer şekilde
        pass

class BuyOpportunityStrategy(BaseCompositeStrategy):
    """Buy Opportunity Score Stratejisi"""
    
    @property
    def required_modules(self) -> List[str]:
        return ["trend_moment", "deriv_sentim", "order_micros", "corr_lead"]
    
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        # Implementasyon benzer şekilde
        pass


# 🔵 5. Liquidity Pressure Index
class LiquidityPressureStrategy(BaseCompositeStrategy):
    """
    Liquidity Pressure Index Stratejisi
    D (Order Flow) %60 + H (Micro Alpha) %40
    """
    
    @property
    def required_modules(self) -> List[str]:
        return ["order_micros", "microalpha"]
    
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Liquidity Pressure Index hesapla"""
        try:
            # Modül skorlarını al
            scores = self._extract_scores(module_results)
            
            # Ağırlıklı ortalama - Order Flow %60, Micro Alpha %40
            order_flow_score = scores.get("order_micros", 0.5)
            micro_alpha_score = scores.get("microalpha", 0.5)
            
            final_score = (order_flow_score * 0.6) + (micro_alpha_score * 0.4)
            
            # Likidite baskı yönünü belirle
            pressure_direction = self._calculate_pressure_direction(
                order_flow_score, micro_alpha_score, module_results
            )
            
            # Güvenilirlik hesapla
            confidence = self._calculate_confidence(module_results)
            
            # Sinyal belirle
            signal = self._get_liquidity_signal(final_score, pressure_direction)
            
            # Detaylı likidite analizi
            liquidity_analysis = self._analyze_liquidity_components(
                module_results, order_flow_score, micro_alpha_score
            )
            
            return {
                'score': round(final_score, 4),
                'confidence': round(confidence, 4),
                'signal': signal,
                'pressure_direction': pressure_direction,
                'components': {
                    'order_micros': {
                        'score': order_flow_score,
                        'weight': 0.6,
                        'contribution': order_flow_score * 0.6
                    },
                    'microalpha': {
                        'score': micro_alpha_score, 
                        'weight': 0.4,
                        'contribution': micro_alpha_score * 0.4
                    }
                },
                'liquidity_analysis': liquidity_analysis,
                'interpretation': self._interpret_liquidity_pressure(final_score, signal, pressure_direction),
                'timestamp': module_results.get('timestamp')
            }
            
        except Exception as e:
            logger.error(f"Liquidity pressure calculation failed: {e}")
            return {
                'score': 0.5,
                'confidence': 0.0,
                'signal': 'error',
                'pressure_direction': 'unknown',
                'components': {},
                'error': str(e)
            }
    
    def _calculate_pressure_direction(self, order_flow_score: float, 
                                    micro_alpha_score: float, 
                                    module_results: Dict) -> str:
        """Likidite baskı yönünü belirle"""
        # Order Flow'dan alım/satım baskısı bilgisini çıkar
        order_flow_data = module_results.get("order_micros", {})
        micro_alpha_data = module_results.get("microalpha", {})
        
        buy_pressure = 0
        sell_pressure = 0
        
        # Order Flow sinyalinden yön tespiti
        if order_flow_data.get('signal') == 'buy_pressure':
            buy_pressure += 1
        elif order_flow_data.get('signal') == 'sell_pressure':
            sell_pressure += 1
        
        # Micro Alpha trendinden yön tespiti
        if micro_alpha_data.get('signal') == 'bullish':
            buy_pressure += 1
        elif micro_alpha_data.get('signal') == 'bearish':
            sell_pressure += 1
        
        # Skor farkından yön tespiti
        if order_flow_score > 0.6:
            buy_pressure += 1
        elif order_flow_score < 0.4:
            sell_pressure += 1
            
        if micro_alpha_score > 0.6:
            buy_pressure += 1
        elif micro_alpha_score < 0.4:
            sell_pressure += 1
        
        if buy_pressure > sell_pressure:
            return "buying_pressure"
        elif sell_pressure > buy_pressure:
            return "selling_pressure"
        else:
            return "balanced"
    
    def _get_liquidity_signal(self, score: float, direction: str) -> str:
        """Likidite sinyali"""
        if score >= 0.7:
            if direction == "buying_pressure":
                return "strong_buying_pressure"
            elif direction == "selling_pressure":
                return "strong_selling_pressure"
            else:
                return "high_liquidity_volatility"
        elif score >= 0.6:
            if direction == "buying_pressure":
                return "moderate_buying_pressure"
            elif direction == "selling_pressure":
                return "moderate_selling_pressure"
            else:
                return "elevated_liquidity"
        elif score <= 0.3:
            return "low_liquidity"
        else:
            return "normal_liquidity"
    
    def _analyze_liquidity_components(self, module_results: Dict, 
                                    order_flow_score: float, 
                                    micro_alpha_score: float) -> Dict[str, Any]:
        """Likidite bileşenlerini detaylı analiz"""
        order_flow_details = module_results.get("order_micros", {})
        micro_alpha_details = module_results.get("microalpha", {})
        
        # Order Flow metrikleri
        ofi_metric = order_flow_details.get('components', {}).get('orderbook_imbalance', 0)
        cvd_metric = order_flow_details.get('components', {}).get('cvd', 0)
        
        # Micro Alpha metrikleri
        alpha_metric = micro_alpha_details.get('score', 0.5)
        market_impact = micro_alpha_details.get('metrics', {}).get('market_impact', 0)
        
        return {
            'order_flow_imbalance': ofi_metric,
            'cumulative_volume_delta': cvd_metric,
            'micro_alpha_strength': alpha_metric,
            'market_impact_factor': market_impact,
            'pressure_strength': abs(order_flow_score - 0.5) * 2,  # 0-1 arası normalize
            'consistency': 1 - abs(order_flow_score - micro_alpha_score)  # Tutarlılık
        }
    
    def _interpret_liquidity_pressure(self, score: float, signal: str, direction: str) -> Dict[str, str]:
        """Likidite baskısını yorumla"""
        interpretations = {
            'strong_buying_pressure': {
                'summary': 'Güçlü alım likiditesi - Fiyat yükselme potansiyeli yüksek',
                'action': 'Long pozisyonlar için uygun ortam',
                'warning': 'Aşırı alım bölgesinde olabilir'
            },
            'moderate_buying_pressure': {
                'summary': 'Orta seviye alım likiditesi',
                'action': 'Kademeli long girişleri değerlendirilebilir',
                'warning': 'Trend devamı için diğer göstergeleri kontrol edin'
            },
            'strong_selling_pressure': {
                'summary': 'Güçlü satım likiditesi - Fiyat düşme potansiyeli yüksek',
                'action': 'Short pozisyonlar veya korunma',
                'warning': 'Aşırı satım bölgesinde olabilir'
            },
            'moderate_selling_pressure': {
                'summary': 'Orta seviye satım likiditesi',
                'action': 'Kademeli short girişleri değerlendirilebilir',
                'warning': 'Stop-loss kullanımı önemli'
            },
            'high_liquidity_volatility': {
                'summary': 'Yüksek likidite oynaklığı - Belirsizlik hakim',
                'action': 'Pozisyon boyutlarını küçük tutun',
                'warning': 'Yanlış sinyal riski yüksek'
            },
            'low_liquidity': {
                'summary': 'Düşük likidite - Sıçrama riski yüksek',
                'action': 'İşlem yapmaktan kaçının veya çok küçük pozisyonlar',
                'warning': 'Slippage riski çok yüksek'
            }
        }
        
        return interpretations.get(signal, {
            'summary': 'Normal likidite koşulları',
            'action': 'Standart işlem stratejileri uygulanabilir',
            'warning': 'Diğer göstergelerle teyit edin'
        })


# ⚪ 7. Anomaly Detection Alert Score
class AnomalyDetectionStrategy(BaseCompositeStrategy):
    """
    Anomaly Detection Alert Score Stratejisi
    J (Regime Change) %40 + C (Sentiment) %30 + G (Risk) %30
    """
    
    @property
    def required_modules(self) -> List[str]:
        return ["regime_anomal", "deriv_sentim", "risk_expos"]
    
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Anomaly Detection Alert Score hesapla"""
        try:
            # Modül skorlarını al
            scores = self._extract_scores(module_results)
            
            # Anomali skorları - yüksek skor anomaliyi gösterir
            regime_score = scores.get("regime_anomal", 0.5)
            sentiment_score = scores.get("deriv_sentim", 0.5)
            risk_score = scores.get("risk_expos", 0.5)
            
            # Anomali skoru: normalde 0.5'ten uzaklaşma
            regime_anomaly = abs(regime_score - 0.5) * 2  # 0-1 arası
            sentiment_anomaly = abs(sentiment_score - 0.5) * 2
            risk_anomaly = risk_score  # Risk zaten yüksek=anomali
            
            # Ağırlıklı anomali skoru
            final_score = (
                regime_anomaly * 0.4 + 
                sentiment_anomaly * 0.3 + 
                risk_anomaly * 0.3
            )
            
            # Anomali tipini belirle
            anomaly_type = self._detect_anomaly_type(
                regime_score, sentiment_score, risk_score, module_results
            )
            
            # Anomali şiddeti
            severity = self._calculate_anomaly_severity(final_score, anomaly_type)
            
            # Güvenilirlik
            confidence = self._calculate_confidence(module_results)
            
            # Aciliyet seviyesi
            urgency = self._calculate_urgency(final_score, anomaly_type, severity)
            
            return {
                'score': round(final_score, 4),
                'confidence': round(confidence, 4),
                'signal': self._get_anomaly_signal(final_score),
                'anomaly_type': anomaly_type,
                'severity': severity,
                'urgency': urgency,
                'components': {
                    'regime_anomal': {
                        'score': regime_score,
                        'anomaly_score': regime_anomaly,
                        'weight': 0.4,
                        'contribution': regime_anomaly * 0.4
                    },
                    'deriv_sentim': {
                        'score': sentiment_score,
                        'anomaly_score': sentiment_anomaly,
                        'weight': 0.3,
                        'contribution': sentiment_anomaly * 0.3
                    },
                    'risk_expos': {
                        'score': risk_score,
                        'anomaly_score': risk_anomaly,
                        'weight': 0.3,
                        'contribution': risk_anomaly * 0.3
                    }
                },
                'anomaly_analysis': self._analyze_anomaly_pattern(
                    regime_score, sentiment_score, risk_score, module_results
                ),
                'recommendations': self._generate_anomaly_recommendations(
                    final_score, anomaly_type, severity, urgency
                ),
                'timestamp': module_results.get('timestamp')
            }
            
        except Exception as e:
            logger.error(f"Anomaly detection calculation failed: {e}")
            return {
                'score': 0.0,
                'confidence': 0.0,
                'signal': 'calculation_error',
                'anomaly_type': 'unknown',
                'severity': 'low',
                'urgency': 'low',
                'components': {},
                'error': str(e)
            }
    
    def _detect_anomaly_type(self, regime_score: float, sentiment_score: float, 
                           risk_score: float, module_results: Dict) -> str:
        """Anomali tipini tespit et"""
        regime_data = module_results.get("regime_anomal", {})
        sentiment_data = module_results.get("deriv_sentim", {})
        risk_data = module_results.get("risk_expos", {})
        
        # Rejim değişikliği anomalisi
        if regime_score > 0.7:
            return "regime_change"
        
        # Sentiment anomalisi
        if abs(sentiment_score - 0.5) > 0.3:
            if sentiment_score > 0.5:
                return "extreme_bullish_sentiment"
            else:
                return "extreme_bearish_sentiment"
        
        # Risk anomalisi
        if risk_score > 0.7:
            return "high_risk_environment"
        
        # Volatilite anomalisi
        regime_signal = regime_data.get('signal', '')
        if 'anomaly' in regime_signal:
            return "volatility_anomaly"
        
        # Kombine anomali
        anomaly_count = 0
        if abs(regime_score - 0.5) > 0.2: anomaly_count += 1
        if abs(sentiment_score - 0.5) > 0.2: anomaly_count += 1
        if risk_score > 0.6: anomaly_count += 1
        
        if anomaly_count >= 2:
            return "combined_anomaly"
        
        return "no_clear_anomaly"
    
    def _calculate_anomaly_severity(self, score: float, anomaly_type: str) -> str:
        """Anomali şiddetini hesapla"""
        if score >= 0.8:
            return "critical"
        elif score >= 0.7:
            return "high"
        elif score >= 0.6:
            return "medium"
        elif score >= 0.5:
            return "low"
        else:
            return "none"
    
    def _calculate_urgency(self, score: float, anomaly_type: str, severity: str) -> str:
        """Aciliyet seviyesi"""
        if severity == "critical":
            return "immediate"
        elif severity == "high":
            return "high"
        elif severity == "medium":
            return "medium"
        else:
            return "low"
    
    def _get_anomaly_signal(self, score: float) -> str:
        """Anomali sinyali"""
        if score >= 0.8:
            return "critical_anomaly"
        elif score >= 0.7:
            return "high_anomaly"
        elif score >= 0.6:
            return "medium_anomaly"
        elif score >= 0.5:
            return "low_anomaly"
        else:
            return "normal"
    
    def _analyze_anomaly_pattern(self, regime_score: float, sentiment_score: float,
                               risk_score: float, module_results: Dict) -> Dict[str, Any]:
        """Anomali pattern analizi"""
        # Anomali korelasyonu
        scores = [regime_score, sentiment_score, risk_score]
        avg_score = sum(scores) / len(scores)
        std_dev = np.std(scores)
        
        # Anomali konsensüsü
        anomaly_consensus = sum(1 for s in scores if abs(s - 0.5) > 0.2)
        
        # Trend yönü
        trend_direction = "bullish" if avg_score > 0.5 else "bearish"
        
        return {
            'average_score': avg_score,
            'volatility': std_dev,
            'anomaly_consensus': anomaly_consensus,
            'trend_direction': trend_direction,
            'deviation_from_normal': abs(avg_score - 0.5) * 2,
            'pattern_strength': min(1.0, std_dev * 2)  # Yüksek std = güçlü pattern
        }
    
    def _generate_anomaly_recommendations(self, score: float, anomaly_type: str,
                                        severity: str, urgency: str) -> Dict[str, str]:
        """Anomali önerileri oluştur"""
        base_recommendations = {
            'critical': {
                'action': 'TÜM POZİSYONLARI KAPAT - ACİL DURUM',
                'monitoring': '15 dakika aralıklarla takip',
                're_entry': 'Anomali çözülene kadar yeni pozisyon AÇMAYIN'
            },
            'high': {
                'action': 'Pozisyon boyutlarını %50 azalt',
                'monitoring': '30 dakika aralıklarla takip',
                're_entry': 'Sinyal netleşene kadar bekleyin'
            },
            'medium': {
                'action': 'Stop-loss seviyelerini sıkılaştır',
                'monitoring': '1 saat aralıklarla takip',
                're_entry': 'Ek doğrulama sinyali bekleyin'
            },
            'low': {
                'action': 'Mevcut stratejiyi sürdür, dikkatli ol',
                'monitoring': 'Normal takip periyodu',
                're_entry': 'Standart kurallara göre işlem yapın'
            }
        }
        
        return base_recommendations.get(severity, {
            'action': 'Normal işlem stratejisi',
            'monitoring': 'Rutin takip',
            're_entry': 'Standart kurallar'
        })



# 🟣 8. Market Health Score
class MarketHealthStrategy(BaseCompositeStrategy):
    """
    Market Health Score Stratejisi
    Trend Strength %30 + Risk Exposure %25 + Liquidity Pressure %25 + Macro Sentiment %20
    """
    
    @property
    def required_modules(self) -> List[str]:
        return ["trend_moment", "risk_expos", "order_micros", "deriv_sentim"]
    
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Market Health Score hesapla"""
        try:
            # Diğer bileşik skorlardan faydalan (eğer hesaplanmışsa)
            # Veya doğrudan modül skorlarını kullan
            
            scores = self._extract_scores(module_results)
            
            # Bileşen skorları
            trend_score = scores.get("trend_moment", 0.5)
            risk_score = 1.0 - scores.get("risk_expos", 0.5)  # Risk tersi = Sağlık
            liquidity_score = scores.get("order_micros", 0.5)
            macro_score = scores.get("deriv_sentim", 0.5)
            
            # Ağırlıklı ortalama
            final_score = (
                trend_score * 0.30 +
                risk_score * 0.25 +
                liquidity_score * 0.25 +
                macro_score * 0.20
            )
            
            # Piyasa sağlık durumu
            health_status = self._assess_health_status(final_score, trend_score, risk_score)
            
            # Güvenilirlik
            confidence = self._calculate_confidence(module_results)
            
            # Sağlık trendi
            health_trend = self._calculate_health_trend(module_results)
            
            return {
                'score': round(final_score, 4),
                'confidence': round(confidence, 4),
                'signal': self._get_health_signal(final_score),
                'health_status': health_status,
                'health_trend': health_trend,
                'components': {
                    'trend_strength': {
                        'score': trend_score,
                        'weight': 0.30,
                        'contribution': trend_score * 0.30
                    },
                    'risk_health': {
                        'score': risk_score,
                        'weight': 0.25,
                        'contribution': risk_score * 0.25
                    },
                    'liquidity_health': {
                        'score': liquidity_score,
                        'weight': 0.25,
                        'contribution': liquidity_score * 0.25
                    },
                    'macro_sentiment': {
                        'score': macro_score,
                        'weight': 0.20,
                        'contribution': macro_score * 0.20
                    }
                },
                'health_analysis': self._analyze_market_health(
                    trend_score, risk_score, liquidity_score, macro_score
                ),
                'market_outlook': self._generate_market_outlook(final_score, health_status),
                'timestamp': module_results.get('timestamp')
            }
            
        except Exception as e:
            logger.error(f"Market health calculation failed: {e}")
            return {
                'score': 0.5,
                'confidence': 0.0,
                'signal': 'error',
                'health_status': 'unknown',
                'components': {},
                'error': str(e)
            }
    
    def _assess_health_status(self, overall_score: float, trend_score: float, 
                            risk_score: float) -> str:
        """Piyasa sağlık durumunu değerlendir"""
        if overall_score >= 0.8:
            return "excellent"
        elif overall_score >= 0.7:
            return "very_healthy"
        elif overall_score >= 0.6:
            return "healthy"
        elif overall_score >= 0.5:
            return "moderate"
        elif overall_score >= 0.4:
            return "weak"
        elif overall_score >= 0.3:
            return "unhealthy"
        else:
            return "critical"
    
    def _calculate_health_trend(self, module_results: Dict) -> str:
        """Sağlık trendini hesapla"""
        # Basit trend analizi - gerçek implementasyonda historical data kullan
        trend_data = module_results.get("trend_moment", {})
        trend_signal = trend_data.get('signal', 'neutral')
        
        if trend_signal in ['strong_bullish', 'bullish']:
            return "improving"
        elif trend_signal in ['strong_bearish', 'bearish']:
            return "deteriorating"
        else:
            return "stable"
    
    def _get_health_signal(self, score: float) -> str:
        """Sağlık sinyali"""
        if score >= 0.7:
            return "very_healthy"
        elif score >= 0.6:
            return "healthy"
        elif score >= 0.5:
            return "moderate"
        elif score >= 0.4:
            return "caution"
        else:
            return "unhealthy"
    
    def _analyze_market_health(self, trend_score: float, risk_score: float,
                             liquidity_score: float, macro_score: float) -> Dict[str, Any]:
        """Piyasa sağlık analizi"""
        # Bileşen uyumu
        component_scores = [trend_score, risk_score, liquidity_score, macro_score]
        consistency = 1.0 - (np.std(component_scores) / 0.5)  # 0-1 arası tutarlılık
        
        # Güçlü ve zayıf yönler
        strengths = []
        weaknesses = []
        
        if trend_score > 0.6: strengths.append("strong_trend")
        elif trend_score < 0.4: weaknesses.append("weak_trend")
        
        if risk_score > 0.6: strengths.append("low_risk")
        elif risk_score < 0.4: weaknesses.append("high_risk")
        
        if liquidity_score > 0.6: strengths.append("good_liquidity")
        elif liquidity_score < 0.4: weaknesses.append("poor_liquidity")
        
        if macro_score > 0.6: strengths.append("positive_sentiment")
        elif macro_score < 0.4: weaknesses.append("negative_sentiment")
        
        return {
            'component_consistency': consistency,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'balanced_market': len(strengths) == len(weaknesses),
            'overall_strength': len(strengths) / 4.0  # 4 bileşen
        }
    
    def _generate_market_outlook(self, score: float, health_status: str) -> Dict[str, str]:
        """Piyasa görünümü oluştur"""
        outlooks = {
            'excellent': {
                'summary': 'Piyasa koşulları mükemmel - Güçlü trend, düşük risk',
                'outlook': 'Çok olumlu',
                'timeframe': 'Kısa-Orta vadeli'
            },
            'very_healthy': {
                'summary': 'Piyasa sağlıklı - İyi trend, makul risk',
                'outlook': 'Olumlu',
                'timeframe': 'Kısa-Orta vadeli'
            },
            'healthy': {
                'summary': 'Piyasa normal sağlıkta - Dengeli koşullar',
                'outlook': 'Nötr-Olumlu',
                'timeframe': 'Kısa vade'
            },
            'moderate': {
                'summary': 'Piyasa orta seviyede - Dikkatli olunması gereken koşullar',
                'outlook': 'Nötr',
                'timeframe': 'Çok kısa vade'
            },
            'weak': {
                'summary': 'Piyasa zayıf - Riskler artmış durumda',
                'outlook': 'Olumsuz',
                'timeframe': 'Çok kısa vade - İzleme'
            },
            'unhealthy': {
                'summary': 'Piyasa sağlıksız - Yüksek risk, zayıf trend',
                'outlook': 'Çok olumsuz',
                'timeframe': 'Pozisyon alınmamalı'
            },
            'critical': {
                'summary': 'Piyasa kritik durumda - Acil önlem gerekli',
                'outlook': 'Acil müdahale',
                'timeframe': 'Pozisyonlardan çıkılmalı'
            }
        }
        
        return outlooks.get(health_status, {
            'summary': 'Piyasa durumu belirsiz',
            'outlook': 'Belirsiz',
            'timeframe': 'İzleme önerilir'
        })
        
# 🔶 9. Swing Trading Signal
class SwingTradingStrategy(BaseCompositeStrategy):
    """
    Swing Trading Signal Stratejisi
    Trend Strength %40 + Buy Opportunity %30 + Risk Exposure %30
    """
    
    @property
    def required_modules(self) -> List[str]:
        return ["trend_moment", "deriv_sentim", "risk_expos"]
    
    async def calculate(self, module_results: Dict, symbol: str) -> Dict[str, Any]:
        """Swing Trading Signal hesapla"""
        try:
            scores = self._extract_scores(module_results)
            
            # Bileşen skorları
            trend_score = scores.get("trend_moment", 0.5)
            buy_opportunity_score = scores.get("deriv_sentim", 0.5)
            risk_score = 1.0 - scores.get("risk_expos", 0.5)  # Risk tersi = Fırsat
            
            # Ağırlıklı ortalama
            final_score = (
                trend_score * 0.40 +
                buy_opportunity_score * 0.30 +
                risk_score * 0.30
            )
            
            # Swing trading sinyali
            swing_signal = self._generate_swing_signal(final_score, trend_score, buy_opportunity_score)
            
            # Zaman çerçevesi uygunluğu
            timeframe_suitability = self._assess_timeframe_suitability(module_results)
            
            # Pozisyon önerisi
            position_suggestion = self._suggest_position(final_score, swing_signal, risk_score)
            
            return {
                'score': round(final_score, 4),
                'confidence': round(self._calculate_confidence(module_results), 4),
                'signal': swing_signal,
                'timeframe_suitability': timeframe_suitability,
                'position_suggestion': position_suggestion,
                'components': {
                    'trend_strength': {
                        'score': trend_score,
                        'weight': 0.40,
                        'contribution': trend_score * 0.40
                    },
                    'buy_opportunity': {
                        'score': buy_opportunity_score,
                        'weight': 0.30,
                        'contribution': buy_opportunity_score * 0.30
                    },
                    'risk_adjusted': {
                        'score': risk_score,
                        'weight': 0.30,
                        'contribution': risk_score * 0.30
                    }
                },
                'swing_analysis': self._analyze_swing_setup(
                    trend_score, buy_opportunity_score, risk_score, module_results
                ),
                'trading_plan': self._generate_trading_plan(
                    final_score, swing_signal, position_suggestion
                ),
                'timestamp': module_results.get('timestamp')
            }
            
        except Exception as e:
            logger.error(f"Swing trading signal calculation failed: {e}")
            return {
                'score': 0.5,
                'confidence': 0.0,
                'signal': 'neutral',
                'timeframe_suitability': 'unknown',
                'position_suggestion': 'no_trade',
                'components': {},
                'error': str(e)
            }
    
    def _generate_swing_signal(self, overall_score: float, trend_score: float, 
                             buy_opportunity_score: float) -> str:
        """Swing trading sinyali oluştur"""
        # Trend ve fırsat kombinasyonu
        if overall_score >= 0.7 and trend_score >= 0.6 and buy_opportunity_score >= 0.6:
            return "strong_buy"
        elif overall_score >= 0.6 and trend_score >= 0.5 and buy_opportunity_score >= 0.5:
            return "buy"
        elif overall_score <= 0.3 and trend_score <= 0.4 and buy_opportunity_score <= 0.4:
            return "strong_sell"
        elif overall_score <= 0.4 and trend_score <= 0.5 and buy_opportunity_score <= 0.5:
            return "sell"
        elif overall_score >= 0.55 and trend_score >= 0.5:
            return "hold_long"
        elif overall_score <= 0.45 and trend_score <= 0.5:
            return "hold_short"
        else:
            return "no_trade"
    
    def _assess_timeframe_suitability(self, module_results: Dict) -> Dict[str, Any]:
        """Zaman çerçevesi uygunluğunu değerlendir"""
        trend_data = module_results.get("trend_moment", {})
        regime_data = module_results.get("regime_anomal", {})
        
        # Trend gücüne göre timeframe
        trend_strength = trend_data.get('score', 0.5)
        if trend_strength > 0.7:
            timeframe = "1-3_gün"
        elif trend_strength > 0.6:
            timeframe = "3-5_gün"
        elif trend_strength > 0.5:
            timeframe = "5-7_gün"
        else:
            timeframe = "1_gün_altı"
        
        # Volatilite uygunluğu
        regime_signal = regime_data.get('signal', 'neutral')
        if 'high_volatility' in regime_signal:
            volatility_suitability = "high_risk"
        elif 'low_volatility' in regime_signal:
            volatility_suitability = "low_opportunity"
        else:
            volatility_suitability = "optimal"
        
        return {
            'recommended_timeframe': timeframe,
            'volatility_suitability': volatility_suitability,
            'swing_potential': trend_strength,
            'risk_period': "short_term" if timeframe == "1_gün_altı" else "medium_term"
        }
    
    def _suggest_position(self, score: float, signal: str, risk_score: float) -> Dict[str, Any]:
        """Pozisyon önerisi"""
        position_sizes = {
            'strong_buy': {'size': 'large', 'allocation': 0.7},
            'buy': {'size': 'medium', 'allocation': 0.5},
            'hold_long': {'size': 'small', 'allocation': 0.3},
            'no_trade': {'size': 'none', 'allocation': 0.0},
            'hold_short': {'size': 'small', 'allocation': 0.3},
            'sell': {'size': 'medium', 'allocation': 0.5},
            'strong_sell': {'size': 'large', 'allocation': 0.7}
        }
        
        base_suggestion = position_sizes.get(signal, {'size': 'none', 'allocation': 0.0})
        
        # Risk adjust
        risk_adjusted_allocation = base_suggestion['allocation'] * risk_score
        
        return {
            'action': signal,
            'position_size': base_suggestion['size'],
            'suggested_allocation': risk_adjusted_allocation,
            'risk_adjusted': True,
            'confidence_level': score
        }
    
    def _analyze_swing_setup(self, trend_score: float, opportunity_score: float,
                           risk_score: float, module_results: Dict) -> Dict[str, Any]:
        """Swing setup analizi"""
        # Setup kalitesi
        quality_score = (trend_score + opportunity_score + risk_score) / 3.0
        
        # Entry timing
        trend_data = module_results.get("trend_moment", {})
        trend_signal = trend_data.get('signal', 'neutral')
        
        if trend_signal in ['strong_bullish', 'bullish']:
            entry_timing = "early_trend"
        elif trend_signal == 'neutral':
            entry_timing = "consolidation"
        else:
            entry_timing = "counter_trend"
        
        # Risk/Reward profile
        if risk_score > 0.7 and trend_score > 0.6:
            risk_reward = "favorable"
        elif risk_score > 0.5 and trend_score > 0.5:
            risk_reward = "moderate"
        else:
            risk_reward = "unfavorable"
        
        return {
            'setup_quality': quality_score,
            'entry_timing': entry_timing,
            'risk_reward_profile': risk_reward,
            'trend_alignment': "aligned" if trend_score > 0.5 else "counter",
            'opportunity_strength': opportunity_score,
            'overall_rating': min(1.0, quality_score * 1.2)  # 1.2 multiplier for emphasis
        }
    
    def _generate_trading_plan(self, score: float, signal: str, 
                             position_suggestion: Dict) -> Dict[str, str]:
        """Trading plan oluştur"""
        plans = {
            'strong_buy': {
                'entry': 'Agressive entry - %70-80 allocation',
                'stop_loss': '3-5% below entry, tight stop',
                'take_profit': '8-12% target, trail stop after 5% gain',
                'management': 'Scale out at 5% and 10% targets'
            },
            'buy': {
                'entry': 'Standard entry - %50 allocation',
                'stop_loss': '5-7% below entry',
                'take_profit': '10-15% target',
                'management': 'Hold for swing duration, partial profit at 8%'
            },
            'sell': {
                'entry': 'Short entry - %50 allocation',
                'stop_loss': '5-7% above entry',
                'take_profit': '8-12% target',
                'management': 'Quick profits, tight stops'
            },
            'no_trade': {
                'entry': 'No position',
                'stop_loss': 'N/A',
                'take_profit': 'N/A',
                'management': 'Wait for better setup'
            }
        }
        
        return plans.get(signal, {
            'entry': 'Wait for confirmation',
            'stop_loss': 'Standard 5% stop',
            'take_profit': '8-10% target',
            'management': 'Monitor closely'
        })

# 
