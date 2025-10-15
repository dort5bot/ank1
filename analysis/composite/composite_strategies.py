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
        