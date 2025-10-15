# analysis/composite/composite_engine.py
# Composite Engine (Merkezi)

"""
Bileşik Skor Motoru - Ana Composite Score Engine
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
import yaml
from pathlib import Path

from analysis.analysis_base_module import BaseAnalysisModule

logger = logging.getLogger(__name__)

class CompositeScoreEngine:
    """
    Ana bileşik skor hesaplama motoru
    Tüm bileşik skor stratejilerini yönetir
    """
    
    def __init__(self, aggregator, config_path: Optional[str] = None):
        self.aggregator = aggregator
        self.strategies = {}
        self.config = self._load_config(config_path)
        self._initialize_strategies()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Config dosyasını yükle"""
        if config_path is None:
            config_path = Path(__file__).parent / "composite_config.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Composite config yüklenemedi: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Varsayılan config"""
        return {
            'composite_scores': {
                'trend_strength': {
                    'description': 'Trend yönü ve momentum gücü',
                    'modules': ['trend_moment', 'regime_anomal', 'deriv_sentim', 'order_micros'],
                    'weights': {'trend_moment': 0.35, 'regime_anomal': 0.25, 
                               'deriv_sentim': 0.20, 'order_micros': 0.20},
                    'thresholds': {
                        'strong_bullish': 0.7,
                        'weak_bullish': 0.55, 
                        'neutral': 0.45,
                        'weak_bearish': 0.3,
                        'strong_bearish': 0.0
                    }
                }
            }
        }
    
    def _initialize_strategies(self):
        """Stratejileri başlat"""
        from .composite_strategies import (
            TrendStrengthStrategy,
            RiskExposureStrategy,
            BuyOpportunityStrategy
        )
        
        strategy_map = {
            'trend_strength': TrendStrengthStrategy(),
            'risk_exposure': RiskExposureStrategy(),
            'buy_opportunity': BuyOpportunityStrategy()
        }
        
        # Config'te tanımlı stratejileri yükle
        for score_name, config in self.config.get('composite_scores', {}).items():
            if score_name in strategy_map:
                strategy_map[score_name].configure(config)
                self.strategies[score_name] = strategy_map[score_name]
                logger.info(f"Composite strategy loaded: {score_name}")
    
    async def calculate_composite_scores(self, symbol: str) -> Dict[str, Any]:
        """
        Tüm bileşik skorları hesapla
        
        Args:
            symbol: Analiz edilecek sembol
            
        Returns:
            Tüm bileşik skorların dict'i
        """
        try:
            # Tüm modül sonuçlarını paralel al
            module_results = await self._gather_module_results(symbol)
            
            # Bileşik skorları hesapla
            composite_scores = {}
            for score_name, strategy in self.strategies.items():
                try:
                    composite_scores[score_name] = await strategy.calculate(
                        module_results, symbol
                    )
                except Exception as e:
                    logger.error(f"Error calculating {score_name}: {e}")
                    composite_scores[score_name] = self._get_error_score(score_name)
            
            return {
                'symbol': symbol,
                'composite_scores': composite_scores,
                'timestamp': asyncio.get_event_loop().time(),
                'metadata': {
                    'strategies_used': list(self.strategies.keys()),
                    'modules_analyzed': list(module_results.keys())
                }
            }
            
        except Exception as e:
            logger.error(f"Composite score calculation failed for {symbol}: {e}")
            return self._get_fallback_response(symbol)
    
    async def _gather_module_results(self, symbol: str) -> Dict[str, Any]:
        """Tüm modül sonuçlarını topla"""
        # Aggregator üzerinden modül sonuçlarını al
        # Bu kısım mevcut aggregator yapınıza göre adapte edilmeli
        required_modules = set()
        for strategy in self.strategies.values():
            required_modules.update(strategy.required_modules)
        
        tasks = []
        for module_name in required_modules:
            task = self.aggregator.get_module_analysis(module_name, symbol)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        module_results = {}
        for i, module_name in enumerate(required_modules):
            if not isinstance(results[i], Exception):
                module_results[module_name] = results[i]
            else:
                logger.warning(f"Module {module_name} failed: {results[i]}")
                module_results[module_name] = None
        
        return module_results
    
    def _get_error_score(self, score_name: str) -> Dict:
        """Hata durumu için default skor"""
        return {
            'score': 0.5,
            'confidence': 0.0,
            'signal': 'error',
            'components': {},
            'error': True
        }
    
    def _get_fallback_response(self, symbol: str) -> Dict:
        """Fallback response"""
        return {
            'symbol': symbol,
            'composite_scores': {},
            'timestamp': asyncio.get_event_loop().time(),
            'error': 'Composite calculation failed',
            'metadata': {'fallback': True}
        }
    
    async def calculate_single_score(self, score_name: str, symbol: str) -> Optional[Dict]:
        """Tekil bileşik skor hesapla"""
        if score_name not in self.strategies:
            logger.error(f"Unknown composite score: {score_name}")
            return None
        
        module_results = await self._gather_module_results(symbol)
        return await self.strategies[score_name].calculate(module_results, symbol)
        