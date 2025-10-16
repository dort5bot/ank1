# analysis/composite/composite_engine.py
# Composite Engine (Merkezi)

"""
Bileşik Skor Motoru - Ana Composite Score Engine
"""
import asyncio
import logging
import time
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

        self.performance_metrics = {
            'calculation_times': [],
            'error_rates': {},
            'cache_efficiency': {'hits': 0, 'misses': 0}
        }
        self.alert_thresholds = {
            'max_calculation_time': 5.0,  # 5 saniye
            'max_error_rate': 0.1,  # %10
            'min_confidence': 0.7
        }
        

    
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
            BuyOpportunityStrategy,
            LiquidityPressureStrategy,      # YENİ
            AnomalyDetectionStrategy,       # YENİ  
            MarketHealthStrategy,           # YENİ
            SwingTradingStrategy           # YENİ
        )
        
        strategy_map = {
            'trend_strength': TrendStrengthStrategy(),
            'risk_exposure': RiskExposureStrategy(),
            'buy_opportunity': BuyOpportunityStrategy(),
            'liquidity_pressure': LiquidityPressureStrategy(),    # YENİ
            'anomaly_alert': AnomalyDetectionStrategy(),         # YENİ
            'market_health': MarketHealthStrategy(),             # YENİ
            'swing_trading': SwingTradingStrategy()              # YENİ
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
        start_time = time.time()
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
                    
                    #1/2
                    calculation_time = time.time() - start_time
                    self._update_performance_metrics(symbol, calculation_time, success=True)
                    self._check_alert_conditions(symbol, calculation_time, composite_scores)
                            
                except Exception as e:
                    logger.error(f"Error calculating {score_name}: {e}")
                    composite_scores[score_name] = self._get_error_score(score_name)
                    # 2/2
                    self._update_performance_metrics(symbol, 0, success=False)
                    self._send_alert(f"CRITICAL: Composite calculation failed for {symbol}: {e}")
            
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
    


    async def _gather_module_results_optimized(self, symbol: str) -> Dict[str, Any]:
        """Paralel module result toplama - optimize edilmiş"""
        required_modules = set()
        for strategy in self.strategies.values():
            required_modules.update(strategy.required_modules)
        
        # ✅ TÜM MODÜLLER AYNI ANDA ÇAĞRILIR
        tasks = {}
        for module_name in required_modules:
            tasks[module_name] = self.aggregator.get_module_analysis(module_name, symbol)
        
        # Asenkron olarak paralel çalıştır
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Result'ları module isimleriyle eşle
        module_results = {}
        for i, (module_name, task) in enumerate(tasks.items()):
            if not isinstance(results[i], Exception):
                module_results[module_name] = results[i]
            else:
                logger.warning(f"Module {module_name} failed: {results[i]}")
                module_results[module_name] = self._get_neutral_module_result(module_name)
        
        return module_results
        
    

    def _update_performance_metrics(self, symbol: str, calculation_time: float, success: bool):
        """Performance metriklerini güncelle"""
        self.performance_metrics['calculation_times'].append(calculation_time)
        
        # Error rate tracking
        if symbol not in self.performance_metrics['error_rates']:
            self.performance_metrics['error_rates'][symbol] = {'success': 0, 'failure': 0}
        
        if success:
            self.performance_metrics['error_rates'][symbol]['success'] += 1
        else:
            self.performance_metrics['error_rates'][symbol]['failure'] += 1
        
        # Keep only last 1000 records
        if len(self.performance_metrics['calculation_times']) > 1000:
            self.performance_metrics['calculation_times'] = self.performance_metrics['calculation_times'][-500:]
    
    def _check_alert_conditions(self, symbol: str, calculation_time: float, composite_scores: Dict):
        """Alert koşullarını kontrol et"""
        # Calculation time alert
        if calculation_time > self.alert_thresholds['max_calculation_time']:
            self._send_alert(f"PERFORMANCE: Slow calculation for {symbol}: {calculation_time:.2f}s")
        
        # Confidence alert
        for score_name, score_data in composite_scores.items():
            if score_data.get('confidence', 1.0) < self.alert_thresholds['min_confidence']:
                self._send_alert(f"CONFIDENCE: Low confidence for {symbol}.{score_name}: {score_data['confidence']:.2f}")
    
    def _send_alert(self, message: str):
        """Alert gönder"""
        logger.warning(f"ALERT: {message}")
        # Buraya email, slack, vs. entegrasyonu eklenebilir
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Performance istatistiklerini getir"""
        calculation_times = self.performance_metrics['calculation_times']
        if not calculation_times:
            return {}
        
        total_requests = sum(
            rates['success'] + rates['failure'] 
            for rates in self.performance_metrics['error_rates'].values()
        )
        
        total_success = sum(rates['success'] for rates in self.performance_metrics['error_rates'].values())
        error_rate = (total_requests - total_success) / total_requests if total_requests > 0 else 0
        
        return {
            'avg_calculation_time': np.mean(calculation_times) if calculation_times else 0,
            'max_calculation_time': max(calculation_times) if calculation_times else 0,
            'min_calculation_time': min(calculation_times) if calculation_times else 0,
            'total_requests': total_requests,
            'success_rate': 1 - error_rate,
            'error_rate': error_rate,
            'cache_efficiency': self.performance_metrics['cache_efficiency']
        }
        
    
    
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
        