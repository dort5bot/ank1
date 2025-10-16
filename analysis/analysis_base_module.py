# analysis/analysis_base_module.py
"""
Base Analysis Module Abstract Class
===================================
Tüm analiz modülleri bu sınıftan türemeli.
Önceki: base_analysis_module.py
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)



class BaseAnalysisModule(ABC):
    """Analiz modülleri için abstract base class"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # default config safe
        self.config = config or {}
        self.module_name = self.__class__.__name__
        self.version = getattr(self, 'version', "1.0.0")
        self.dependencies: List[str] = getattr(self, 'dependencies', [])
        # Performance tracking
        self._execution_times: List[float] = []
        self._success_count: int = 0
        self._error_count: int = 0

    
    
    
    @abstractmethod
    async def compute_metrics(self, symbol: str, priority: Optional[str] = None) -> Dict[str, Any]:
        """
        Ana metrik hesaplama metodu
        
        Args:
            symbol: Analiz yapılacak sembol (Örnek: "BTCUSDT")
            priority: Öncelik seviyesi ("*", "**", "***")
            
        Returns:
            Dict: Analiz sonuçları
        """
        pass
    
    @abstractmethod 
    async def aggregate_output(self, metrics: Dict[str, float], symbol: str) -> Dict[str, Any]:
        """
        Metrikleri aggregate edip final sonuç üret
        
        Args:
            metrics: Hesaplanan metrikler
            symbol: Sembol
            
        Returns:
            Dict: Aggregate edilmiş sonuç
        """
        pass
    
    @abstractmethod
    def generate_report(self) -> Dict[str, Any]:
        """
        Modül durum raporu oluştur
        
        Returns:
            Dict: Rapor verisi
        """
        pass
    
    async def _fetch_ohlcv_data(self, symbol: str, interval: str = "1h", limit: int = 100) -> pd.DataFrame:
        """
        OHLCV verisi çekme - ortak utility
        
        Args:
            symbol: Sembol
            interval: Zaman aralığı
            limit: Veri limiti
            
        Returns:
            pd.DataFrame: OHLCV verisi
        """
        try:
            # Binance API implementasyonu buraya gelecek
            # Şimdilik mock data döndürüyoruz
            dates = pd.date_range(end=datetime.now(), periods=limit, freq=interval)
            data = pd.DataFrame({
                'open': np.random.random(limit) * 1000 + 50000,
                'high': np.random.random(limit) * 1000 + 50100,
                'low': np.random.random(limit) * 1000 + 49900,
                'close': np.random.random(limit) * 1000 + 50000,
                'volume': np.random.random(limit) * 1000
            }, index=dates)
            
            logger.debug(f"Fetched OHLCV data for {symbol}, shape: {data.shape}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data for {symbol}: {e}")
            raise
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Performans metriklerini getir
        
        Returns:
            Dict: Performans metrikleri
        """
        avg_time = sum(self._execution_times) / len(self._execution_times) if self._execution_times else 0
        success_rate = (
            self._success_count / (self._success_count + self._error_count) 
            if (self._success_count + self._error_count) > 0 else 0
        )
        
        return {
            "module": self.module_name,
            "version": self.version,
            "total_executions": len(self._execution_times),
            "average_execution_time": avg_time,
            "success_rate": success_rate,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "dependencies": self.dependencies
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Health check endpoint
        
        Returns:
            Dict: Health status
        """
        try:
            # Basit bir health check
            test_data = await self._fetch_ohlcv_data("BTCUSDT", limit=10)
            data_healthy = not test_data.empty and len(test_data) > 0
            
            return {
                "module": self.module_name,
                "status": "healthy" if data_healthy else "degraded",
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "data_available": data_healthy,
                "dependencies_healthy": True
            }
            
        except Exception as e:
            logger.error(f"Health check failed for {self.module_name}: {e}")
            return {
                "module": self.module_name,
                "status": "unhealthy",
                "version": self.version,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _record_execution(self, execution_time: float, success: bool = True):
        """Execution kaydı tut"""
        self._execution_times.append(execution_time)
        if success:
            self._success_count += 1
        else:
            self._error_count += 1
            
        # Execution times dizisini trim et
        if len(self._execution_times) > 1000:
            self._execution_times = self._execution_times[-500:]

# Backward compatibility için decorator
def legacy_compatible(cls):
    """
    Eski run() fonksiyonu ile uyumluluk sağlayan decorator
    """
    original_init = cls.__init__
    
    def new_init(self, config=None):
        if config is None:
            # Varsayılan config yükle
            config_module_name = f"config{cls.__name__.replace('Module', '')}"
            try:
                config = self._load_config(config_module_name)
            except:
                config = {}
        original_init(self, config)
    
    cls.__init__ = new_init
    
    # Eski run fonksiyonunu ekle
    async def run(symbol: str, priority: Optional[str] = None):
        instance = cls()
        return await instance.compute_metrics(symbol, priority)
    
    cls.run = staticmethod(run)
    return cls