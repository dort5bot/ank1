# analysis/analysis_core.py
import asyncio
import logging
import importlib.util
import time
import hashlib
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass
from contextlib import asynccontextmanager

# Schema imports
from .analysis_schema_manager import (
    load_analysis_schema, 
    load_module_run_function, 
    AnalysisSchema, 
    Module,
    CircuitBreaker  # CircuitBreaker import edilmeli
)

from .analysis_base_module import BaseAnalysisModule
from .analysis_base_module import legacy_compatible

# Configure logging
logger = logging.getLogger(__name__)

class AnalysisPriority(Enum):
    BASIC = "basic"
    PRO = "pro" 
    EXPERT = "expert"

class AnalysisStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class AnalysisResult:
    module_name: str
    command: str
    status: AnalysisStatus
    data: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None
    priority: Optional[str] = None

@dataclass
class AggregatedResult:
    symbol: str
    results: List[AnalysisResult]
    total_execution_time: float
    success_count: int
    failed_count: int
    overall_score: Optional[float] = None
    # Cache için timestamp ekle
    _created_at: float = None
    
    def __post_init__(self):
        if self._created_at is None:
            self._created_at = time.time()


class AnalysisAggregator:
    _instance: Optional['AnalysisAggregator'] = None
    _lock: asyncio.Lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.schema: Optional[AnalysisSchema] = None
        self._module_cache: Dict[str, Any] = {}
        self._result_cache: Dict[str, AggregatedResult] = {}
        self._execution_locks: Dict[str, asyncio.Lock] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running: bool = False
        
        # Performance monitoring
        self._execution_times: List[float] = []
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        
        # Yeni özellikler
        self._module_instances: Dict[str, Any] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        self._initialized = True
        logger.info("AnalysisAggregator initialized successfully")

    def _get_cache_key(self, symbol: str, module_name: str, priority: Optional[str] = None, user_level: Optional[str] = None) -> str:
        """Geliştirilmiş cache key - modül bazlı"""
        key_parts = [
            symbol.strip().upper(),
            module_name,
            priority.strip() if priority else "default",
            user_level.strip().lower() if user_level else "default"
        ]
        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

    async def _load_module_function(self, module_file: str):
        """Modül fonksiyonunu cache'li şekilde yükle"""
        cache_key = f"module_{module_file}"
        
        if cache_key in self._module_cache:
            self._cache_hits += 1
            return self._module_cache[cache_key]
        
        self._cache_misses += 1
        try:
            run_function = load_module_run_function(module_file)
            self._module_cache[cache_key] = run_function
            return run_function
        except (ImportError, AttributeError) as e:
            logger.error(f"Module load failed for {module_file}: {str(e)}")
            raise

    async def run_single_analysis(
        self, 
        module: Module, 
        symbol: str, 
        priority: Optional[str] = None
    ) -> AnalysisResult:
        """Circuit breaker ile geliştirilmiş analiz çalıştırma"""
        start_time = time.time()
        result = AnalysisResult(
            module_name=module.name,
            command=module.command,
            status=AnalysisStatus.PENDING,
            data={},
            execution_time=0.0,
            priority=priority
        )
        
        # Circuit breaker oluştur (eğer yoksa)
        if module.name not in self._circuit_breakers:
            self._circuit_breakers[module.name] = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=30,
                expected_exception=(Exception,)
            )
        
        async def execute_analysis():
            """Ana analiz fonksiyonu"""
            try:
                async with self._get_module_lock(module.name):
                    logger.info(f"Starting analysis: {module.name} for {symbol}")
                    
                    # Modül instance kontrolü - safe approach
                    if module.name in self._module_instances:
                        try:
                            module_instance = self._module_instances[module.name]
                            if hasattr(module_instance, 'compute_metrics'):
                                analysis_data = await module_instance.compute_metrics(symbol, priority)
                                return analysis_data
                        except Exception as e:
                            logger.warning(f"Module instance failed, using fallback: {e}")
                    
                    # Fallback: eski run fonksiyonu
                    run_function = await self._load_module_function(module.file)
                    analysis_data = await run_function(symbol=symbol, priority=priority)
                    
                    # Result validation
                    if not isinstance(analysis_data, dict):
                        raise ValueError(f"Invalid result type: {type(analysis_data)}")
                    
                    return analysis_data
                    
            except asyncio.CancelledError:
                logger.warning(f"Analysis cancelled: {module.name}")
                raise
            except Exception as e:
                logger.error(f"Analysis execution failed for {module.name}: {str(e)}")
                raise
        
        async def fallback_analysis():
            """Fallback analiz fonksiyonu"""
            logger.warning(f"Using fallback analysis for {module.name} - symbol: {symbol}")
            return {
                "score": 0.5, 
                "status": "fallback", 
                "components": {},
                "fallback_reason": "Circuit breaker activated",
                "timestamp": time.time(),
                "module": module.name
            }
        
        try:
            result.status = AnalysisStatus.RUNNING
            
            # Circuit breaker ile analizi çalıştır
            result.data = await self._circuit_breakers[module.name].execute_with_fallback(
                execute_analysis, 
                fallback_analysis
            )
            
            result.status = AnalysisStatus.COMPLETED
            
        except asyncio.CancelledError:
            result.status = AnalysisStatus.CANCELLED
            result.error = "Analysis cancelled"
            logger.warning(f"Analysis cancelled: {module.name}")
            raise
            
        except Exception as e:
            result.status = AnalysisStatus.FAILED
            result.error = f"Analysis failed: {str(e)}"
            logger.error(f"Analysis failed for {module.name}: {str(e)}", exc_info=True)
            
        finally:
            result.execution_time = time.time() - start_time
            self._execution_times.append(result.execution_time)
            
            if result.status == AnalysisStatus.COMPLETED:
                logger.info(f"Analysis completed: {module.name} in {result.execution_time:.2f}s")
            else:
                logger.warning(f"Analysis {result.status.value}: {module.name}")
        
        return result

    async def _periodic_cleanup(self):
        """Geliştirilmiş cache cleanup"""
        logger.info("Starting periodic cleanup task")
        
        while self._is_running:
            try:
                await asyncio.sleep(300)  # 5 dakikada bir
                
                # Eski cache entry'lerini temizle
                current_time = time.time()
                keys_to_remove = []
                
                for key, result in self._result_cache.items():
                    if current_time - result._created_at > 600:  # 10 dakika
                        keys_to_remove.append(key)
                
                for key in keys_to_remove:
                    del self._result_cache[key]
                
                # Module cache cleanup (24 saat)
                module_keys_to_remove = []
                for key in self._module_cache.keys():
                    # Basit LRU benzeri cleanup
                    if len(self._module_cache) > 50:  # Max 50 modül cache'le
                        module_keys_to_remove.append(key)
                        break
                
                for key in module_keys_to_remove:
                    del self._module_cache[key]
                
                # Execution times trim
                if len(self._execution_times) > 1000:
                    self._execution_times = self._execution_times[-500:]
                
                logger.debug(f"Cleanup completed: {len(keys_to_remove)} cache, {len(module_keys_to_remove)} module entries removed")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {str(e)}")

# Global aggregator instance
aggregator = AnalysisAggregator()

async def get_aggregator() -> AnalysisAggregator:
    """Dependency injection için aggregator instance'ı"""
    if not aggregator._is_running:
        await aggregator.start()
    return aggregator