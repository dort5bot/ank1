# analysis/analysis_core.py

import os
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
# en üstte
from .analysis_schema_manager import (
    load_analysis_schema,
    load_module_run_function,
    resolve_module_path,   # ✅ yeni eklendi
    AnalysisSchema,
    Module,
    CircuitBreaker  # CircuitBreaker import edilmeli
)

from .analysis_base_module import BaseAnalysisModule
from .analysis_base_module import legacy_compatible
from analysis.composite.composite_engine import CompositeScoreEngine

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
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


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
        self._lock = asyncio.Lock()
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
        
        # birleşik skorlar
        self.composite_engine = CompositeScoreEngine(self)
        
        self._initialized = True
        logger.info("AnalysisAggregator initialized successfully")


    # ✅ 1️- Lock erişim fonksiyonu — üst kısma (helper fonksiyonların yanına)
    @asynccontextmanager
    async def _get_module_lock(self, module_name: str):
        if module_name not in self._execution_locks:
            self._execution_locks[module_name] = asyncio.Lock()
        lock = self._execution_locks[module_name]
        async with lock:
            yield

    

    # ✅ 2️- Aggregator başlatma / durdurma metodları — ortalara (lifecycle management bölümü) 
    async def start(self):
        if self._is_running:
            return
        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("Aggregator started")

    async def stop(self):
        if not self._is_running:
            return
        self._is_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                logger.info("Aggregator cleanup task was cancelled")
        logger.info("Aggregator stopped")




    

    def _get_cache_key(
        self, 
        symbol: str, 
        module_name: str, 
        priority: Optional[str] = None, 
        user_id: Optional[str] = None
    ) -> str:
        """Sembol + modül + öncelik + kullanıcı bazlı cache anahtarı üretir"""
        normalized_name = os.path.splitext(os.path.basename(module_name))[0].lower()
        # Güvenli string: None değerlerini 'unknown' veya 'anon' olarak fallback yapar
        key_string = f"{(symbol or 'unknown').upper()}:{normalized_name}:{priority or 'default'}:{user_id or 'anon'}"
        return hashlib.md5(key_string.encode()).hexdigest()





    async def _load_module_function(self, module_file: str):
        """Modül fonksiyonunu cache'li şekilde yükler (normalize edilmiş path ile)"""
        normalized_key = os.path.splitext(os.path.basename(module_file))[0].lower()
        cache_key = f"module_{normalized_key}"

        if cache_key in self._module_cache:
            self._cache_hits += 1
            return self._module_cache[cache_key]

        self._cache_misses += 1
        try:
            resolved_path = resolve_module_path(module_file)
            
            # ✅ Güvenli modül yolu kontrolü
            allowed_prefix = os.path.abspath("analysis/modules/")
            if not resolved_path.startswith(allowed_prefix):
                raise PermissionError(f"Unauthorized module path: {resolved_path}")

            
            run_function = load_module_run_function(resolved_path)
            self._module_cache[cache_key] = run_function
            return run_function
        except (ImportError, AttributeError, FileNotFoundError) as e:
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

        # run_single_analysis() — instance key normalize
        instance_key = os.path.splitext(os.path.basename(module.file))[0].lower()

        # Circuit breaker
        if instance_key not in self._circuit_breakers:
            self._circuit_breakers[instance_key] = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=30,
                expected_exception=(Exception,)
            )

        async def execute_analysis():
            async with self._get_module_lock(instance_key):
                logger.info(f"Starting analysis: {module.name} ({module.file}) for {symbol}")
                
                if instance_key in self._module_instances:
                    module_instance = self._module_instances[instance_key]
                    if hasattr(module_instance, "compute_metrics"):
                        try:
                            analysis_data = await module_instance.compute_metrics(symbol, priority)
                            return analysis_data
                        except Exception as e:
                            logger.warning(f"Module instance failed, fallback used: {e}")

                # fallback: eski run fonksiyonu
                run_function = await self._load_module_function(module.file)
                analysis_data = await run_function(symbol=symbol, priority=priority)

                if not isinstance(analysis_data, dict):
                    raise ValueError(f"Invalid result type: {type(analysis_data)}")

                return analysis_data
        
        
        
        
        
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
                logger.info(f"[{module.name}] ({module.file}) → COMPLETED in {result.execution_time:.2f}s")
            else:
                logger.warning(f"[{module.name}] ({module.file}) → {result.status.value.upper()} ({result.error or 'no error info'})")

        return result

    # ✅ 3️- Toplu çalışma metodu — alt kısma (analiz çalıştırma metodlarının yanına)
    async def run_all(self, symbol: str, priority: Optional[str] = None):
        if not self.schema:
            self.schema = load_analysis_schema()
        
        start = time.time()
        
        # ✅ Eşzamanlı çalışan görevleri sınırla
        sem = asyncio.Semaphore(6)
        async def safe_run(m):
            async with sem:
                return await self.run_single_analysis(m, symbol, priority)

        # Tüm modülleri güvenli şekilde sıraya koy
        tasks = [self.run_single_analysis(m, symbol, priority) for m in self.schema.modules]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 🔹 HATA YAKALAMA BLOĞU
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Module failed during run_all: {res}", exc_info=True)

        valid_results = [r for r in results if isinstance(r, AnalysisResult)]

        return AggregatedResult(
            symbol=symbol,
            results=valid_results,
            total_execution_time=time.time() - start,
            success_count=sum(1 for r in valid_results if r.status == AnalysisStatus.COMPLETED),
            failed_count=sum(1 for r in valid_results if r.status == AnalysisStatus.FAILED)
        )


    # birleşik
    # analysis_core.py (mevcut aggregator)
    async def get_comprehensive_analysis(self, symbol: str):
        # Mevcut modül analizleri
        module_results = await self.run_all_analyses(symbol)
        
        # Bileşik skorlar
        composite_scores = await self.composite_engine.calculate_composite_scores(symbol)
        
        return {
            'symbol': symbol,
            'module_analyses': module_results,
            'composite_scores': composite_scores['composite_scores'],
            'summary': self._generate_summary(composite_scores),
            'timestamp': composite_scores['timestamp']
        }
    
    async def get_trend_strength(self, symbol: str):
        """Sadece trend strength skoru al"""
        return await self.composite_engine.calculate_single_score('trend_strength', symbol)

    
    #
    async def _periodic_cleanup(self):
        """Geliştirilmiş resource cleanup"""
        import gc

        while self._is_running:
            try:
                await asyncio.sleep(300)

                current_time = time.time()

                # Result cache cleanup
                for key in list(self._result_cache.keys()):
                    result = self._result_cache[key]
                    if current_time - result.created_at > 600:
                        del self._result_cache[key]

                # Module instance cleanup - SYNTAX HATASI DÜZELTİLDİ
                if self.schema:  # schema kontrolü eklendi
                    valid_module_keys = [
                        self._get_module_instance_key(m.file)
                        for m in self.schema.modules
                    ]

                    for module_name in list(self._module_instances.keys()):
                        if module_name not in valid_module_keys:
                            del self._module_instances[module_name]

                # Performance tracking cleanup
                if len(self._execution_times) > 1000:
                    self._execution_times = self._execution_times[-500:]

                gc.collect()
                logger.debug("Cleanup completed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {str(e)}")

    def _get_module_instance_key(self, module_file: str) -> str:
        """Benzersiz instance key oluştur"""
        base_name = os.path.splitext(os.path.basename(module_file))[0]
        return f"{base_name}_{hashlib.md5(module_file.encode()).hexdigest()[:8]}"
        
    
    
    
    # analysis_core.py'de health check geliştirmesi
    async def comprehensive_health_check(self):
        """Kapsamlı sistem sağlık kontrolü"""
        checks = {
            "module_health": await self._check_module_health(),
            "cache_health": self._check_cache_health(),
            "memory_usage": self._get_memory_usage(),
            "api_connectivity": await self._check_api_connectivity()
        }
        return all(checks.values()), checks
    
    



# analysis_core.py'ye eklenmeli
class UserAwareAnalysisAggregator(AnalysisAggregator):
    def __init__(self):
        super().__init__()
        self._user_sessions: Dict[str, UserSession] = {}
    
    async def run_analysis_for_user(self, user_id: str, symbol: str, modules: List[str]):
        """Kullanıcı bazlı analiz çalıştırma"""
        user_session = self._get_user_session(user_id)
        async with user_session.lock:
            return await self._run_user_analysis(user_session, symbol, modules)


# analysis_core.py'ye eklenmeli
class PerformanceOptimizedAggregator(AnalysisAggregator):
    async def run_priority_batch(self, symbols: List[str], priority_modules: List[str]):
        """Öncelikli batch processing"""
        semaphore = asyncio.Semaphore(10)  # Concurrent limit
        
        async def process_symbol(symbol):
            async with semaphore:
                tasks = []
                for module_name in priority_modules:
                    module = self._get_module(module_name)
                    task = self.run_single_analysis(module, symbol, "high")
                    tasks.append(task)
                return await asyncio.gather(*tasks, return_exceptions=True)
        
        return await asyncio.gather(*[process_symbol(s) for s in symbols])



# Çoklu Kullanıcı ve Modül Performansı class EnhancedAnalysisAggregator+ class UserLimit:
class EnhancedAnalysisAggregator(AnalysisAggregator):
    def __init__(self):
        super().__init__()
        self._user_limits: Dict[str, UserLimit] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        
    async def run_analysis_for_user(self, user_id: str, symbol: str, module_names: List[str], 
                                  priority: Optional[str] = None) -> AggregatedResult:
        """Kullanıcı bazlı limitli analiz"""
        # Kullanıcı limitlerini kontrol et
        user_limit = self._get_user_limit(user_id)
        if not user_limit.can_execute(len(module_names)):
            raise HTTPException(status_code=429, detail="Too many requests")
        
        # Concurrent limit semaphore
        semaphore = self._get_semaphore(user_id)
        async with semaphore:
            modules = [m for m in self.schema.modules if m.name in module_names]
            tasks = [self.run_single_analysis(m, symbol, priority) for m in modules]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            user_limit.record_execution(len(modules))
            
            valid_results = [r for r in results if isinstance(r, AnalysisResult)]
            return AggregatedResult(
                symbol=symbol,
                results=valid_results,
                total_execution_time=0,  # Hesaplanacak
                success_count=sum(1 for r in valid_results if r.status == AnalysisStatus.COMPLETED),
                failed_count=sum(1 for r in valid_results if r.status == AnalysisStatus.FAILED)
            )
    
    def _get_user_limit(self, user_id: str) -> 'UserLimit':
        if user_id not in self._user_limits:
            self._user_limits[user_id] = UserLimit(
                max_concurrent=5,  # Aynı anda max 5 modül
                max_per_minute=30,  # Dakikada max 30 modül
                max_modules_per_request=10  # Tek requestte max 10 modül
            )
        return self._user_limits[user_id]
    
    def _get_semaphore(self, user_id: str) -> asyncio.Semaphore:
        if user_id not in self._semaphores:
            self._semaphores[user_id] = asyncio.Semaphore(3)  # Kullanıcı başına 3 concurrent
        return self._semaphores[user_id]

@dataclass
class UserLimit:
    max_concurrent: int
    max_per_minute: int 
    max_modules_per_request: int
    _current_minute: int = 0
    _minute_count: int = 0
    
    def can_execute(self, module_count: int) -> bool:
        current_minute = time.time() // 60
        if current_minute != self._current_minute:
            self._current_minute = current_minute
            self._minute_count = 0
            
        return (module_count <= self.max_modules_per_request and 
                self._minute_count + module_count <= self.max_per_minute)
    
    def record_execution(self, module_count: int):
        self._minute_count += module_count



# Global aggregator instance
aggregator = AnalysisAggregator()

async def get_aggregator() -> AnalysisAggregator:
    """Dependency injection için aggregator instance'ı"""
    if not aggregator._is_running:
        await aggregator.start()
    return aggregator
    


#------------------------------
# Kullanım - birleşik skor
aggregator = AnalysisAggregator()
result = await aggregator.get_trend_strength("BTCUSDT")

print(f"Trend Strength: {result['score']}")
print(f"Signal: {result['signal']}")
print(f"Confidence: {result['confidence']}")
