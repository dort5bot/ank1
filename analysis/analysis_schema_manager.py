# Module + EnhancedModule → AnalysisModule olarak birleştirildi.
# analysis_schema_manager.py
import os
import time
import logging
import importlib.util
from enum import Enum
from typing import Dict, Any, List, Optional, Literal, Union, Callable, Type, Tuple
from pydantic import BaseModel, validator
import yaml

from analysis.analysis_base_module import BaseAnalysisModule



logger = logging.getLogger(__name__)
ANALYSIS_BASE_PATH = os.path.join(os.getcwd(), "analysis")

# -----------------------------#
# 🧩 ENUM & TİP TANIMLARI
# -----------------------------#
PriorityLevel = Literal["*", "**", "***"]

class ModuleLifecycle(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"

class ParallelMode(str, Enum):
    BATCH = "batch"
    ASYNC = "async"
    STREAM = "stream"
    EVENT = "event"

class Metric(BaseModel):
    name: str
    priority: PriorityLevel
    
    # ✅ Ek validation
    @validator('name')
    def validate_metric_name(cls, v):
        if not v.replace('_', '').isalnum():
            raise ValueError('Metric name can only contain alphanumeric characters and underscores')
        return v.lower()


# -----------------------------#
# 📦 MODÜL ŞEMASI
# -----------------------------#
class AnalysisModule(BaseModel):
    name: str
    file: str
    command: str
    api_type: str
    endpoints: List[str]
    methods: List[Literal["GET", "POST", "PUT", "DELETE", "WebSocket"]]

    classical_metrics: Optional[List[Union[str, Metric]]] = []
    professional_metrics: Optional[List[Metric]] = []
    composite_metrics: Optional[List[str]] = []

    development_notes: Optional[str] = None
    objective: Optional[str] = None
    output_type: Optional[str] = None

    lifecycle: ModuleLifecycle = ModuleLifecycle.DEVELOPMENT
    parallel_mode: ParallelMode = ParallelMode.BATCH
    config_file: Optional[str] = None
    required_metrics: List[str] = []
    outputs: List[str] = []
    version: str = "1.0.0"
    dependencies: List[str] = []

    # ✅ YAML'den gelen ekstra alanlar:
    config: Optional[str] = None
    command_aliases: List[str] = []
    job_type: Optional[str] = None
    description: Optional[str] = None
    maintainer: Optional[str] = None

    class Config:
        extra = "ignore"  # ✅ Fazla alanlar hataya neden olmaz



class AnalysisSchema(BaseModel):
    modules: List[AnalysisModule]




# ---------Singleton Schema Manager--------------------#
class SchemaManager:
    _instance = None
    _schema = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_schema(cls) -> AnalysisSchema:
        if cls._schema is None:
            cls._schema = load_analysis_schema()
        return cls._schema
    
    @classmethod
    def reload_schema(cls):
        cls._schema = load_analysis_schema()
        


# -----------------------------#
# 🔧 HELPER: Dinamik yükleme
# -----------------------------#

def resolve_module_path(module_file: str) -> str:
    module_file = os.path.basename(module_file.strip())
    module_path = os.path.join(ANALYSIS_BASE_PATH, module_file)
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Module not found: {module_path}")
    return module_path


def load_python_module(module_path: str):
    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if not spec or not spec.loader:
        raise ImportError(f"Module could not be loaded: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_analysis_module_class(mod) -> Optional[Type[BaseAnalysisModule]]:
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and issubclass(attr, BaseAnalysisModule) and attr != BaseAnalysisModule:
            return attr
    return None


def load_module_run_function(module_file: str) -> Callable:
    module_path = resolve_module_path(module_file)
    mod = load_python_module(module_path)

    if hasattr(mod, "run"):
        return mod.run

    cls = find_analysis_module_class(mod)
    if cls:
        async def run_wrapper(symbol: str, priority: Optional[str] = None):
            instance = cls()
            return await instance.compute_metrics(symbol, priority)
        return run_wrapper

    raise AttributeError(f"'run()' veya BaseAnalysisModule sınıfı bulunamadı: {module_file}")


# -----------------------------#
# 🏭 ModuleFactory + cache
# -----------------------------#

class ModuleFactory:
    _module_cache: Dict[str, Type[BaseAnalysisModule]] = {}
    
    @staticmethod
    def create_module(module_name: str, config: Dict[str, Any]) -> BaseAnalysisModule:
        if module_name in ModuleFactory._module_cache:
            cls = ModuleFactory._module_cache[module_name]
            return cls(config)
            
        module_file = f"analysis_{module_name.lower()}.py"
        module_path = resolve_module_path(module_file)
        mod = load_python_module(module_path)
        cls = find_analysis_module_class(mod)
        
        if cls:
            ModuleFactory._module_cache[module_name] = cls
            return cls(config)
            
        raise AttributeError(f"Module class not found in {module_file}")


# -----------------------------#
# 🧱 Circuit Breaker
# -----------------------------#

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 30,
                 expected_exception: Tuple[Type[Exception], ...] = (Exception,)):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failures = 0
        self.last_failure_time = 0
        self.state = "CLOSED"

    async def execute_with_fallback(self, command: callable, fallback: callable):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker half-open, testing recovery")
            else:
                logger.warning("Circuit breaker OPEN, using fallback")
                return await fallback()

        try:
            result = await command()
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            logger.warning(f"Circuit breaker failure: {e}")
            return await fallback()

    def _on_success(self):
        self.failures = 0
        self.state = "CLOSED"
        logger.info("Circuit breaker reset to CLOSED")

    def _on_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            logger.error("Circuit breaker OPENED")
            
    # ✅ Ek: Context manager desteği
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, self.expected_exception):
            self._on_failure()
        else:
            self._on_success()
    
    # Kullanım örneği:
    # async with circuit_breaker:
    #     result = await some_operation()


# -----------------------------#
# 📥 Yükleyici + Error Handling
# -----------------------------#

def load_analysis_schema(yaml_path: str = "analysis/analysis_metric_schema.yaml") -> AnalysisSchema:
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        # ✅ Validation öncesi basit check
        if not data or "modules" not in data:
            raise ValueError("Invalid schema format: 'modules' key missing")
            
        return AnalysisSchema(**data)
    except FileNotFoundError:
        logger.error(f"Schema file not found: {yaml_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error: {e}")
        raise


# -----------------------------#
# 🔎 Filtreleme Fonksiyonları
# -----------------------------#

def filter_modules_by_priority(schema: AnalysisSchema, priority: PriorityLevel) -> List[AnalysisModule]:
    return [m for m in schema.modules if any(metric.priority == priority for metric in m.professional_metrics or [])]


def get_metrics_by_priority(module: AnalysisModule, priority: PriorityLevel) -> List[Metric]:
    return [m for m in (module.professional_metrics or []) if m.priority == priority]



def get_module_by_field(schema: AnalysisSchema, field: str, value: str) -> Optional[AnalysisModule]:
    return next((m for m in schema.modules if getattr(m, field, None) == value), None)

# get_module_by_field İçin Yardımcı Metodlar ek
def get_module_by_name(schema: AnalysisSchema, name: str) -> Optional[AnalysisModule]:
    return get_module_by_field(schema, "name", name)

def get_module_by_command(schema: AnalysisSchema, command: str) -> Optional[AnalysisModule]:
    return get_module_by_field(schema, "command", command)

def get_module_by_file(schema: AnalysisSchema, file: str) -> Optional[AnalysisModule]:
    return get_module_by_field(schema, "file", file)



def get_modules_by_lifecycle(schema: AnalysisSchema, lifecycle: ModuleLifecycle) -> List[AnalysisModule]:
    return [m for m in schema.modules if m.lifecycle == lifecycle]


def get_module_dependencies(schema: AnalysisSchema, module_name: str) -> List[str]:
    module = get_module_by_field(schema, "name", module_name)
    return module.dependencies if module else []


# -----------------------------#
# 👤 Kullanıcı Seviyesi
# -----------------------------#

USER_LEVEL_PRIORITY = {
    "basic": "*",
    "pro": "**",
    "expert": "***"
}

def get_modules_for_user_level(schema: AnalysisSchema, level: str) -> List[AnalysisModule]:
    priority = USER_LEVEL_PRIORITY.get(level.lower())
    if priority:
        return filter_modules_by_priority(schema, priority)
    return []


# -----------------------------#
# 🧪 ÖRNEK TEST
# -----------------------------#

if __name__ == "__main__":
    schema = load_analysis_schema()

    print("📊 Tüm modüller:")
    for module in schema.modules:
        print(f" - [{module.command}] {module.name} (file: {module.file})")

    print("\n🎯 *** öncelikli metriklere sahip modüller:")
    for mod in filter_modules_by_priority(schema, "***"):
        metrics = get_metrics_by_priority(mod, "***")
        print(f"🔍 {mod.name} ({mod.command})")
        for m in metrics:
            print(f"   - {m.name} ({m.priority})")

    print("\n👤 Pro seviye kullanıcı modülleri:")
    for mod in get_modules_for_user_level(schema, "pro"):
        print(f" - {mod.name} ({mod.command})")
