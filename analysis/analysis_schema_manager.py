# analysis_schema_manager.py  yükleyici yapısına bire bir uyumlu bir şema tanımlar ve yükleyiciyi içerir
# Geliştirilmiş versiyon: priority filtresi + kullanıcı seviyesi desteği + modül/metrik arama


from enum import Enum
from typing import Dict, Any, List, Optional, Literal, Union, Callable
from pydantic import BaseModel
import yaml
import importlib.util
import os
from analysis.base_module import BaseAnalysisModule

# Global analiz klasör yolu
ANALYSIS_BASE_PATH = os.path.join(os.getcwd(), "analysis")


# --- Şema Tanımları ---
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


class Metric(BaseModel):
    name: str
    priority: PriorityLevel

# analysis_metric_schema.yaml için modül
class Module(BaseModel):
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


class AnalysisSchema(BaseModel):
    modules: List[Module]


class EnhancedModule(Module):
    lifecycle: ModuleLifecycle = ModuleLifecycle.DEVELOPMENT
    parallel_mode: ParallelMode = ParallelMode.BATCH
    config_file: Optional[str] = None
    required_metrics: List[str] = []
    outputs: List[str] = []
    version: str = "1.0.0"
    dependencies: List[str] = []  # Metric dependency graph için
    


# Factory pattern implementation
class ModuleFactory:
    @staticmethod
    def create_module(module_name: str, config: Dict[str, Any]) -> 'BaseAnalysisModule':
        """Modül factory'si - dinamik olarak modül oluşturur"""
        module_mapping = {
            "trend": "TrendModule",
            "volatility": "VolatilityModule", 
            "sentiment": "SentimentModule"
        }
        
        if module_name not in module_mapping:
            raise ValueError(f"Unknown module: {module_name}")
        
        # Dinamik import
        module_file = f"analysis_{module_name.lower()}.py"
        run_function = load_module_run_function(module_file)
        
        # BaseAnalysisModule instance'ı oluştur
        return run_function(config)

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=60, expected_exception=Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def execute_with_fallback(self, main_func, fallback_func):
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.info(f"Circuit breaker half-open for recovery")
            else:
                logger.warning(f"Circuit breaker OPEN, using fallback")
                return await fallback_func()
        
        try:
            result = await main_func()
            
            # Başarılı execution - state'i resetle
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info(f"Circuit breaker reset to CLOSED")
            
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
            
            logger.warning(f"Circuit breaker failure count: {self.failure_count}")
            
            # Fallback fonksiyonunu çağır
            return await fallback_func()


# --- Yükleyici Fonksiyon ---

def load_analysis_schema(yaml_path: str = "analysis/analysis_metric_schema.yaml") -> AnalysisSchema:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AnalysisSchema(**data)


# --- Filtreleme & Yardımcı Fonksiyonlar ---

def filter_modules_by_priority(schema: AnalysisSchema, priority: PriorityLevel) -> List[Module]:
    """
    Verilen priority seviyesine göre modülleri filtreler (sadece ilgili metrik içerenler döner)
    """
    filtered = []
    for module in schema.modules:
        if any(m.priority == priority for m in module.professional_metrics or []):
            filtered.append(module)
    return filtered

def get_metrics_by_priority(module: Module, priority: PriorityLevel) -> List[Metric]:
    """
    Bir modül içindeki belirli öncelikteki metrikleri döner
    """
    return [m for m in (module.professional_metrics or []) if m.priority == priority]

def get_module_by_command(schema: AnalysisSchema, command: str) -> Optional[Module]:
    return next((m for m in schema.modules if m.command == command), None)

def get_module_by_file(schema: AnalysisSchema, file: str) -> Optional[Module]:
    return next((m for m in schema.modules if m.file == file), None)

def get_module_by_name(schema: AnalysisSchema, name: str) -> Optional[Module]:
    return next((m for m in schema.modules if m.name == name), None)

def get_modules_by_lifecycle(schema: AnalysisSchema, lifecycle: ModuleLifecycle) -> List[EnhancedModule]:
    """Yaşam döngüsüne göre modülleri filtrele"""
    return [m for m in schema.modules if getattr(m, 'lifecycle', ModuleLifecycle.DEVELOPMENT) == lifecycle]

def get_module_dependencies(schema: AnalysisSchema, module_name: str) -> List[str]:
    """Modül bağımlılıklarını getir"""
    module = get_module_by_name(schema, module_name)
    return getattr(module, 'dependencies', []) if module else []


# her analiz modül dosyasının içinden run() fonksiyonunu otomatik yükler.
# analysis_schema_manager.py - Yükleyici 
# analysis/analysis_schema_manager.py
# ============================================================
# Modül: Schema + Dynamic Loader
# Standart: Mutlak yol çözümü, class/fonksiyon destekli yükleme
# ============================================================


# Global analiz klasör yoluna göre işlemler
def resolve_module_path(module_file: str) -> str:
    """
    Normalize edilmiş mutlak dosya yolu döndürür.
    Yalnızca dosya adı verilmişse analysis klasörü altından çözer.
    """
    module_file = os.path.basename(module_file.strip())
    module_path = os.path.join(ANALYSIS_BASE_PATH, module_file)

    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Module not found: {module_path}")

    return module_path


def load_module_run_function(module_file: str) -> Callable:
    """
    Geliştirilmiş modül yükleyici — hem BaseAnalysisModule class'ı
    hem legacy run() fonksiyonunu destekler.
    """
    module_path = resolve_module_path(module_file)
    module_name = os.path.splitext(os.path.basename(module_file))[0]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if not spec or not spec.loader:
        raise ImportError(f"Modül yüklenemedi: {module_file}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 1️⃣ Legacy run() fonksiyonu
    if hasattr(mod, "run"):
        return mod.run

    # 2️⃣ BaseAnalysisModule’dan türeyen sınıf
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if (isinstance(attr, type)
            and issubclass(attr, BaseAnalysisModule)
            and attr != BaseAnalysisModule):
            
            async def run_wrapper(symbol: str, priority: Optional[str] = None):
                instance = attr()
                return await instance.compute_metrics(symbol, priority)
            
            return run_wrapper

    raise AttributeError(
        f"{module_file} içinde 'run()' fonksiyonu veya BaseAnalysisModule class'ı bulunamadı."
    )




# --- Kullanıcı Seviyesi Filtresi ---

USER_LEVEL_PRIORITY = {
    "basic": "*",
    "pro": "**",
    "expert": "***"
}

def get_modules_for_user_level(schema: AnalysisSchema, level: str) -> List[Module]:
    """
    Kullanıcı seviyesine göre modülleri döner
    """
    priority = USER_LEVEL_PRIORITY.get(level.lower())
    if priority:
        return filter_modules_by_priority(schema, priority)
    return []


# --- Test Örneği ---

if __name__ == "__main__":
    schema = load_analysis_schema()

    print("📊 Tüm modüller:")
    for module in schema.modules:
        print(f" - [{module.command}] {module.name} (file: {module.file})")

    print("\n🎯 Önceliği *** olan metrikleri içeren modüller:")
    high_priority_modules = filter_modules_by_priority(schema, "***")
    for mod in high_priority_modules:
        metrics = get_metrics_by_priority(mod, "***")
        print(f"\n🔍 {mod.name} ({mod.command})")
        for m in metrics:
            print(f"   - {m.name} ({m.priority})")

    print("\n👤 Pro seviye kullanıcıya uygun modüller:")
    pro_mods = get_modules_for_user_level(schema, "pro")
    for mod in pro_mods:
        print(f" - {mod.name} ({mod.command})")




"""
🔧 Bundan Sonra Ne Yapabilirim?

Şimdi bu yapıyı kullanarak şunları kolayca ekleyebiliriz:
| Amaç                       | Ne Yapabilirim?                                             |
| -------------------------- | ----------------------------------------------------------- |
| 📡 API router              | FastAPI route'ları otomatik üretirim (`/trend?priority=**`) |
| 🧪 Test                    | Her modül için otomatik test iskeleti çıkarabilirim         |
| 🖥️ CLI                    | `python analyze.py --module /flow --priority=**` gibi       |
| 📊 UI menü                 | Streamlit/Dash için menüleri `priority` bazlı oluştururum   |
| 🧠 Sınıf Tabanlı Yorumlama | Her modüle özel analiz sınıfı oluşturma mantığını eklerim   |


analiz modüllerini merkezi olarak tanımlama, yükleme, filtreleme ve çalıştırma yapar
Sorumluluk: YAML şemasını yükleme, validasyon, filtreleme işlemleri
Neden ayrı?: Data access layer pattern - veri erişim mantığını soyutlama
Avantaj: Router ve core modüllerinden bağımsız çalışabilir

| Özellik                      | Açıklama                                             |
| ---------------------------- | ---------------------------------------------------- |
| 🧩 Tam `pydantic` uyumu      | `analysis_metric_schema.yaml` ile birebir eşleşir            |
| 🎛️ `priority` filtresi      | `*`, `**`, `***` seviyelerinde filtreleme fonksiyonu |
| 🚀 Kullanıcı seviyesi seçimi | "basic", "pro", "expert" gibi user level uyarlaması  |
| 🔍 Modül & metrik arama      | Komut, dosya ya da isimle modül bulma                |
| 🧪 Gelişmiş test örneği      | Modül & metrikleri filtreleyerek yazdırır            |
| 🧠 Genişletmeye hazır yapı   | API, CLI, UI ya da test framework için uygun         |

Analizleri özelleştirebilirsin
GET /regime → default (tümü)
GET /regime?priority=* → sadece hızlı/temel
GET /regime?priority=*** → yalnızca ileri düzey

⚙️ GEREKSİNİM

Her analiz modül dosyasında (örneğin tremo.py) şu fonksiyon tanımlı olmalı:

# örnek: analysis/tremo.py
async def run(symbol: str, priority: Optional[str] = None) -> dict:
    return {"score": 0.74, "symbol": symbol, "priority": priority}


Bu işlevin symbol ve priority parametresini alması gerekiyor.

"""