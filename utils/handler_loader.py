"""
PROD GARANTİLERİ
✔️ Dosya yoksa oluşturmaz
✔️ Router yoksa hata verir
✔️ Router yanlış tipse hata verir
✔️ Aynı router iki kere eklenmez
✔️ Hatalı modül net şekilde raporlanır
✔️ Silent failure yok
Beklenen Handler Dosyası (ÖRNEK)

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

router = Router(name="start")

@router.message(Command("start"))
async def start_cmd(message: Message):
    await message.answer("Bot aktif")



"""

# handler_loader.py
import sys
import logging
import importlib.util
from pathlib import Path
from typing import Optional, Dict, Any, List, Set
from dataclasses import dataclass

from aiogram import Dispatcher, Router

logger = logging.getLogger("handler_loader")


# ---------------------------
# RESULT MODEL
# ---------------------------

@dataclass
class HandlerLoadResult:
    loaded: int = 0
    failed: int = 0
    skipped: int = 0
    total_files: int = 0
    errors: List[str] = None
    loaded_modules: List[str] = None

    def __post_init__(self):
        self.errors = self.errors or []
        self.loaded_modules = self.loaded_modules or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "loaded": self.loaded,
            "failed": self.failed,
            "skipped": self.skipped,
            "total_files": self.total_files,
            "errors": self.errors,
            "loaded_modules": self.loaded_modules,
        }


# ---------------------------
# CACHE
# ---------------------------

class HandlerCache:
    def __init__(self):
        self._loaded: Set[str] = set()

    def is_loaded(self, module: str) -> bool:
        return module in self._loaded

    def mark_loaded(self, module: str):
        self._loaded.add(module)

    def clear(self):
        self._loaded.clear()


# ---------------------------
# PROD HANDLER LOADER
# ---------------------------
    """
    PROD ONLY
    - Dosya / klasör oluşturmaz
    - Sadece var olan .py dosyaları yükler
    - Router zorunludur
    - Hariç tutulan listesi var
    """
    
class HandlerLoader:
    def __init__(
        self,
        dispatcher: Dispatcher,
        base_path: str = "handlers",
        handler_dirs: Optional[List[str]] = None,
        # 1️⃣ Hariç tutulacak dosya isimlerini buraya ekliyoruz
        exclude_files: Optional[List[str]] = None, 
    ):
        self.dispatcher = dispatcher
        self.base_path = Path(base_path)
        self.handler_dirs = handler_dirs or [
            "commands", "callbacks", "messages", "admin", "errors"
        ]
        # Varsayılan olarak engellenecek dosyalar
        self.exclude_files = exclude_files or [
            "report_format.py", "market_report.py","notes_module.py",
            "base.py"
        ]
        self.cache = HandlerCache()

    # -----------------------

    async def load_handlers(self) -> Dict[str, Any]:
        result = HandlerLoadResult()
        self.cache.clear()

        # handlers dizini yoksa
        if not self.base_path.exists():
            msg = f"handlers dizini yok: {self.base_path}"
            logger.critical(msg)
            result.errors.append(msg)
            return result.to_dict()

        # ---------------------------
        # 1️⃣ ROOT handlers (handlers/*.py)
        # ---------------------------
        for file in self.base_path.glob("*.py"):
            if file.name.startswith("__"):
                continue

            result.total_files += 1
            module_name = f"handlers.{file.stem}"
            await self._load_single_module(module_name, file, result)

        # ---------------------------
        # 2️⃣ ALT KLASÖR handlers
        # ---------------------------
        for directory in self.handler_dirs:
            dir_path = self.base_path / directory

            if not dir_path.exists():
                logger.warning(f"⏭️ Klasör yok, atlandı: {dir_path}")
                continue

            for file in dir_path.glob("*.py"):
                if file.name.startswith("__"):
                    continue

                result.total_files += 1
                module_name = f"handlers.{directory}.{file.stem}"
                await self._load_single_module(module_name, file, result)

        logger.info(
            f"Handler yükleme tamamlandı | "
            f"Loaded: {result.loaded} | "
            f"Failed: {result.failed} | "
            f"Skipped: {result.skipped}"
        )

        return result.to_dict()

    # -----------------------
    async def _load_single_module(
        self,
        module_name: str,
        file_path: Path,
        result: HandlerLoadResult,
    ):
        # 2️⃣ Dosya ismini kontrol et (Örn: report_format.py)
        if file_path.name in self.exclude_files:
            logger.debug(f"⏭️ Hariç tutuldu (Exclude list): {file_path.name}")
            # result.skipped += 1  # İsteğe bağlı istatistiğe ekle
            return

        if self.cache.is_loaded(module_name):
            logger.debug(f"⏭️ Zaten yüklü: {module_name}")
            result.skipped += 1
            return

        try:
            module = self._import_module(module_name, file_path)
            self._register_router(module, module_name, result)
        except Exception as e:
            logger.error(f"❌ {module_name} yüklenemedi: {e}", exc_info=True)
            result.failed += 1
            result.errors.append(f"{module_name}: {e}")

    # -----------------------

    def _import_module(self, module_name: str, file_path: Path):
        if module_name in sys.modules:
            del sys.modules[module_name]

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            raise ImportError("spec oluşturulamadı")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    # -----------------------

    def _register_router(
        self,
        module,
        module_name: str,
        result: HandlerLoadResult,
    ):
        if not hasattr(module, "router"):
            raise RuntimeError("router tanımı yok")

        router = module.router

        if not isinstance(router, Router):
            raise TypeError(f"router Router değil: {type(router).__name__}")

        if router in getattr(self.dispatcher, "_routers", []):
            logger.warning(f"⚠️ Router zaten ekli: {module_name}")
            result.skipped += 1
            return

        self.dispatcher.include_router(router)
        self.cache.mark_loaded(module_name)

        result.loaded += 1
        result.loaded_modules.append(module_name)

        router_name = getattr(router, "name", "unnamed")
        logger.info(f"✅ Router yüklendi: {router_name} ({module_name})")


# ---------------------------
# PUBLIC INIT
# ---------------------------

async def initialize_handlers(dispatcher: Dispatcher) -> Dict[str, Any]:
    loader = HandlerLoader(dispatcher)
    return await loader.load_handlers()
