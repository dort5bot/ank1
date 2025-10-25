#  utils/handler_loader.py
"""
Handler Loader Module - Optimized for Telegram Bot
FIXED VERSION - Aiogram 3.x compatible
"""

import os
import asyncio
import logging
import importlib
import inspect
from typing import Dict, List, Any, Optional, Set, Type, Callable
from pathlib import Path
from dataclasses import dataclass

from aiogram import Dispatcher, Router
from aiogram.filters import Filter
from aiogram.types import Update

from utils.context_logger import get_context_logger

# Configure logger
logger = get_context_logger(__name__)

@dataclass
class HandlerLoadResult:
    """Handler loading result with detailed metrics."""
    loaded: int = 0
    failed: int = 0
    skipped: int = 0
    total_files: int = 0
    errors: List[str] = None
    loaded_handlers: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.loaded_handlers is None:
            self.loaded_handlers = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for logging."""
        return {
            'loaded': self.loaded,
            'failed': self.failed,
            'skipped': self.skipped,
            'total_files': self.total_files,
            'errors': self.errors,
            'loaded_handlers': self.loaded_handlers
        }

class HandlerCache:
    """Handler cache management with thread safety."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._loaded_modules: Set[str] = set()
    
    async def is_module_loaded(self, module_path: str) -> bool:
        """Check if module is already loaded."""
        async with self._lock:
            return module_path in self._loaded_modules
    
    async def mark_module_loaded(self, module_path: str):
        """Mark module as loaded."""
        async with self._lock:
            self._loaded_modules.add(module_path)
    
    async def clear_cache(self):
        """Clear handler cache."""
        async with self._lock:
            self._cache.clear()
            self._loaded_modules.clear()
            logger.info("✅ Handler cache cleared")

class HandlerLoader:
    """
    Dynamic handler loader with caching and validation.
    FIXED VERSION - No false Router detection
    """
    
    def __init__(self, dispatcher: Dispatcher, base_path: str = "handlers"):
        self.dispatcher = dispatcher
        self.base_path = Path(base_path)
        self.cache = HandlerCache()
        
        # Handler directories to scan
        self.handler_dirs = [
            "commands",
            "callbacks", 
            "messages",
            "errors",
            "admin"
        ]
        
        logger.info(f"🔄 HandlerLoader initialized with base path: {base_path}")
    
    async def load_handlers(self, dispatcher: Dispatcher) -> Dict[str, int]:
        """
        Load all handlers dynamically.
        
        Returns:
            Dict with loading statistics
        """
        result = HandlerLoadResult()
        
        try:
            # Clear previous cache to ensure fresh loading
            await self.cache.clear_cache()
            
            # Check if handlers directory exists
            if not self.base_path.exists():
                logger.warning(f"⚠️ Handlers directory not found: {self.base_path}")
                await self._create_default_handlers()
                return result.to_dict()
            
            # Load handlers from all directories
            for handler_dir in self.handler_dirs:
                dir_path = self.base_path / handler_dir
                await self._load_handlers_from_directory(dir_path, handler_dir, result, dispatcher)
            
            # Load root handlers
            await self._load_handlers_from_directory(self.base_path, "root", result, dispatcher)
            
            logger.info(f"✅ Handler loading completed: {result.loaded} loaded, {result.failed} failed")
            
        except Exception as e:
            logger.error(f"❌ Critical error in handler loading: {e}")
            result.failed += 1
            result.errors.append(f"Critical error: {str(e)}")
        
        return result.to_dict()
    
    async def _load_handlers_from_directory(self, dir_path: Path, category: str, 
                                          result: HandlerLoadResult, dispatcher: Dispatcher):
        """Load handlers from a specific directory."""
        if not dir_path.exists():
            logger.debug(f"📁 Handler directory not found, skipping: {dir_path}")
            return
        
        logger.info(f"📁 Loading handlers from: {category}")
        
        # Find all Python files in directory
        py_files = list(dir_path.glob("*.py"))
        result.total_files += len(py_files)
        
        for py_file in py_files:
            if py_file.name.startswith("__"):
                continue
                
            module_name = f"handlers.{category}.{py_file.stem}"
            await self._load_handler_module(module_name, py_file, result, dispatcher)
    
    async def _load_handler_module(self, module_name: str, file_path: Path, 
                                 result: HandlerLoadResult, dispatcher: Dispatcher):
        """Load and register handlers from a module."""
        try:
            # Check cache to avoid duplicate loading
            if await self.cache.is_module_loaded(module_name):
                logger.debug(f"📦 Module already loaded, skipping: {module_name}")
                result.skipped += 1
                return
            
            # Import module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                logger.warning(f"❌ Could not load spec for: {module_name}")
                result.failed += 1
                result.errors.append(f"Spec loading failed: {module_name}")
                return
            
            module = importlib.util.module_from_spec(spec)
            
            # Execute module
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                logger.error(f"❌ Error executing module {module_name}: {e}")
                result.failed += 1
                result.errors.append(f"Module execution failed: {module_name} - {str(e)}")
                return
            
            # ✅ FIXED: Doğru router kontrolü
            if hasattr(module, 'router'):
                router = module.router
                
                # ✅ CRITICAL FIX: Router INSTANCE kontrolü
                if isinstance(router, Router):
                    # Router'ı dispatcher'a ekle
                    try:
                        dispatcher.include_router(router)
                        await self.cache.mark_module_loaded(module_name)
                        result.loaded += 1
                        result.loaded_handlers.append(module_name)
                        
                        # Handler sayısını tahmin et (manuel counting)
                        handler_count = await self._count_router_handlers(router, module)
                        logger.info(f"✅ Loaded router '{getattr(router, 'name', 'unnamed')}' with ~{handler_count} handlers from {module_name}")
                        
                    except RuntimeError as e:
                        if "already attached" in str(e):
                            logger.warning(f"⚠️ Router already attached, skipping: {module_name}")
                            result.skipped += 1
                        else:
                            logger.error(f"❌ Router inclusion error for {module_name}: {e}")
                            result.failed += 1
                            result.errors.append(f"Router inclusion failed: {module_name} - {str(e)}")
                
                else:
                    # ❌ Bu bir Router INSTANCE değil
                    logger.error(f"❌ 'router' is not a Router instance in {module_name}")
                    logger.error(f"   Type: {type(router)}")
                    logger.error(f"   Value: {router}")
                    result.failed += 1
                    result.errors.append(f"Router is not instance: {module_name}")
                    
            else:
                logger.debug(f"⏭️ No router found in {module_name}")
                result.skipped += 1
                
        except ImportError as e:
            logger.error(f"❌ Import error for {module_name}: {e}")
            result.failed += 1
            result.errors.append(f"Import error: {module_name} - {str(e)}")
        except Exception as e:
            logger.error(f"❌ Unexpected error loading {module_name}: {e}")
            result.failed += 1
            result.errors.append(f"Unexpected error: {module_name} - {str(e)}")
    

    # handler_loader.py - _count_router_handlers iyileştirmesi
    async def _count_router_handlers(self, router, module) -> int:
        """Router'daki handler sayısını daha doğru tahmin et."""
        try:
            # Method 1: Router'ın içindeki handler'ları say
            if hasattr(router, 'sub_routers'):
                total_handlers = 0
                # Ana router ve sub router'ları kontrol et
                all_routers = [router] + router.sub_routers
                for r in all_routers:
                    # Message handler'ları
                    if hasattr(r, 'message'):
                        total_handlers += 1
                    # Callback handler'ları  
                    if hasattr(r, 'callback_query'):
                        total_handlers += 1
                    # Diğer handler türleri...
                return max(total_handlers, 1)
            
            # Method 2: Fallback - module source code'dan say
            source_code = inspect.getsource(module)
            decorator_count = source_code.count('@router.')
            return max(decorator_count, 1)
            
        except:
            return 1  # Fallback
        
    
    
    async def _create_default_handlers(self):
        """Create default handler structure if it doesn't exist."""
        logger.info("🏗️ Creating default handler structure...")
        
        try:
            self.base_path.mkdir(exist_ok=True)
            
            # Create handler directories
            for handler_dir in self.handler_dirs:
                (self.base_path / handler_dir).mkdir(exist_ok=True)
            
            # Create __init__.py files
            for py_init in self.base_path.rglob("__init__.py"):
                if not py_init.exists():
                    py_init.touch()
            
            logger.info("✅ Default handler structure created")
            
        except Exception as e:
            logger.error(f"❌ Failed to create default handler structure: {e}")
    
    async def clear_handler_cache(self):
        """Clear handler cache - used during reloads."""
        await self.cache.clear_cache()
        logger.info("✅ Handler cache cleared")
    
    async def get_loading_statistics(self) -> Dict[str, Any]:
        """Get detailed loading statistics."""
        return {
            'cache_size': len(self.cache._loaded_modules),
            'loaded_modules': list(self.cache._loaded_modules),
            'handler_dirs': self.handler_dirs,
            'base_path': str(self.base_path)
        }

class EmergencyHandlerLoader:
    """Emergency handler loader for fallback scenarios."""
    
    @staticmethod
    async def load_emergency_handlers(dispatcher: Dispatcher) -> int:
        """Load minimal emergency handlers."""
        from aiogram import Router
        from aiogram.filters import Command
        from aiogram.types import Message

        emergency_router = Router(name="emergency")

        @emergency_router.message(Command("start"))
        async def emergency_start(message: Message):
            await message.answer("🆘 Bot acil durum modunda çalışıyor. Lütfen daha sonra tekrar deneyin.")

        @emergency_router.message(Command("help"))
        async def emergency_help(message: Message):
            await message.answer("ℹ️ Bot şu anda acil durum modunda. Temel komutlar çalışıyor.")

        dispatcher.include_router(emergency_router)
        logger.info("✅ Emergency handlers loaded")
        
        return 2  # 2 handler yüklendi

# Singleton instance for global access
_handler_loader_instance: Optional[HandlerLoader] = None

async def get_handler_loader(dispatcher: Dispatcher) -> HandlerLoader:
    """Get or create singleton handler loader instance."""
    global _handler_loader_instance
    if _handler_loader_instance is None:
        _handler_loader_instance = HandlerLoader(dispatcher)
    return _handler_loader_instance

async def initialize_handlers(dispatcher: Dispatcher) -> Dict[str, int]:
    """
    Initialize handlers with comprehensive error handling.
    
    Returns:
        Dict with loading results
    """
    try:
        loader = await get_handler_loader(dispatcher)
        return await loader.load_handlers(dispatcher)
        
    except Exception as e:
        logger.critical(f"💥 Critical error in handler initialization: {e}")
        
        # Fallback to emergency handlers
        emergency_count = await EmergencyHandlerLoader.load_emergency_handlers(dispatcher)
        
        return {
            'loaded': emergency_count,
            'failed': 1,
            'skipped': 0,
            'total_files': 0,
            'errors': [f"Critical initialization error: {str(e)}"],
            'loaded_handlers': ['emergency_handlers'],
            'emergency': emergency_count
        }