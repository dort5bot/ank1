# analysis/config/loader.py
"""
Merkezi config yükleyici ve yönetici
"""

import importlib
from typing import Dict, Any, Type
from .cm_base import BaseModuleConfig

class ConfigManager:
    """Config yöneticisi - singleton pattern"""
    
    _instance = None
    _configs: Dict[str, BaseModuleConfig] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._configs:
            self._load_all_configs()
    
    def _load_all_configs(self):
        """Tüm modül config'lerini yükle"""
        modules_to_load = ['trend', 'volatility', 'sentiment']
        
        for module_name in modules_to_load:
            try:
                #module = importlib.import_module(f'analysis.config.{module_name}')
                #config_obj = getattr(module, f'{module_name.upper()}_CONFIG')
                #self._configs[module_name] = config_obj
                
                module = importlib.import_module(f'analysis.config.{module_name}')
                config_obj = getattr(module, "CONFIG")
                self._configs[module_name] = config_obj
           
            except (ImportError, AttributeError) as e:
                print(f"Config load failed for {module_name}: {e}")
    
    def get_config(self, module_name: str) -> BaseModuleConfig:
        """Modül config'ini getir"""
        return self._configs.get(module_name)
    
    def get_all_configs(self) -> Dict[str, BaseModuleConfig]:
        """Tüm config'leri getir"""
        return self._configs.copy()
    
    def update_config(self, module_name: str, **updates):
        """Config güncelleme (validation ile)"""
        if module_name in self._configs:
            current_config = self._configs[module_name].dict()
            current_config.update(updates)
            
            # Yeni instance oluştur (validation için)
            config_class = type(self._configs[module_name])
            self._configs[module_name] = config_class(**current_config)

# Global instance
config_manager = ConfigManager()