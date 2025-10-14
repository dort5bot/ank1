# analysis/config/base.py
"""
Base configuration classes
"""

from abc import ABC
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, validator
from enum import Enum

class ModuleLifecycle(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing" 
    PRODUCTION = "production"


class BaseModuleConfig(BaseModel):
    """Tüm modül config'leri için base class"""
    module_name: str
    file: str
    config: str
    command: str
    api_type: str
    job_type: str
    parallel_mode: str
    output_type: str
    objective: str
    maintainer: str
    description: Optional[str] = None
    version: str
    lifecycle: ModuleLifecycle = ModuleLifecycle.DEVELOPMENT
    enabled: bool = True

    class Config:
        extra = "forbid"




class TrendConfig(BaseModuleConfig):
    """Trend modülü için specific config"""
    parameters: Dict[str, Any]
    weights: Dict[str, float]
    thresholds: Dict[str, float]
    normalization: Dict[str, Any] = {"method": "zscore"}
    
    @validator('weights')
    def validate_weights(cls, v):
        total = sum(v.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f'Weights must sum to 1.0, got {total}')
        return v

class VolatilityConfig(BaseModuleConfig):
    """Volatility modülü için specific config"""
    garch_params: Dict[str, Any]
    hurst_window: int
    entropy_bins: int
    weights: Dict[str, float]
 
 
 
"""

Uzun ömürlü çözüm önerisi

Tüm c_*.py CONFIG’leri BaseModuleConfig subclass kullanacak.

CONFIG dictionary yerine CONFIG = ClassName(...) olacak.

Loader (cm_loader.py) BaseModuleConfig tipini doğrulayacak:

"""


from analysis.config.base import BaseModuleConfig

def load_config(module_name: str) -> BaseModuleConfig:
    mod = importlib.import_module(f"analysis.config.c_{module_name}")
    if hasattr(mod, "CONFIG") and isinstance(mod.CONFIG, BaseModuleConfig):
        return mod.CONFIG
    raise ValueError(f"CONFIG missing or invalid in {module_name}")
