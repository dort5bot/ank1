"""
utils/context_logger.py
Enhanced logging with context information
"""

import os
import logging
import inspect
from typing import Dict, Any, Optional
from datetime import datetime

class ContextFilter(logging.Filter):
    """Custom filter to add context to log records"""
    
    def __init__(self):
        super().__init__()
        self._context_data = {}
    
    def add_context(self, key: str, value: Any):
        """Add context data"""
        self._context_data[key] = value
    
    def remove_context(self, key: str):
        """Remove context data"""
        self._context_data.pop(key, None)
    
    def clear_context(self):
        """Clear all context data"""
        self._context_data.clear()
    
    def filter(self, record):
        # Add platform context
        record.platform = "render" if "RENDER" in os.environ else (
            "railway" if "RAILWAY" in os.environ else "local"
        )
        
        # Security sanitization
        if hasattr(record, 'msg') and isinstance(record.msg, dict):
            record.msg = security_auditor.sanitize_log_data(record.msg)
                
    
        # Add multi-user context
        record.multi_user = True
        record.timestamp_iso = datetime.now().isoformat()
        
        # Add custom context data
        for key, value in self._context_data.items():
            setattr(record, key, value)
        
        # Add function context if available
        frame = inspect.currentframe()
        try:
            # Go back a few frames to find the caller
            for _ in range(6):  # Maximum 6 frames back
                frame = frame.f_back
                if frame is None:
                    break
                
                # Get function name and module
                func_name = frame.f_code.co_name
                module = frame.f_globals.get('__name__', 'unknown')
                
                if func_name != 'filter' and not func_name.startswith('_'):
                    record.function_context = f"{module}.{func_name}"
                    break
            else:
                record.function_context = "unknown"
        except:
            record.function_context = "error_getting_context"
        finally:
            del frame
        
        return True


class ContextAwareLogger:
    """Context-aware logger factory"""
    
    _initialized = False
    _context_filter = ContextFilter()
    
    @classmethod
    def initialize(cls):
        """Initialize context-aware logging system"""
        if cls._initialized:
            return
        
        # Get root logger
        root_logger = logging.getLogger()
        
        # Add context filter to all handlers
        for handler in root_logger.handlers:
            handler.addFilter(cls._context_filter)
        
        # Also add to our module logger
        module_logger = logging.getLogger(__name__)
        for handler in module_logger.handlers:
            handler.addFilter(cls._context_filter)
        
        cls._initialized = True
        logging.info("✅ Context-aware logging initialized")
    
    @classmethod
    def get_logger(cls, name: Optional[str] = None):
        """Get a context-aware logger"""
        if not cls._initialized:
            cls.initialize()
        
        logger = logging.getLogger(name)
        
        # Ensure our filter is added to this logger's handlers
        for handler in logger.handlers:
            if not any(isinstance(f, ContextFilter) for f in handler.filters):
                handler.addFilter(cls._context_filter)
        
        return logger
    
    @classmethod
    def add_context(cls, key: str, value: Any):
        """Add global context data"""
        cls._context_filter.add_context(key, value)
    
    @classmethod
    def remove_context(cls, key: str):
        """Remove global context data"""
        cls._context_filter.remove_context(key)
    
    @classmethod
    def clear_context(cls):
        """Clear all global context data"""
        cls._context_filter.clear_context()
    
    @classmethod
    def set_user_context(cls, user_id: Optional[int] = None):
        """Set user-specific context"""
        if user_id:
            cls.add_context('user_id', user_id)
            cls.add_context('log_prefix', f"[User:{user_id}]")
        else:
            cls.remove_context('user_id')
            cls.remove_context('log_prefix')
    
    @classmethod
    def set_request_context(cls, request_id: str, endpoint: str):
        """Set request-specific context"""
        cls.add_context('request_id', request_id)
        cls.add_context('endpoint', endpoint)
        cls.add_context('log_prefix', f"[Req:{request_id[:8]}]")




# Convenience function for quick setup
def setup_context_logging():
    """One-time setup for context-aware logging"""
    ContextAwareLogger.initialize()


# Shortcut for getting context-aware logger
def get_context_logger(name: Optional[str] = None):
    """Get a context-aware logger (shortcut)"""
    return ContextAwareLogger.get_logger(name)