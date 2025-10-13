"""
utils/performance_monitor.py
Performance monitoring decorators and utilities
"""

import functools
import time
import asyncio
import logging
from typing import Any, Callable, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Performance monitoring singleton"""
    
    _instance: Optional["PerformanceMonitor"] = None
    _metrics: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self):
        if self._instance is not None:
            return
        self._metrics = {}
        self._start_time = datetime.now()
        PerformanceMonitor._instance = self
    
    @classmethod
    def get_instance(cls) -> "PerformanceMonitor":
        if cls._instance is None:
            cls._instance = PerformanceMonitor()
        return cls._instance
    
    def record_metric(self, func_name: str, duration: float, success: bool = True):
        """Record performance metric"""
        if func_name not in self._metrics:
            self._metrics[func_name] = {
                'call_count': 0,
                'total_time': 0.0,
                'success_count': 0,
                'error_count': 0,
                'avg_time': 0.0,
                'max_time': 0.0,
                'min_time': float('inf')
            }
        
        metric = self._metrics[func_name]
        metric['call_count'] += 1
        metric['total_time'] += duration
        
        if success:
            metric['success_count'] += 1
        else:
            metric['error_count'] += 1
            
        metric['avg_time'] = metric['total_time'] / metric['call_count']
        metric['max_time'] = max(metric['max_time'], duration)
        metric['min_time'] = min(metric['min_time'], duration)
    
    def get_metrics(self, func_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics"""
        if func_name:
            return self._metrics.get(func_name, {})
        return self._metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        total_calls = sum(m['call_count'] for m in self._metrics.values())
        total_time = sum(m['total_time'] for m in self._metrics.values())
        avg_time = total_time / total_calls if total_calls > 0 else 0
        
        return {
            'total_functions_monitored': len(self._metrics),
            'total_calls': total_calls,
            'total_processing_time': total_time,
            'average_call_time': avg_time,
            'uptime_seconds': (datetime.now() - self._start_time).total_seconds(),
            'top_slow_functions': sorted(
                [(name, data['avg_time']) for name, data in self._metrics.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }


# Performance monitoring enhancement
class AdvancedPerformanceMonitor(PerformanceMonitor):
    async def track_memory_usage(self):
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    async def generate_performance_report(self):
        return {
            **self.get_summary(),
            "memory_usage_mb": await self.track_memory_usage(),
            "active_coroutines": len([t for t in asyncio.all_tasks() if not t.done()])
        }


def monitor_performance(func_name: Optional[str] = None, warning_threshold: float = 2.0):
    """
    Performance monitoring decorator for async functions
    
    Args:
        func_name: Custom function name for logging (defaults to function.__name__)
        warning_threshold: Threshold in seconds for warning logs
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            monitor = PerformanceMonitor.get_instance()
            actual_func_name = func_name or func.__name__
            
            start_time = asyncio.get_event_loop().time()
            try:
                result = await func(*args, **kwargs)
                processing_time = asyncio.get_event_loop().time() - start_time
                
                # Record metric
                monitor.record_metric(actual_func_name, processing_time, success=True)
                
                # Log warning if too slow
                if processing_time > warning_threshold:
                    logger.warning(
                        f"⏱️ PERFORMANCE - {actual_func_name} took {processing_time:.2f}s "
                        f"(threshold: {warning_threshold}s)"
                    )
                elif processing_time > warning_threshold * 0.5:  # 50% of threshold
                    logger.info(
                        f"⏱️ Performance - {actual_func_name} took {processing_time:.2f}s"
                    )
                
                return result
                
            except Exception as e:
                processing_time = asyncio.get_event_loop().time() - start_time
                monitor.record_metric(actual_func_name, processing_time, success=False)
                
                logger.error(
                    f"❌ PERFORMANCE ERROR - {actual_func_name} failed after "
                    f"{processing_time:.2f}s: {e}"
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            monitor = PerformanceMonitor.get_instance()
            actual_func_name = func_name or func.__name__
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                processing_time = time.time() - start_time
                
                monitor.record_metric(actual_func_name, processing_time, success=True)
                
                if processing_time > warning_threshold:
                    logger.warning(
                        f"⏱️ PERFORMANCE - {actual_func_name} took {processing_time:.2f}s "
                        f"(threshold: {warning_threshold}s)"
                    )
                
                return result
                
            except Exception as e:
                processing_time = time.time() - start_time
                monitor.record_metric(actual_func_name, processing_time, success=False)
                
                logger.error(
                    f"❌ PERFORMANCE ERROR - {actual_func_name} failed after "
                    f"{processing_time:.2f}s: {e}"
                )
                raise
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# Convenience decorators for specific thresholds
def monitor_critical_performance(func_name: Optional[str] = None):
    """For critical functions with 1s threshold"""
    return monitor_performance(func_name, warning_threshold=1.0)

def monitor_fast_performance(func_name: Optional[str] = None):
    """For fast functions with 0.5s threshold"""
    return monitor_performance(func_name, warning_threshold=0.5)