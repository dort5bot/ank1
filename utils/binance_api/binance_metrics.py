# utils/binance_api/binance_metrics.py
# v2 - Simplified, async-safe metrics collection
from __future__ import annotations

import time
import asyncio
from typing import Dict, List, Any, Optional, Union
from collections import deque, Counter
import statistics
import logging
import json

logger = logging.getLogger(__name__)

class BinanceMetrics:
    """
    Simple async-safe metrics collection for Binance API.
    
    Usage:
        metrics = BinanceMetrics()
        await metrics.record_request(...)
        await metrics.get_metrics()
    """
    
    def __init__(self, window_size: int = 5000):
        self.window_size = window_size
        self.start_time = time.time()
        self._lock = asyncio.Lock()
        
        # Core metrics storage
        self._metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_response_time': 0.0,
            'response_times': deque(maxlen=window_size),
            'errors_by_type': Counter(),
            'status_counters': Counter(),
            'binance_code_counters': Counter(),
            'response_time_buckets': Counter(),
            'endpoint_metrics': {},
            'total_retries': 0,
            'retry_attempts': Counter(),
            'rate_limits': {
                'weight_used_1m': 0,
                'weight_limit_1m': 1200,
                'last_reset_1m': time.time(),
                'weight_used_1s': 0,
                'weight_limit_1s': 20,
                'last_reset_1s': time.time(),
            }
        }
        
        logger.info("BinanceMetrics initialized")

    @staticmethod
    def _parse_binance_error_code(body: Union[str, bytes, Dict[str, Any], None]) -> Optional[int]:
        """Parse Binance error code from response body."""
        if body is None:
            return None
            
        try:
            if isinstance(body, (bytes, bytearray)):
                body = body.decode("utf-8", errors="ignore")
            if isinstance(body, str):
                parsed = json.loads(body)
            else:
                parsed = body

            if isinstance(parsed, dict):
                if code := parsed.get("code"):
                    return int(code)
                for key in ("error", "error_code", "err"):
                    if key in parsed and isinstance(parsed[key], int):
                        return parsed[key]
        except Exception:
            pass
            
        return None

    @staticmethod
    def _get_response_time_bucket(response_time: float) -> str:
        """Categorize response time into bucket."""
        if response_time <= 0.1: return "0.1s"
        if response_time <= 0.5: return "0.5s"
        if response_time <= 1.0: return "1.0s"
        if response_time <= 2.0: return "2.0s"
        if response_time <= 5.0: return "5.0s"
        return "5.0s+"

    async def record_request(
        self,
        *,
        endpoint: Optional[str] = None,
        response_time: float,
        status_code: Optional[int] = None,
        error: Optional[Exception] = None,
        binance_code: Optional[int] = None,
        weight_used: int = 0,
        headers: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a single API request.
        
        Args:
            endpoint: API endpoint identifier
            response_time: Request duration in seconds
            status_code: HTTP status code
            error: Exception if request failed
            binance_code: Binance-specific error code
            weight_used: Rate limit weight consumed
            headers: Response headers for rate limit parsing
        """
        async with self._lock:
            m = self._metrics
            
            # Parse rate limits from headers if provided
            if headers:
                for key, value in headers.items():
                    key_lower = key.lower()
                    try:
                        if "used-weight-1m" in key_lower or "used_weight_1m" in key_lower:
                            m['rate_limits']['weight_used_1m'] = int(value)
                        elif "used-weight-1s" in key_lower or "used_weight_1s" in key_lower:
                            m['rate_limits']['weight_used_1s'] = int(value)
                    except (ValueError, TypeError):
                        continue

            # Update weight usage
            if weight_used > 0:
                m['rate_limits']['weight_used_1m'] += weight_used

            # Determine success
            success = error is None and (status_code is None or 200 <= status_code < 300)
            
            # Update basic metrics
            m['total_requests'] += 1
            m['total_response_time'] += response_time
            m['response_times'].append(response_time)
            
            if success:
                m['successful_requests'] += 1
            else:
                m['failed_requests'] += 1

            # Error classification
            if error:
                m['errors_by_type'][type(error).__name__] += 1
            elif binance_code:
                m['errors_by_type'][f"binance_code_{binance_code}"] += 1
                m['binance_code_counters'][f"code_{binance_code}"] += 1
            elif status_code and status_code >= 400:
                m['errors_by_type'][f"http_{status_code}"] += 1

            # Status code tracking
            if status_code:
                m['status_counters'][f"http_{status_code}"] += 1

            # Response time bucket
            bucket = self._get_response_time_bucket(response_time)
            m['response_time_buckets'][bucket] += 1

            # Endpoint-specific metrics
            if endpoint:
                if endpoint not in m['endpoint_metrics']:
                    m['endpoint_metrics'][endpoint] = {
                        'total_requests': 0,
                        'successful_requests': 0,
                        'total_response_time': 0.0,
                        'response_times': deque(maxlen=1000)
                    }
                
                ep_metrics = m['endpoint_metrics'][endpoint]
                ep_metrics['total_requests'] += 1
                ep_metrics['total_response_time'] += response_time
                ep_metrics['response_times'].append(response_time)
                if success:
                    ep_metrics['successful_requests'] += 1

    async def record_retry(self, endpoint: str, attempt: int) -> None:
        """Record a retry attempt."""
        async with self._lock:
            self._metrics['retry_attempts'][f"{endpoint}_attempt_{attempt}"] += 1
            self._metrics['total_retries'] += 1

    async def update_rate_limits(self, weight_1m: Optional[int] = None, weight_1s: Optional[int] = None) -> None:
        """Update rate limit counters."""
        async with self._lock:
            if weight_1m is not None:
                self._metrics['rate_limits']['weight_used_1m'] = weight_1m
            if weight_1s is not None:
                self._metrics['rate_limits']['weight_used_1s'] = weight_1s

    async def reset_rate_limits(self) -> None:
        """Reset rate limit counters."""
        async with self._lock:
            now = time.time()
            self._metrics['rate_limits']['weight_used_1m'] = 0
            self._metrics['rate_limits']['last_reset_1m'] = now
            self._metrics['rate_limits']['weight_used_1s'] = 0
            self._metrics['rate_limits']['last_reset_1s'] = now

    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics snapshot."""
        async with self._lock:
            return self._compile_metrics()

    def _compile_metrics(self) -> Dict[str, Any]:
        """Compile metrics without lock (internal use)."""
        m = self._metrics
        response_times = list(m['response_times'])
        total_requests = m['total_requests']
        
        # Calculate percentiles
        p95 = self._safe_percentile(response_times, 0.95)
        p99 = self._safe_percentile(response_times, 0.99)
        
        # Calculate rates
        uptime = time.time() - self.start_time
        uptime_minutes = uptime / 60 if uptime > 0 else 1.0
        current_rpm = total_requests / uptime_minutes if uptime_minutes > 0 else 0.0
        
        # Compile endpoint metrics
        endpoint_metrics = {}
        for endpoint, ep_data in m['endpoint_metrics'].items():
            ep_total = ep_data['total_requests']
            endpoint_metrics[endpoint] = {
                'total_requests': ep_total,
                'success_rate': (ep_data['successful_requests'] / ep_total * 100) if ep_total > 0 else 100.0,
                'average_response_time': (ep_data['total_response_time'] / ep_total) if ep_total > 0 else 0.0,
            }

        return {
            'uptime_seconds': uptime,
            'total_requests': total_requests,
            'successful_requests': m['successful_requests'],
            'failed_requests': m['failed_requests'],
            'success_rate': (m['successful_requests'] / total_requests * 100) if total_requests > 0 else 100.0,
            'average_response_time': (m['total_response_time'] / total_requests) if total_requests > 0 else 0.0,
            'min_response_time': min(response_times) if response_times else 0.0,
            'max_response_time': max(response_times) if response_times else 0.0,
            'p95_response_time': p95,
            'p99_response_time': p99,
            'current_rpm': current_rpm,
            'response_time_buckets': dict(m['response_time_buckets']),
            'endpoint_metrics': endpoint_metrics,
            'errors_by_type': dict(m['errors_by_type']),
            'status_counters': dict(m['status_counters']),
            'binance_code_counters': dict(m['binance_code_counters']),
            'total_retries': m['total_retries'],
            'retry_attempts': dict(m['retry_attempts']),
            'rate_limits': {
                'weight_used_1m': m['rate_limits']['weight_used_1m'],
                'weight_limit_1m': m['rate_limits']['weight_limit_1m'],
                'weight_remaining_1m': max(0, m['rate_limits']['weight_limit_1m'] - m['rate_limits']['weight_used_1m']),
                'weight_percentage_1m': (m['rate_limits']['weight_used_1m'] / m['rate_limits']['weight_limit_1m'] * 100) if m['rate_limits']['weight_limit_1m'] > 0 else 0.0,
                'last_reset_seconds_ago_1m': time.time() - m['rate_limits']['last_reset_1m'],
            }
        }

    @staticmethod
    def _safe_percentile(values: List[float], percentile: float) -> float:
        """Calculate percentile safely."""
        if not values:
            return 0.0
            
        try:
            return statistics.quantiles(values, n=100, method="inclusive")[min(98, int(percentile * 99))]
        except Exception:
            sorted_vals = sorted(values)
            index = int(len(sorted_vals) * percentile)
            return sorted_vals[min(index, len(sorted_vals) - 1)]

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status summary."""
        metrics = await self.get_metrics()
        
        status = "HEALTHY"
        issues = []
        warnings = []

        success_rate = metrics['success_rate']
        avg_rt = metrics['average_response_time']
        p95_rt = metrics['p95_response_time']
        weight_pct = metrics['rate_limits']['weight_percentage_1m']

        if success_rate < 90.0:
            status = "CRITICAL"
            issues.append(f"Low success rate: {success_rate:.1f}%")
        elif success_rate < 95.0:
            status = "DEGRADED"
            warnings.append(f"Low success rate: {success_rate:.1f}%")

        if avg_rt > 3.0:
            status = "CRITICAL"
            issues.append(f"High avg response time: {avg_rt:.2f}s")
        elif avg_rt > 1.5:
            if status == "HEALTHY":
                status = "DEGRADED"
            warnings.append(f"Elevated avg response time: {avg_rt:.2f}s")

        if p95_rt > 5.0:
            status = "CRITICAL"
            issues.append(f"High p95 response time: {p95_rt:.2f}s")
        elif p95_rt > 2.5:
            if status == "HEALTHY":
                status = "DEGRADED"
            warnings.append(f"High p95 response time: {p95_rt:.2f}s")

        if weight_pct > 95.0:
            status = "CRITICAL"
            issues.append(f"Rate limit near exhausted: {weight_pct:.1f}%")
        elif weight_pct > 80.0:
            if status == "HEALTHY":
                status = "DEGRADED"
            warnings.append(f"High rate limit usage: {weight_pct:.1f}%")

        return {
            'status': status,
            'issues': issues,
            'warnings': warnings,
            'timestamp': time.time(),
            'metrics': metrics
        }

    async def reset(self) -> None:
        """Reset all metrics."""
        async with self._lock:
            self._metrics = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'total_response_time': 0.0,
                'response_times': deque(maxlen=self.window_size),
                'errors_by_type': Counter(),
                'status_counters': Counter(),
                'binance_code_counters': Counter(),
                'response_time_buckets': Counter(),
                'endpoint_metrics': {},
                'total_retries': 0,
                'retry_attempts': Counter(),
                'rate_limits': {
                    'weight_used_1m': 0,
                    'weight_limit_1m': 1200,
                    'last_reset_1m': time.time(),
                    'weight_used_1s': 0,
                    'weight_limit_1s': 20,
                    'last_reset_1s': time.time(),
                }
            }
            self.start_time = time.time()
            logger.info("Metrics reset")

# Global instance for convenience
_global_metrics = BinanceMetrics()

# Module-level API
async def record_request(
    endpoint: Optional[str] = None,
    response_time: float = 0.0,
    status_code: Optional[int] = None,
    error: Optional[Exception] = None,
    response_body: Union[str, bytes, Dict[str, Any], None] = None,
    headers: Optional[Dict[str, Any]] = None,
    weight_used: int = 0
) -> None:
    """Record an API request."""
    binance_code = BinanceMetrics._parse_binance_error_code(response_body)
    await _global_metrics.record_request(
        endpoint=endpoint,
        response_time=response_time,
        status_code=status_code,
        error=error,
        binance_code=binance_code,
        weight_used=weight_used,
        headers=headers
    )

async def record_retry(endpoint: str, attempt: int) -> None:
    """Record a retry attempt."""
    await _global_metrics.record_retry(endpoint, attempt)

async def get_metrics() -> Dict[str, Any]:
    """Get current metrics."""
    return await _global_metrics.get_metrics()

async def get_health_status() -> Dict[str, Any]:
    """Get health status."""
    return await _global_metrics.get_health_status()

async def reset_metrics() -> None:
    """Reset all metrics."""
    await _global_metrics.reset()

async def update_rate_limits(weight_1m: Optional[int] = None, weight_1s: Optional[int] = None) -> None:
    """Update rate limit counters."""
    await _global_metrics.update_rate_limits(weight_1m, weight_1s)

async def reset_rate_limits() -> None:
    """Reset rate limit counters."""
    await _global_metrics.reset_rate_limits()

# aiogram integration
try:
    from aiogram import Router
    from aiogram.types import Message
    from aiogram.filters import Command

    metrics_router = Router()

    @metrics_router.message(Command(commands=["metrics", "metrics_status"]))
    async def metrics_status_handler(message: Message) -> None:
        """Handle /metrics command."""
        metrics = await get_metrics()
        health = await get_health_status()
        
        text = (
            f"Status: {health['status']}\n"
            f"Success: {metrics['success_rate']:.1f}% | "
            f"Avg: {metrics['average_response_time']:.3f}s | "
            f"p95: {metrics['p95_response_time']:.3f}s\n"
            f"Rate Limit: {metrics['rate_limits']['weight_used_1m']}/{metrics['rate_limits']['weight_limit_1m']} "
            f"({metrics['rate_limits']['weight_percentage_1m']:.1f}%)\n"
            f"Requests: {metrics['total_requests']} | "
            f"RPM: {metrics['current_rpm']:.1f}\n"
        )
        
        if health['issues']:
            text += f"Issues: {', '.join(health['issues'])}\n"
        if health['warnings']:
            text += f"Warnings: {', '.join(health['warnings'])}"
            
        await message.answer(text)

except ImportError:
    metrics_router = None