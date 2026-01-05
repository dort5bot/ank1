"""
utils/binance_api/binance_exceptions.py
v8 - DÜZELTİLMİŞ VERSİYON
Binance API custom exceptions.
"""

from typing import Optional, Dict, Any

class BinanceError(Exception):
    """Base exception for all Binance-related errors."""
    pass

class BinanceRequestError(Exception):
    def __init__(self, message, status_code=None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response
        self.message = message
    
    def __str__(self):
        if self.status_code:
            return f"Request Error (HTTP {self.status_code}): {self.message}"
        return f"Request Error: {self.message}"


class BinanceRequestError(Exception):
    def __init__(self, message, status_code=None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response
        self.message = message
    
    def __str__(self):
        if self.status_code:
            return f"Request Error (HTTP {self.status_code}): {self.message}"
        return f"Request Error: {self.message}"




class BinanceConfigError(BinanceError):
    """Config hatası için özel sınıf"""
    pass


class BinanceAPIError(BinanceError):
    """Exception raised for API-related errors."""

    def __init__(self, message: str, code: Optional[int] = None, response: Optional[Dict[str, Any]] = None):
        self.message = message
        self.code = code
        self.response = response
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.code and self.response:
            return f"API Error {self.code}: {self.message} - Response: {self.response}"
        elif self.code:
            return f"API Error {self.code}: {self.message}"
        else:
            return f"API Error: {self.message}"

    def __str__(self) -> str:
        return self._format_message()

    @classmethod
    def from_response(cls, status_code: int, response: Optional[Dict[str, Any]] = None):
        """
        Factory method to map Binance API error response to proper exception.
        """
        msg = response.get("msg") if response else "Unknown error"
        code = response.get("code") if response else None

        # First check HTTP status code
        if status_code == 400:
            return BinanceInvalidParameterError(msg, code, response)
        elif status_code == 401:
            return BinanceAuthenticationError(msg, code, response)
        elif status_code == 403:
            return BinancePermissionError(msg, code, response)
        elif status_code == 404:
            return BinanceNotFoundError(msg, code, response)
        elif status_code == 429:
            return BinanceRateLimitError(msg, code, response)
        elif status_code >= 500:
            return BinanceServerError(msg, code, response)


        # Then check Binance error code mapping
        if code is not None:
            mapping = {
                -2019: BinanceInsufficientBalanceError,
                -2010: BinanceOrderError,
                -2011: BinanceOrderError,
                -2013: BinanceInvalidSymbolError,  # No symbol on this market
                -2014: BinanceAuthenticationError,  # API-key format invalid
                -2015: BinanceAuthenticationError,  # Invalid API-key, IP, or permissions
                -1100: BinanceInvalidParameterError,
                -1101: BinanceInvalidParameterError,  # Too many parameters
                -1102: BinanceInvalidParameterError,  # Mandatory parameter was not sent
                -1021: BinanceTimestampError,
                -1003: BinanceRateLimitError,
                -1001: BinanceServerError,  # Internal error
            }
            exc_class = mapping.get(code, BinanceAPIError)
            return exc_class(msg, code, response)




        # Default fallback
        return BinanceAPIError(msg, code, response)


class BinanceAuthenticationError(BinanceAPIError):
    """Exception raised for authentication errors."""
    pass


class BinancePermissionError(BinanceAPIError):
    """403 Forbidden or insufficient permissions."""
    pass


class BinanceWebSocketError(BinanceError):
    def __init__(self, message: str, connection_id: Optional[str] = None):
        self.message = message
        self.connection_id = connection_id
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        if self.connection_id:
            return f"WebSocket Error [Connection {self.connection_id}]: {self.message}"
        return f"WebSocket Error: {self.message}"


class BinanceValidationError(BinanceAPIError):
    """Exception raised for general validation errors."""
    pass


class BinanceRateLimitError(BinanceAPIError):
    """Exception raised for rate limit errors (HTTP 429 or -1003)."""
    pass


class BinancePortfolioError(BinanceAPIError):
    """Portfolio işlemleri için özel hata sınıfı"""
    pass


class BinanceOrderError(BinanceAPIError):
    """Exception raised for order-related errors (-2010/-2011)."""
    pass

class BinanceOrderRejectedError(BinanceOrderError):
    """Exception raised when an order is rejected by Binance."""
    pass


class BinanceInvalidSymbolError(BinanceAPIError):
    """Exception raised for invalid symbol errors."""
    pass


class BinanceInvalidIntervalError(BinanceAPIError):
    """Exception raised for invalid interval errors."""
    pass


class BinanceTimeoutError(BinanceError):
    """Exception raised for timeout errors."""
    pass


class BinanceConnectionError(BinanceError):
    """Exception raised for connection errors."""
    pass


class BinanceInvalidParameterError(BinanceAPIError):
    """Exception raised for invalid parameter errors (HTTP 400 or -1100)."""
    pass


class BinanceInsufficientBalanceError(BinanceAPIError):
    """Exception raised for insufficient balance errors (-2019)."""
    pass


class BinanceNotFoundError(BinanceAPIError):
    """Exception raised for 404 Not Found errors (endpoint or symbol)."""
    pass


class BinanceServerError(BinanceAPIError):
    """Exception raised for 5xx Binance server errors."""
    pass


class BinanceTimestampError(BinanceAPIError):
    """Exception raised for invalid timestamp or recvWindow issues (-1021)."""
    pass


class BinanceCancelRejectedError(BinanceOrderError):
    """Exception raised when an order cancellation is rejected by Binance."""
    pass

class BinanceWithdrawalError(BinanceAPIError):
    """Exception raised for withdrawal-related errors."""
    pass

class BinanceMaintenanceError(BinanceAPIError):
    """Exception raised when Binance is under maintenance."""
    pass


class BinanceBannedError(BinanceAPIError):
    """Exception raised when API key/account is banned or restricted."""
    pass
    
class BinanceIPBannedError(BinanceBannedError):
    """Exception raised when IP address is banned."""
    pass

class BinanceMarketClosedError(BinanceAPIError):
    """Exception raised when trying to trade on a closed market."""
    pass



