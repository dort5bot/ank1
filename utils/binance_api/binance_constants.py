#utils/binance_api/binance_constants.py
"""
v7
Spot + Futures + Margin hata kodları kapsandı
Parametre doğrulama fonksiyonları eklendi
Exception mapping için map_error_code eklendi
Futures özel interval desteği ayrıldı
Endpoint path'leri için sabitler eklendi
WebSocket stream'ler için ters lookup eklendi

fapi: Futures API anlamına gelir. Özellikle USD-M Futures (USDT-margined).
v1 / v2: API versiyonudur. v2 genellikle v1'in genişletilmiş/iyileştirilmiş halidir.
/ws/...: WebSocket endpoint’idir. Sürekli veri akışı sağlar (REST API değil).

Binance API constants, configuration, and validation helpers.
Uygulama: Spot + Futures + Margin + Testnet desteği
"""

from typing import Any, Dict, Final, List, Optional, Union

# binance_constants.py sonuna ekde:
from pydantic_settings import BaseSettings


# =============================================================================
# Base URLs
# =============================================================================
BASE_URL: Final[str] = "https://api.binance.com"
FUTURES_URL: Final[str] = "https://fapi.binance.com"
MARGIN_URL: Final[str] = BASE_URL  # margin aynı domain altında
TESTNET_BASE_URL: Final[str] = "https://testnet.binance.vision"
TESTNET_FUTURES_URL: Final[str] = "https://testnet.binancefuture.com"

# WebSocket Base URLs 
WS_BASE_URL: Final[str] = "wss://stream.binance.com:9443/ws"
WS_FUTURES_URL: Final[str] = "wss://fstream.binance.com/ws"
WS_TESTNET_URL: Final[str] = "wss://testnet.binance.vision/ws"
WS_TESTNET_FUTURES_URL: Final[str] = "wss://stream.binancefuture.com/ws"



# API Versions
API_VERSION: Final[str] = "v3"
FUTURES_API_VERSION: Final[str] = "v1"


# =============================================================================
# Rate Limits
# =============================================================================
RATE_LIMIT_IP: Final[int] = 1200  # Requests per minute per IP
RATE_LIMIT_ORDER: Final[int] = 10  # Orders per second
RATE_LIMIT_RAW_REQUESTS: Final[int] = 5000  # Raw requests per 5 minutes


# hard-coded değerler constants'a
KEEPALIVE_INTERVAL = 30 * 60  # 30 dakika
RECEIVE_TIMEOUT = 1.0


# Symbol Filters (eklenebilir)
SYMBOL_FILTERS: Final[List[str]] = [
    "PRICE_FILTER", "LOT_SIZE", "MIN_NOTIONAL", 
    "MAX_NUM_ORDERS", "MAX_ALGO_ORDERS", "PERCENT_PRICE"
]


# =============================================================================
# Time Intervals (candlestick)
# =============================================================================
INTERVALS_SPOT: Final[List[str]] = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
]

INTERVALS_FUTURES: Final[List[str]] = [
    "1s", "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
]


# =============================================================================
# Orders
# =============================================================================
ORDER_TYPES: Final[List[str]] = [
    "LIMIT", "MARKET", "STOP_LOSS", "STOP_LOSS_LIMIT",
    "TAKE_PROFIT", "TAKE_PROFIT_LIMIT", "LIMIT_MAKER",
]

ORDER_SIDES: Final[List[str]] = ["BUY", "SELL"]

# Order Status
ORDER_STATUS: Final[List[str]] = [
    "NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED", 
    "PENDING_CANCEL", "REJECTED", "EXPIRED"
]


TIME_IN_FORCE: Final[List[str]] = ["GTC", "IOC", "FOK"]

RESPONSE_TYPES: Final[List[str]] = ["ACK", "RESULT", "FULL"]



# =============================================================================
# Types
# =============================================================================
# Futures Working Types
WORKING_TYPES: Final[List[str]] = ["MARK_PRICE", "CONTRACT_PRICE"]

# Margin Types
MARGIN_TYPES: Final[List[str]] = ["ISOLATED", "CROSSED"]


# =============================================================================
# Kline Fields
# =============================================================================
KLINE_FIELDS: Final[List[str]] = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
]


# =============================================================================
# Endpoint Paths
# =============================================================================

# Public Spot Endpoints
SPOT_PING_ENDPOINT: Final[str] = "/api/v3/ping"
SPOT_TIME_ENDPOINT: Final[str] = "/api/v3/time"
SPOT_EXCHANGE_INFO_ENDPOINT: Final[str] = "/api/v3/exchangeInfo"
SPOT_ORDER_BOOK_ENDPOINT: Final[str] = "/api/v3/depth"
SPOT_KLINE_ENDPOINT: Final[str] = "/api/v3/klines"
SPOT_AGG_TRADE_ENDPOINT: Final[str] = "/api/v3/aggTrades"
SPOT_TICKER_24H_ENDPOINT: Final[str] = "/api/v3/ticker/24hr"
SPOT_TICKER_PRICE_ENDPOINT: Final[str] = "/api/v3/ticker/price"
SPOT_TICKER_BOOK_ENDPOINT: Final[str] = "/api/v3/ticker/bookTicker"
SPOT_AVG_PRICE_ENDPOINT: Final[str] = "/api/v3/avgPrice"
SPOT_TRADES_ENDPOINT: Final[str] = "/api/v3/trades"
SPOT_HISTORICAL_TRADES_ENDPOINT: Final[str] = "/api/v3/historicalTrades"
SPOT_TICKER_GENERIC_ENDPOINT: Final[str] = "/api/v3/ticker"
SPOT_UI_KLINES_ENDPOINT: Final[str] = "/api/v3/uiKlines"  # undocumented (UI use only)


# Private Spot Endpoints
SPOT_ORDER_ENDPOINT: Final[str] = "/api/v3/order"
SPOT_ORDER_TEST_ENDPOINT: Final[str] = "/api/v3/order/test"
SPOT_OPEN_ORDERS_ENDPOINT: Final[str] = "/api/v3/openOrders"
SPOT_ALL_ORDERS_ENDPOINT: Final[str] = "/api/v3/allOrders"
SPOT_ACCOUNT_ENDPOINT: Final[str] = "/api/v3/account"
SPOT_MY_TRADES_ENDPOINT: Final[str] = "/api/v3/myTrades"

# Futures Endpoints
FUTURES_PING_ENDPOINT: Final[str] = "/fapi/v1/ping"
FUTURES_TIME_ENDPOINT: Final[str] = "/fapi/v1/time"
FUTURES_EXCHANGE_INFO_ENDPOINT: Final[str] = "/fapi/v1/exchangeInfo"
FUTURES_ORDER_BOOK_ENDPOINT: Final[str] = "/fapi/v1/depth"
FUTURES_KLINE_ENDPOINT: Final[str] = "/fapi/v1/klines"
FUTURES_ACCOUNT_ENDPOINT: Final[str] = "/fapi/v2/account"
FUTURES_ORDER_ENDPOINT: Final[str] = "/fapi/v1/order"
FUTURES_POSITION_RISK_ENDPOINT: Final[str] = "/fapi/v2/positionRisk"
FUTURES_PREMIUM_INDEX_ENDPOINT = "/fapi/v1/premiumIndex"
FUTURES_LEVERAGE_BRACKET_ENDPOINT: Final[str] = "/fapi/v1/leverageBracket"
FUTURES_LIQUIDATION_ORDERS_ENDPOINT = "/fapi/v1/liquidationOrders"
FUTURES_CONTINUOUS_KLINES_ENDPOINT = "/fapi/v1/continuousKlines"
FUTURES_INDEX_PRICE_KLINES_ENDPOINT = "/fapi/v1/indexPriceKlines"
FUTURES_MARK_PRICE_KLINES_ENDPOINT = "/fapi/v1/markPriceKlines"
FUTURES_FUNDING_RATE_ENDPOINT = "/fapi/v1/fundingRate"
FUTURES_TICKER_24HR_ENDPOINT = "/fapi/v1/ticker/24hr"
FUTURES_TICKER_PRICE_ENDPOINT = "/fapi/v1/ticker/price"
FUTURES_TICKER_BOOK_ENDPOINT = "/fapi/v1/ticker/bookTicker"
FUTURES_OPEN_INTEREST_ENDPOINT = "/fapi/v1/openInterest"
FUTURES_OPEN_INTEREST_HIST_ENDPOINT = "/futures/data/openInterestHist"
FUTURES_TOP_LONG_SHORT_POSITION_RATIO_ENDPOINT = "/futures/data/topLongShortPositionRatio"
FUTURES_TOP_LONG_SHORT_ACCOUNT_RATIO_ENDPOINT = "/futures/data/topLongShortAccountRatio"
FUTURES_GLOBAL_LONG_SHORT_ACCOUNT_RATIO_ENDPOINT = "/futures/data/globalLongShortAccountRatio"
FUTURES_TAKER_BUY_SELL_VOL_ENDPOINT = "/futures/data/takerBuySellVol"
FUTURES_BASIS_ENDPOINT = "/futures/data/basis"

FUTURES_FORCE_ORDERS_ENDPOINT = "/fapi/v1/forceOrders"
FUTURES_ALL_FORCE_ORDERS_ENDPOINT = "/fapi/v1/allForceOrders"
FUTURES_LEVERAGE_BRACKET_ENDPOINT = "/fapi/v1/leverageBracket"


# Margin Endpoints 
MARGIN_ACCOUNT_ENDPOINT: Final[str] = "/sapi/v1/margin/account"
MARGIN_ORDER_ENDPOINT: Final[str] = "/sapi/v1/margin/order"
MARGIN_OPEN_ORDERS_ENDPOINT: Final[str] = "/sapi/v1/margin/openOrders"
MARGIN_ALL_ORDERS_ENDPOINT: Final[str] = "/sapi/v1/margin/allOrders"
MARGIN_MY_TRADES_ENDPOINT: Final[str] = "/sapi/v1/margin/myTrades"



# =============================================================================
# WebSocket Streams
# =============================================================================
WS_STREAMS: Final[Dict[str, str]] = {
    "agg_trade": "aggTrade",
    "trade": "trade",
    "kline": "kline",
    "mini_ticker": "miniTicker",
    "ticker": "ticker",
    "book_ticker": "bookTicker",
    "depth": "depth",
    "depth_level": "depth{}",
    "user_data": "userData",
}

# Reverse lookup for WebSocket streams
WS_STREAMS_REVERSE: Final[Dict[str, str]] = {v: k for k, v in WS_STREAMS.items()}


# =============================================================================
# Error Codes (Spot + Futures + Margin)
# Kaynak: https://binance-docs.github.io/apidocs
# =============================================================================
ERROR_CODES: Final[Dict[int, str]] = {
    -1000: "UNKNOWN",
    -1001: "DISCONNECTED",
    -1002: "UNAUTHORIZED",
    -1003: "TOO_MANY_REQUESTS",
    -1006: "UNEXPECTED_RESP",
    -1007: "TIMEOUT",
    -1010: "ERROR_MSG_RECEIVED",
    -1013: "INVALID_MESSAGE",
    -1014: "UNKNOWN_ORDER_COMPOSITION",
    -1015: "TOO_MANY_ORDERS",
    -1016: "SERVICE_SHUTTING_DOWN",
    -1020: "UNSUPPORTED_OPERATION",
    -1021: "INVALID_TIMESTAMP",
    -1022: "INVALID_SIGNATURE",
    -1100: "ILLEGAL_CHARS",
    -1101: "TOO_MANY_PARAMETERS",
    -1102: "MANDATORY_PARAM_EMPTY_OR_MALFORMED",
    -1103: "UNKNOWN_PARAM",
    -1104: "UNREAD_PARAMETERS",
    -1105: "PARAM_EMPTY",
    -1106: "PARAM_NOT_REQUIRED",
    -1111: "BAD_PRECISION",
    -1112: "NO_DEPTH",
    -1114: "TIF_NOT_REQUIRED",
    -1115: "INVALID_TIF",
    -1116: "INVALID_ORDER_TYPE",
    -1117: "INVALID_SIDE",
    -1118: "EMPTY_NEW_CL_ORD_ID",
    -1119: "EMPTY_ORG_CL_ORD_ID",
    -1120: "BAD_INTERVAL",
    -1121: "BAD_SYMBOL",
    -1125: "INVALID_LISTEN_KEY",
    -1127: "MORE_THAN_XX_HOURS",
    -1128: "OPTIONAL_PARAMS_BAD_COMBO",
    -1130: "INVALID_PARAMETER",
    -2010: "NEW_ORDER_REJECTED",
    -2011: "CANCEL_REJECTED",
    -2013: "NO_SUCH_ORDER",
    -2014: "BAD_API_KEY_FMT",
    -2015: "REJECTED_MBX_KEY",
    -2016: "NO_SUCH_LISTEN_KEY",
    -2021: "ORDER_ARCHIVED",
    -2022: "REDUCE_ONLY_REJECT",
    -2025: "ORDER_LIST_NOT_FOUND",
    -2027: "INSUFFICIENT_BALANCE_IN_ISOLATED_MARGIN_ACCOUNT",
    -3000: "INTERNAL_ERROR",  # margin
    -3005: "ASSET_NOT_BELONG_TO_USER",
    -4003: "INVALID_MARGIN_ACCOUNT",
}


# =============================================================================
# HTTP Status Codes
# =============================================================================
HTTP_STATUS_CODES: Final[Dict[int, str]] = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    415: "Unsupported Media Type",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}


# =============================================================================
# Default Configuration
# =============================================================================
DEFAULT_CONFIG: Final[Dict[str, Any]] = {
    "timeout": 30,
    "recv_window": 5000,
    "max_retries": 3,
    "retry_delay": 1,
    "debug": False,
}


# =============================================================================
# Endpoint Weights (Rate Limit Weights)
# =============================================================================
ENDPOINT_WEIGHT_MAP: Final[Dict[str, int]] = {
    SPOT_PING_ENDPOINT: 1,
    SPOT_TIME_ENDPOINT: 1,
    SPOT_EXCHANGE_INFO_ENDPOINT: 10,
    SPOT_ORDER_BOOK_ENDPOINT: 1,
    SPOT_KLINE_ENDPOINT: 1,
    SPOT_AGG_TRADE_ENDPOINT: 1,
    SPOT_TICKER_24H_ENDPOINT: 1,
    SPOT_TICKER_PRICE_ENDPOINT: 1,
    SPOT_TICKER_BOOK_ENDPOINT: 1,
    SPOT_ORDER_ENDPOINT: 1,
    SPOT_ORDER_TEST_ENDPOINT: 1,
    SPOT_OPEN_ORDERS_ENDPOINT: 1,
    SPOT_ALL_ORDERS_ENDPOINT: 5,
    SPOT_ACCOUNT_ENDPOINT: 5,
    SPOT_MY_TRADES_ENDPOINT: 5,
    FUTURES_PING_ENDPOINT: 1,
    FUTURES_TIME_ENDPOINT: 1,
    FUTURES_EXCHANGE_INFO_ENDPOINT: 1,
    FUTURES_ORDER_BOOK_ENDPOINT: 1,
    FUTURES_KLINE_ENDPOINT: 1,
    FUTURES_ACCOUNT_ENDPOINT: 5,
    FUTURES_ORDER_ENDPOINT: 1,
    FUTURES_POSITION_RISK_ENDPOINT: 5,
    FUTURES_LEVERAGE_BRACKET_ENDPOINT: 1,
    MARGIN_ACCOUNT_ENDPOINT: 10,
    MARGIN_ORDER_ENDPOINT: 1,
    MARGIN_OPEN_ORDERS_ENDPOINT: 1,
    MARGIN_ALL_ORDERS_ENDPOINT: 10,
    MARGIN_MY_TRADES_ENDPOINT: 10,
}


# =============================================================================
# Validation Helpers
# =============================================================================
def validate_order_type(order_type: str) -> None:
    if order_type not in ORDER_TYPES:
        raise ValueError(f"Invalid order type: {order_type}, allowed: {ORDER_TYPES}")


def validate_order_side(side: str) -> None:
    if side not in ORDER_SIDES:
        raise ValueError(f"Invalid side: {side}, allowed: {ORDER_SIDES}")

def validate_order_status(status: str) -> None:
    if status not in ORDER_STATUS:
        raise ValueError(f"Invalid order status: {status}, allowed: {ORDER_STATUS}")

def validate_working_type(working_type: str) -> None:
    if working_type not in WORKING_TYPES:
        raise ValueError(f"Invalid working type: {working_type}, allowed: {WORKING_TYPES}")

def validate_margin_type(margin_type: str) -> None:
    if margin_type not in MARGIN_TYPES:
        raise ValueError(f"Invalid margin type: {margin_type}, allowed: {MARGIN_TYPES}")
        

def validate_time_in_force(tif: str) -> None:
    if tif not in TIME_IN_FORCE:
        raise ValueError(f"Invalid timeInForce: {tif}, allowed: {TIME_IN_FORCE}")


def validate_interval(interval: str, futures: bool = False) -> None:
    intervals = INTERVALS_FUTURES if futures else INTERVALS_SPOT
    if interval not in intervals:
        raise ValueError(f"Invalid interval: {interval}, allowed: {intervals}")


# =============================================================================
# Exception Mapping
# =============================================================================
class BinanceAPIException(Exception):
    """Base Binance API exception."""


def map_error_code(code: int, message: Optional[str] = None) -> BinanceAPIException:
    """
    Map Binance error code to Exception.
    """
    desc = ERROR_CODES.get(code, "UNKNOWN_ERROR")
    return BinanceAPIException(f"[{code}] {desc}. {message or ''}")
    



class BinanceConfig(BaseSettings):
    """Binance configuration with environment variables support"""
    max_concurrent_users: int = 100
    request_timeout: int = 30
    cache_ttl: int = 300
    enable_metrics: bool = True
    enable_audit: bool = True
    database_url: str = "data/apikeys.db"
    master_key: str  # Required
    
    class Config:
        env_prefix = "BINANCE_"

# Mevcut DEFAULT_CONFIG'u güncelle
DEFAULT_CONFIG.update({
    "max_concurrent_users": 100,
    "cache_ttl": 300,
    "enable_metrics": True,
    "enable_audit": True
})
