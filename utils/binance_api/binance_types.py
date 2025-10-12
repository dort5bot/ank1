# utils/binance_api/binance_types.py
"""
Binance API type definitions with validation and enhanced error handling.
Includes:
- Literal types for OrderSide, OrderType, TimeInForce, Interval, PositionSide, MarginType
- TypedDict for Spot, Futures, WebSocket, ExchangeInfo
- Enhanced request types with validation helpers
- Detailed ErrorResponse mapping and exception helpers

Algo trading stratejileri geliştirebilirsiniz
Risk management sistemleri kurabilirsiniz
Advanced order types (OCO, Bracket, Scale) kullanabilirsiniz
Detaylı market analysis yapabilirsiniz
High-frequency data processing yapabilirsiniz

"""

from typing import TypedDict, List, Union, Optional, Dict, Any, Literal
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

import logging

 
logger = logging.getLogger(__name__)

class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    
class UrgencyLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM" 
    HIGH = "HIGH"

@dataclass
class TradingSignal:
    symbol: str
    side: OrderSide
    confidence: float
    price_targets: List[Decimal]
    stop_loss: Decimal
    timestamp: int
    
# ------------------------------
# Literal Types
# ------------------------------
OrderSide = Literal["BUY", "SELL"]
OrderType = Literal["LIMIT", "MARKET", "STOP_LOSS", "STOP_LOSS_LIMIT", 
                   "TAKE_PROFIT", "TAKE_PROFIT_LIMIT", "LIMIT_MAKER"]
TimeInForce = Literal["GTC", "IOC", "FOK"]
Interval = Literal["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
OrderStatus = Literal["NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED", "PENDING_CANCEL", "REJECTED", "EXPIRED"]
PositionSide = Literal["BOTH", "LONG", "SHORT"]
MarginType = Literal["ISOLATED", "CROSSED"]
WorkingType = Literal["MARK_PRICE", "CONTRACT_PRICE"]
NewOrderRespType = Literal["ACK", "RESULT", "FULL"]
SelfTradePreventionMode = Literal["EXPIRE_TAKER", "EXPIRE_MAKER", "EXPIRE_BOTH", "NONE"]

# ------------------------------
# Base Types
# ------------------------------
Symbol = str
Asset = str

# ------------------------------
# TypedDict Definitions
# ------------------------------
class Kline(TypedDict):
    open_time: int
    open: str
    high: str
    low: str
    close: str
    volume: str
    close_time: int
    quote_asset_volume: str
    number_of_trades: int
    taker_buy_base_asset_volume: str
    taker_buy_quote_asset_volume: str
    ignore: str

class OrderBookEntry(TypedDict):
    price: str
    quantity: str

class OrderBook(TypedDict):
    lastUpdateId: int
    bids: List[List[str]]
    asks: List[List[str]]

class Ticker(TypedDict):
    symbol: str
    priceChange: str
    priceChangePercent: str
    weightedAvgPrice: str
    prevClosePrice: str
    lastPrice: str
    lastQty: str
    bidPrice: str
    askPrice: str
    openPrice: str
    highPrice: str
    lowPrice: str
    volume: str
    quoteVolume: str
    openTime: int
    closeTime: int
    firstId: int
    lastId: int
    count: int

class Balance(TypedDict):
    asset: str
    free: str
    locked: str

class AccountInfo(TypedDict):
    makerCommission: int
    takerCommission: int
    buyerCommission: int
    sellerCommission: int
    canTrade: bool
    canWithdraw: bool
    canDeposit: bool
    updateTime: int
    accountType: str
    balances: List[Balance]
    permissions: List[str]

class Order(TypedDict):
    symbol: str
    orderId: int
    orderListId: int
    clientOrderId: str
    price: str
    origQty: str
    executedQty: str
    cummulativeQuoteQty: str
    status: OrderStatus
    timeInForce: TimeInForce
    type: OrderType
    side: OrderSide
    stopPrice: str
    icebergQty: str
    time: int
    updateTime: int
    isWorking: bool
    origQuoteOrderQty: str

class OrderResponse(Order):
    transactTime: int

class Trade(TypedDict):
    id: int
    price: str
    qty: str
    quoteQty: str
    time: int
    isBuyerMaker: bool
    isBestMatch: bool


# ------------------------------
#  Spot Ek
# ------------------------------
class OCOOrder(TypedDict):
    orderListId: int
    contingencyType: str
    listStatusType: str
    listOrderStatus: str
    listClientOrderId: str
    transactionTime: int
    symbol: str
    orders: List[Dict[str, Any]]
    orderReports: List[Dict[str, Any]]

class DustLog(TypedDict):
    total: int
    userAssetDribblets: List[Dict[str, Any]]


# ------------------------------
# Futures Types
# ------------------------------
class Position(TypedDict):
    symbol: str
    positionAmt: str
    entryPrice: str
    markPrice: str
    unRealizedProfit: str
    liquidationPrice: str
    leverage: str
    maxNotionalValue: str
    marginType: MarginType
    isolatedMargin: str
    isAutoAddMargin: str
    positionSide: PositionSide
    notional: str
    isolatedWallet: str
    updateTime: int

class FuturesBalance(TypedDict):
    accountAlias: str
    asset: str
    balance: str
    crossWalletBalance: str
    crossUnPnl: str
    availableBalance: str
    maxWithdrawAmount: str

class FuturesAccount(TypedDict):
    assets: List[FuturesBalance]
    positions: List[Position]
    canDeposit: bool
    canTrade: bool
    canWithdraw: bool
    feeTier: int
    updateTime: int
    totalInitialMargin: str
    totalMaintMargin: str
    totalWalletBalance: str
    totalUnrealizedProfit: str
    totalMarginBalance: str
    totalPositionInitialMargin: str
    totalOpenOrderInitialMargin: str
    totalCrossWalletBalance: str
    totalCrossUnPnl: str
    availableBalance: str
    maxWithdrawAmount: str

class FuturesOrderBook(TypedDict):
    lastUpdateId: int
    E: int  # Message output time
    T: int  # Transaction time
    bids: List[List[str]]
    asks: List[List[str]]

class OpenInterest(TypedDict):
    symbol: str
    openInterest: str
    time: int

class FundingRate(TypedDict):
    symbol: str
    fundingRate: str
    fundingTime: int
    markPrice: str

class LeverageInfo(TypedDict):
    symbol: str
    leverage: int
    maxNotionalValue: str

class IncomeHistory(TypedDict):
    symbol: str
    incomeType: str
    income: str
    asset: str
    time: int
    info: str
    tranId: str
    tradeId: str


# ------------------------------
# Futures Advanced Data
# ------------------------------
class LongShortRatio(TypedDict):
    symbol: str
    longAccount: str
    shortAccount: str
    longShortRatio: str
    timestamp: int

class TopTraderLongShortRatio(LongShortRatio):
    longPosition: str
    shortPosition: str

class TakerBuySellVolume(TypedDict):
    symbol: str
    pair: str
    contractType: str
    takerBuyVol: str
    takerSellVol: str
    takerBuyVolValue: str
    takerSellVolValue: str
    timestamp: int

class Basis(TypedDict):
    symbol: str
    futuresPrice: str
    indexPrice: str
    basis: str
    basisRate: str
    timestamp: int



# ------------------------------
# Margin Trading
# ------------------------------

class MarginBalance(Balance):
    asset: str
    free: str
    borrowed: str
    interest: str
    locked: str
    netAsset: str


class MarginAccountInfo(TypedDict):
    borrowEnabled: bool
    marginLevel: str
    totalAssetOfBtc: str
    totalLiabilityOfBtc: str
    totalNetAssetOfBtc: str
    tradeEnabled: bool
    transferEnabled: bool
    userAssets: List[MarginBalance]


# ------------------------------
# Request Types
# ------------------------------
class OrderRequest(TypedDict, total=False):
    symbol: str
    side: OrderSide
    type: OrderType
    quantity: Optional[str]
    quoteOrderQty: Optional[str]
    price: Optional[str]
    timeInForce: Optional[TimeInForce]
    newClientOrderId: Optional[str]
    stopPrice: Optional[str]
    icebergQty: Optional[str]
    newOrderRespType: Optional[str]
    recvWindow: Optional[int]

class FuturesOrderRequest(OrderRequest):
    positionSide: Optional[PositionSide]
    reduceOnly: Optional[bool]
    activationPrice: Optional[str]
    callbackRate: Optional[str]
    closePosition: Optional[bool]
    workingType: Optional[str]
    priceRate: Optional[str]
    priceProtect: Optional[bool]



# ------------------------------
# Advanced Order Types
# ------------------------------
class OCOOrderRequest(TypedDict, total=False):
    symbol: str
    listClientOrderId: Optional[str]
    side: OrderSide
    quantity: str
    limitClientOrderId: Optional[str]
    price: str
    limitIcebergQty: Optional[str]
    stopClientOrderId: Optional[str]
    stopPrice: str
    stopLimitPrice: Optional[str]
    stopIcebergQty: Optional[str]
    stopLimitTimeInForce: Optional[TimeInForce]
    newOrderRespType: Optional[NewOrderRespType]
    recvWindow: Optional[int]

class TrailingStopOrderRequest(OrderRequest):
    callbackRate: str
    workingType: WorkingType

class IcebergOrderRequest(OrderRequest):
    icebergQty: str


# Algo Order Tipleri:
# ------------------------------
class TWAPOrderRequest(TypedDict, total=False):
    symbol: str
    side: OrderSide
    quantity: str
    duration: int  # minutes
    slices: int
    limitPrice: Optional[str]
    recvWindow: Optional[int]

class VPOrderRequest(TypedDict, total=False):
    symbol: str
    side: OrderSide
    quantity: str
    urgency: Literal["LOW", "MEDIUM", "HIGH"]
    participationRate: float
    recvWindow: Optional[int]


# Strategy Order Tipleri:
# ------------------------------
class BracketOrderRequest(TypedDict, total=False):
    symbol: str
    side: OrderSide
    quantity: str
    price: str
    stopLossPrice: str
    takeProfitPrice: str
    stopLossLimitPrice: Optional[str]
    takeProfitLimitPrice: Optional[str]
    recvWindow: Optional[int]

class ScaleOrderRequest(TypedDict, total=False):
    symbol: str
    side: OrderSide
    totalQuantity: str
    initialPrice: str
    scaleSteps: int
    priceDecrement: str
    quantityPerStep: str
    recvWindow: Optional[int]


# ------------------------------
# Advanced Market Data Types
# ------------------------------
class DepthSnapshot(TypedDict):
    lastUpdateId: int
    bids: List[List[str]]
    asks: List[List[str]]

class Ticker24hr(TypedDict):
    symbol: str
    priceChange: str
    priceChangePercent: str
    weightedAvgPrice: str
    prevClosePrice: str
    lastPrice: str
    lastQty: str
    bidPrice: str
    askPrice: str
    openPrice: str
    highPrice: str
    lowPrice: str
    volume: str
    quoteVolume: str
    openTime: int
    closeTime: int
    firstId: int
    lastId: int
    count: int

class TickerPrice(TypedDict):
    symbol: str
    price: str

class BookTicker(TypedDict):
    symbol: str
    bidPrice: str
    bidQty: str
    askPrice: str
    askQty: str


# ------------------------------
# Market Depth & Liquidity
# ------------------------------
class LiquiditySnapshot(TypedDict):
    symbol: str
    timestamp: int
    bidLiquidity: List[Dict[str, str]]  # price -> quantity
    askLiquidity: List[Dict[str, str]]
    totalBidLiquidity: str
    totalAskLiquidity: str

class MarketDepth(TypedDict):
    symbol: str
    timestamp: int
    depth_5: Dict[str, List[List[str]]]  # 5 levels
    depth_10: Dict[str, List[List[str]]]  # 10 levels
    depth_20: Dict[str, List[List[str]]]  # 20 levels


# ------------------------------
# WebSocket Types
# ------------------------------
class WSMessage(TypedDict):
    stream: str
    data: Dict[str, Any]

class WSKlineData(TypedDict):
    t: int
    T: int
    s: str
    i: str
    o: str
    c: str
    h: str
    l: str
    v: str
    n: int
    x: bool
    q: str
    V: str
    Q: str
    B: str

class WSOrderUpdate(TypedDict):
    s: str  # symbol
    c: str  # client order id
    S: OrderSide
    o: OrderType
    f: TimeInForce
    q: str  # quantity
    p: str  # price
    x: str  # execution type
    X: OrderStatus
    i: int  # order id
    l: str  # last executed quantity
    z: str  # cumulative filled quantity
    L: str  # last executed price
    n: str  # commission
    T: int  # transaction time
    t: int  # trade id

class WSAggTrade(TypedDict):
    e: str  # Event type
    E: int  # Event time
    s: str  # Symbol
    a: int  # Aggregate trade ID
    p: str  # Price
    q: str  # Quantity
    f: int  # First trade ID
    l: int  # Last trade ID
    T: int  # Trade time
    m: bool # Is buyer market maker?

class WSAccountUpdate(TypedDict):
    e: str  # Event type
    E: int  # Event time
    m: int  # Maker commission
    t: int  # Taker commission
    b: int  # Buyer commission
    s: int  # Seller commission
    T: bool # Can trade?
    W: bool # Can withdraw?
    D: bool # Can deposit?
    u: int  # Last update time
    B: List[Balance]  # Balances


# ------------------------------
# Advanced WebSocket Types
# ------------------------------
class WSDepthUpdate(TypedDict):
    e: str  # event type
    E: int  # event time
    s: str  # symbol
    U: int  # first update ID in event
    u: int  # final update ID in event
    b: List[List[str]]  # bids to update
    a: List[List[str]]  # asks to update

class WSTicker24hr(TypedDict):
    e: str  # event type
    E: int  # event time
    s: str  # symbol
    p: str  # price change
    P: str  # price change percent
    w: str  # weighted average price
    x: str  # previous day's close price
    c: str  # current day's close price
    Q: str  # close trade's quantity
    b: str  # best bid price
    B: str  # best bid quantity
    a: str  # best ask price
    A: str  # best ask quantity
    o: str  # open price
    h: str  # high price
    l: str  # low price
    v: str  # total traded base asset volume
    q: str  # total traded quote asset volume
    O: int  # statistics open time
    C: int  # statistics close time
    F: int  # first trade ID
    L: int  # last trade ID
    n: int  # total number of trades

class WSBookTicker(TypedDict):
    u: int  # order book updateId
    s: str  # symbol
    b: str  # best bid price
    B: str  # best bid quantity
    a: str  # best ask price
    A: str  # best ask quantity
    


# ------------------------------
# Risk Management Types
# ------------------------------
class RiskParameters(TypedDict):
    max_position_size_percent: float
    max_risk_per_trade_percent: float
    daily_loss_limit_percent: float
    max_leverage: int
    allowed_symbols: List[str]
    blacklisted_symbols: List[str]

class TradePlan(TypedDict):
    symbol: str
    side: OrderSide
    entry_price: str
    stop_loss: str
    take_profit: str
    quantity: str
    risk_reward_ratio: float
    position_size_percent: float






# ------------------------------
# Response & Error Types
# ------------------------------
class BinanceResponse(TypedDict):
    success: bool
    data: Optional[Union[Dict[str, Any], List[Any]]]
    error: Optional[str]
    error_code: Optional[int]
    timestamp: int

class ErrorResponse(TypedDict):
    code: int
    msg: str


# ------------------------------
# Health Status
# ------------------------------
class HealthStatus(TypedDict):
    status: Literal["HEALTHY", "DEGRADED", "CRITICAL"]
    issues: Optional[List[str]]
    warnings: Optional[List[str]]
    metrics: Optional[Dict[str, Any]]
    timestamp: float


# ------------------------------
# Exchange Info Types
# ------------------------------
class RateLimit(TypedDict):
    rateLimitType: str
    interval: str
    intervalNum: int
    limit: int
    weight_used: int
    weight_limit: int
    last_reset_time: float
    requests_per_second: float

class SymbolFilter(TypedDict):
    filterType: str
    minPrice: Optional[str]
    maxPrice: Optional[str]
    tickSize: Optional[str]
    minQty: Optional[str]
    maxQty: Optional[str]
    stepSize: Optional[str]
    minNotional: Optional[str]
    limit: Optional[int]
    maxNumOrders: Optional[int]
    maxNumAlgoOrders: Optional[int]

class SymbolInfo(TypedDict):
    symbol: str
    status: str
    baseAsset: str
    baseAssetPrecision: int
    quoteAsset: str
    quotePrecision: int
    quoteAssetPrecision: int
    baseCommissionPrecision: int
    quoteCommissionPrecision: int
    orderTypes: List[OrderType]
    icebergAllowed: bool
    ocoAllowed: bool
    quoteOrderQtyMarketAllowed: bool
    allowTrailingStop: bool
    cancelReplaceAllowed: bool
    isSpotTradingAllowed: bool
    isMarginTradingAllowed: bool
    filters: List[SymbolFilter]
    permissions: List[str]
    defaultSelfTradePreventionMode: str
    allowedSelfTradePreventionModes: List[str]

class ExchangeInfo(TypedDict):
    timezone: str
    serverTime: int
    rateLimits: List[RateLimit]
    exchangeFilters: List[Any]
    symbols: List[SymbolInfo]


# ------------------------------
#  Performance Metrics
# ------------------------------
class PerformanceMetrics(TypedDict):
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_duration: int
    
    
# ------------------------------
# Validation Helpers
# ------------------------------
def validate_order_request(req: OrderRequest, symbol_info: SymbolInfo):
    """
    Validate order request against SymbolInfo filters.
    Raises ValueError on invalid input.
    """
    for f in symbol_info["filters"]:
        if f["filterType"] == "LOT_SIZE" and "minQty" in f and req.get("quantity"):
            if float(req["quantity"]) < float(f["minQty"]):
                raise ValueError(f"Quantity {req['quantity']} below minQty {f['minQty']}")
            if float(req["quantity"]) > float(f["maxQty"]):
                raise ValueError(f"Quantity {req['quantity']} above maxQty {f['maxQty']}")
        if f["filterType"] == "PRICE_FILTER" and req.get("price"):
            if float(req["price"]) < float(f["minPrice"]):
                raise ValueError(f"Price {req['price']} below minPrice {f['minPrice']}")
            if float(req["price"]) > float(f["maxPrice"]):
                raise ValueError(f"Price {req['price']} above maxPrice {f['maxPrice']}")


def validate_futures_order_request(req: FuturesOrderRequest, symbol_info: SymbolInfo):
    """Futures order validation with additional checks"""
    validate_order_request(req, symbol_info)
    
    # Futures-specific validations
    if req.get("reduceOnly") and req.get("closePosition"):
        raise ValueError("reduceOnly and closePosition cannot both be True")
    
    # Leverage validation
    if req.get("positionSide") == "BOTH" and req.get("reduceOnly"):
        logger.warning("reduceOnly may not work as expected with BOTH position side")


def validate_interval(interval: str) -> bool:
    """Validate kline interval"""
    valid_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
    return interval in valid_intervals

def get_symbol_filters(symbol_info: SymbolInfo, filter_type: str) -> List[SymbolFilter]:
    """Get specific filters by type from symbol info"""
    return [f for f in symbol_info["filters"] if f["filterType"] == filter_type]

def calculate_lot_size_quantity(quantity: float, symbol_info: SymbolInfo) -> str:
    """Calculate valid quantity based on LOT_SIZE filter"""
    lot_filter = next((f for f in symbol_info["filters"] if f["filterType"] == "LOT_SIZE"), None)
    if lot_filter and "stepSize" in lot_filter:
        step = float(lot_filter["stepSize"])
        adjusted_qty = round(quantity / step) * step
        return f"{adjusted_qty:.8f}".rstrip('0').rstrip('.')
    return str(quantity)


# ------------------------------
# Advanced Validation Helpers
# ------------------------------

def validate_oco_order_request(req: OCOOrderRequest, symbol_info: SymbolInfo):
    """Validate OCO order request"""
    # Price validation
    price_filter = get_symbol_filters(symbol_info, "PRICE_FILTER")[0]
    if float(req["price"]) < float(price_filter["minPrice"]):
        raise ValueError(f"Limit price below minimum")
    if float(req["stopPrice"]) < float(price_filter["minPrice"]):
        raise ValueError(f"Stop price below minimum")
    
    # Quantity validation
    lot_filter = get_symbol_filters(symbol_info, "LOT_SIZE")[0]
    if float(req["quantity"]) < float(lot_filter["minQty"]):
        raise ValueError(f"Quantity below minimum")

def validate_bracket_order(req: BracketOrderRequest, symbol_info: SymbolInfo):
    """Validate bracket order parameters"""
    if float(req["stopLossPrice"]) >= float(req["price"]):
        raise ValueError("Stop loss must be below entry price for LONG")
    if float(req["takeProfitPrice"]) <= float(req["price"]):
        raise ValueError("Take profit must be above entry price for LONG")

def calculate_position_size(account_balance: float, risk_percent: float, entry_price: float, stop_loss: float) -> str:
    """Calculate position size based on risk management"""
    risk_amount = account_balance * (risk_percent / 100)
    price_diff = abs(entry_price - stop_loss)
    position_size = risk_amount / price_diff
    return f"{position_size:.8f}"

def get_optimal_quantity(symbol: str, price: float, max_position_percent: float, account_balance: float) -> str:
    """Calculate optimal quantity based on account balance and risk limits"""
    max_investment = account_balance * (max_position_percent / 100)
    quantity = max_investment / price
    return f"{quantity:.8f}"

# ------------------------------
# Error Mapping Helper
# ------------------------------
BINANCE_ERROR_MAP = {
    -1000: "Unknown error",
    -1001: "Disconnected from server", 
    -1002: "Unauthorized request",
    -1003: "Too many requests",
    -1006: "Unexpected response",
    -1007: "Timeout waiting for response",
    -1013: "Invalid quantity/price",
    -1014: "Unsupported order combination",
    -1015: "Too many new orders",
    -1016: "Service shutting down",
    -1020: "Unsupported operation",
    -1021: "Invalid timestamp",
    -1022: "Invalid signature",
    -1100: "Illegal characters in parameter",
    -1101: "Too many parameters",
    -1102: "Mandatory parameter empty or malformed",
    -1103: "Unknown parameter",
    -1104: "Unread parameters",
    -1105: "Parameter empty",
    -1106: "Parameter not required",
    -1111: "Bad precision",
    -1112: "No depth for symbol",
    -1114: "TIF not required",
    -1115: "Invalid TIF",
    -1116: "Invalid order type",
    -1117: "Invalid side",
    -1118: "Empty new client order id",
    -1119: "Empty original client order id",
    -1120: "Bad interval",
    -1121: "Bad symbol",
    -1125: "Invalid listen key",
    -1127: "More than XX hours",
    -1128: "Combination of optional parameters invalid",
    -1130: "Invalid data sent for a parameter",
    -2008: "Invalid Api-Key ID",
    -2010: "Order would trigger liquidation",
    -2011: "Order canceled",
    -2013: "Order not found",
    -2014: "Insufficient balance",
    -2015: "Invalid order type for cancel replace",
    -2016: "Invalid cancel replace order",
    -2018: "Balance not sufficient for self trade prevention",
    -2019: "Self trade prevention threshold exceeded",
}

class BinanceAPIError(Exception):
    """Detailed Binance API exception with code and message."""
    def __init__(self, code: int, msg: str):
        self.code = code
        self.msg = msg
        super().__init__(f"[Binance API Error {code}] {msg}")

def raise_if_error(response: Union[ErrorResponse, BinanceResponse]):
    """Raise BinanceAPIError if response contains an error."""
    if isinstance(response, dict):
        code = response.get("code") or response.get("error_code")
        msg = response.get("msg") or response.get("error")
        if code is not None:
            raise BinanceAPIError(code=code, msg=msg or BINANCE_ERROR_MAP.get(code, "Unknown error"))

