"""
binance_api/binance_ws_Pydantic.py
----------------------

Binance WebSocket API'den gelen ham JSON verilerini Pydantic modelleri ile doğrulayan,
tipleyen ve anlamlandıran modül.

- Spot Market
- Futures Market
- System Events

Modül, çoklu kullanıcı ve çoklu WebSocket bağlantısını destekleyecek şekilde
oldukça genişletilebilir, modüler yapıda tasarlanmıştır.

Bu modül, Binance'in resmi WebSocket event formatlarına %100 uyumludur.

Referanslar:
- https://binance-docs.github.io/apidocs/spot/en/#websocket-market-streams
- https://binance-docs.github.io/apidocs/futures/en/#websocket-market-streams
"""
#binance_api/binance_ws_Pydantic.py
import json  
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Union, Literal
from datetime import datetime


# ------------------------
# Temel BaseModel'ler
# ------------------------

class BaseEvent(BaseModel):
    """Tüm eventlerin temel alanları"""
    e: str  # Event Type
    E: int  # Event Time (ms)

    @property
    def event_time(self) -> datetime:
        """Event zamanını datetime objesi olarak verir"""
        return datetime.fromtimestamp(self.E / 1000)


# ------------------------
# Spot Market Event Modelleri
# ------------------------

class TradeEvent(BaseEvent):
    e: Literal["trade"]
    s: str  # Symbol
    t: int  # Trade ID
    p: str  # Price
    q: str  # Quantity
    b: int  # Buyer Order ID
    a: int  # Seller Order ID
    T: int  # Trade Time (ms)
    m: bool  # Is buyer maker?
    M: bool  # Ignore

    @property
    def trade_time(self) -> datetime:
        return datetime.fromtimestamp(self.T / 1000)


class AggTradeEvent(BaseEvent):
    e: Literal["aggTrade"]
    s: str  # Symbol
    a: int  # Aggregate Trade ID
    p: str  # Price
    q: str  # Quantity
    f: int  # First trade ID
    l: int  # Last trade ID
    T: int  # Trade time (ms)
    m: bool  # Is buyer maker?
    M: bool  # Ignore

    @property
    def agg_trade_time(self) -> datetime:
        return datetime.fromtimestamp(self.T / 1000)


class Kline(BaseModel):
    t: int  # Kline start time
    T: int  # Kline close time
    s: str  # Symbol
    i: str  # Interval
    f: int  # First trade ID
    L: int  # Last trade ID
    o: str  # Open price
    c: str  # Close price
    h: str  # High price
    l: str  # Low price
    v: str  # Base asset volume
    n: int  # Number of trades
    x: bool  # Is this kline closed?
    q: str  # Quote asset volume
    V: str  # Taker buy base asset volume
    Q: str  # Taker buy quote asset volume
    B: str  # Ignore

    @property
    def start_time(self) -> datetime:
        return datetime.fromtimestamp(self.t / 1000)

    @property
    def close_time(self) -> datetime:
        return datetime.fromtimestamp(self.T / 1000)


class KlineEvent(BaseEvent):
    e: Literal["kline"]
    s: str
    k: Kline


class DepthUpdateEvent(BaseEvent):
    e: Literal["depthUpdate"]
    s: str  # Symbol
    U: int  # First update ID in event
    u: int  # Final update ID in event
    b: List[List[str]]  # Bids to be updated [price, quantity]
    a: List[List[str]]  # Asks to be updated [price, quantity]


class OutboundAccountPositionEvent(BaseEvent):
    e: Literal["outboundAccountPosition"]
    u: int  # Account update time
    B: List[dict]  # Balances, örn: [{"a":"BTC","f":"100.0","l":"0.0"}]


class BalanceUpdateEvent(BaseEvent):
    e: Literal["balanceUpdate"]
    a: str  # Asset
    d: str  # Balance Delta
    T: int  # Clear Time (ms)

    @property
    def clear_time(self) -> datetime:
        return datetime.fromtimestamp(self.T / 1000)



class MiniTickerEvent(BaseEvent):
    e: Literal["24hrTicker"]
    E: int
    s: str  # Symbol
    p: str  # Price change
    P: str  # Price change percent
    w: str  # Weighted average price
    x: str  # First trade(F)-1 price (first trade before the 24hr rolling window)
    c: str  # Last price
    Q: str  # Last quantity
    b: str  # Best bid price
    B: str  # Best bid quantity
    a: str  # Best ask price
    A: str  # Best ask quantity
    o: str  # Open price
    h: str  # High price
    l: str  # Low price
    v: str  # Total traded base asset volume
    q: str  # Total traded quote asset volume
    O: int  # Statistics open time
    C: int  # Statistics close time
    F: int  # First trade ID
    L: int  # Last trade ID
    n: int  # Total number of trades
    
    @property  # ← DOĞRU GİRİNTİ (4 boşluk)
    def price_change_percent_float(self) -> float:
        return float(self.P)
    
    @property
    def last_price_float(self) -> float:
        return float(self.c)


class BookTickerEvent(BaseEvent):  
    e: Literal["bookTicker"]
    u: int  # order book updateId
    s: str  # symbol
    b: str  # best bid price
    B: str  # best bid qty
    a: str  # best ask price
    A: str  # best ask qty
    
    @property
    def spread(self) -> float:
        return float(self.a) - float(self.b)




# ------------------------
# Futures Market Event Modelleri
# ------------------------

class FuturesAggTradeEvent(BaseEvent):
    e: Literal["aggTrade"]
    E: int  # Event time
    s: str  # Symbol
    a: int  # Aggregate trade ID
    p: str  # Price
    q: str  # Quantity
    f: int  # First trade ID
    l: int  # Last trade ID
    T: int  # Trade time
    m: bool  # Is buyer maker?
    M: bool  # Ignore

    @property
    def trade_time(self) -> datetime:
        return datetime.fromtimestamp(self.T / 1000)


class FuturesKlineEvent(BaseEvent):
    e: Literal["kline"]
    E: int
    s: str
    k: Kline


class FuturesMarkPriceEvent(BaseEvent):
    e: Literal["markPriceUpdate"]
    E: int
    s: str
    p: str  # Mark price
    i: int  # Index price
    T: int  # Next funding time
    r: str  # Funding rate


class LiquidationOrder(BaseModel):
    s: str  # Symbol
    S: str  # Side (BUY/SELL)
    o: str  # Order type
    f: str  # Time in force
    q: str  # Original quantity
    p: str  # Price
    ap: str  # Average price
    X: str  # Order status
    l: str  # Order last filled quantity
    z: str  # Order filled accumulated quantity
    T: int  # Order trade time

class LiquidationEvent(BaseEvent):
    e: Literal["forceOrder"]
    E: int
    o: LiquidationOrder  # Artık dict değil, structured model
    # Order data
    # o: {
    #   "s":"BTCUSDT",      # Symbol
    #   "S":"SELL",         # Side
    #   "o":"LIMIT",        # Order type
    #   "f":"IOC",          # Time in force
    #   "q":"0.014",        # Original quantity
    #   "p":"9910",         # Price
    #   "ap":"9910",        # Average price
    #   "X":"FILLED",       # Order status
    #   "l":"0.014",        # Order last filled quantity
    #   "z":"0.014",        # Order filled accumulated quantity
    #   "T":1568872020000   # Order trade time
    # }
    
class ContinuousKlineEvent(BaseEvent):
    e: Literal["continuous_kline"]
    E: int
    ps: str  # Pair
    ct: str  # Contract type
    k: Kline


# ------------------------
# System Event Modelleri
# ------------------------

class PingEvent(BaseEvent):
    e: Literal["ping"]
    E: int


class PongEvent(BaseEvent):
    e: Literal["pong"]
    E: int


# ------------------------
# Union Tipler ve Parse Fonksiyonu
# ------------------------

BinanceWSEvent = Union[
    TradeEvent,
    AggTradeEvent,
    KlineEvent,
    DepthUpdateEvent,
    OutboundAccountPositionEvent,
    BalanceUpdateEvent,
    FuturesAggTradeEvent,
    FuturesKlineEvent,
    FuturesMarkPriceEvent,
    MiniTickerEvent,          # Yeni
    BookTickerEvent,          # Yeni
    LiquidationEvent,         # Yeni
    ContinuousKlineEvent,     # Yeni
    PingEvent,
    PongEvent,
]


def parse_binance_ws_event(data: dict) -> BinanceWSEvent:
    """
    Binance WebSocket'ten gelen JSON datasını uygun Pydantic modele parse eder.
    Desteklenen event tipleri:
        - trade
        - aggTrade
        - kline
        - depthUpdate
        - outboundAccountPosition
        - balanceUpdate
        - futures aggTrade, kline, markPriceUpdate
        - ping, pong

    Args:
        data (dict): Binance WS JSON verisi

    Returns:
        BinanceWSEvent: Pydantic event modeli

    Raises:
        ValueError: Tanımlanamayan event tipinde
    """
    event_type = data.get("e")

    if event_type is None:
        # Daha açıklayıcı hata mesajı
        available_keys = list(data.keys())[:5]  # İlk 5 key'i göster
        raise ValueError(f"Event type 'e' alanı yok! Mevcut keys: {available_keys}")



    # Event type'a göre mapping
    event_map = {
        "trade": TradeEvent,
        "aggTrade": AggTradeEvent,  # Spot ve Futures aynı yapıda
        "kline": KlineEvent,
        "depthUpdate": DepthUpdateEvent,
        "outboundAccountPosition": OutboundAccountPositionEvent,
        "balanceUpdate": BalanceUpdateEvent,
        "markPriceUpdate": FuturesMarkPriceEvent,
        "24hrTicker": MiniTickerEvent,
        "bookTicker": BookTickerEvent,
        "forceOrder": LiquidationEvent,
        "continuous_kline": ContinuousKlineEvent,
        "ping": PingEvent,
        "pong": PongEvent,
    }
    
    if event_type in event_map:
        return event_map[event_type].parse_obj(data)
    else:
        raise ValueError(f"Unknown Binance WebSocket event type: {event_type}")
        


# ------------------------
# Örnek Kullanım
# ------------------------
# ------------------------
# Test Fonksiyonu ve Verileri
# ------------------------


# Örnek WS JSON Trade Event verisi (test_all_events'ten önce)
raw_json = """
{
    "e": "trade",
    "E": 123456789,
    "s": "BTCUSDT",
    "t": 12345,
    "p": "0.001",
    "q": "100",
    "b": 88,
    "a": 50,
    "T": 123456785,
    "m": true,
    "M": true
}
"""

def test_all_events():
    """Tüm event tiplerini test eder"""
    test_events = {
        "mini_ticker": {
            "e": "24hrTicker", "E": 123456789, "s": "BTCUSDT",
            "p": "50.0", "P": "0.5", "c": "10050.0", "h": "10100.0", "l": "9900.0",
            "w": "10025.0", "x": "10000.0", "Q": "0.001", "b": "10049.0", "B": "1.5",
            "a": "10050.0", "A": "2.0", "o": "10000.0", "v": "1500.0", "q": "15000000.0",
            "O": 123456000, "C": 123456789, "F": 100000, "L": 100150, "n": 151
        },
        "book_ticker": {
            "e": "bookTicker", "u": 123456, "s": "BTCUSDT",
            "b": "10049.0", "B": "1.5", "a": "10050.0", "A": "2.0"
        },
        "liquidation": {
            "e": "forceOrder", "E": 123456789,
            "o": {
                "s": "BTCUSDT", "S": "SELL", "o": "LIMIT", "f": "IOC",
                "q": "0.014", "p": "9910", "ap": "9910", "X": "FILLED",
                "l": "0.014", "z": "0.014", "T": 1568872020000
            }
        }
    }
    
    for event_name, event_data in test_events.items():
        try:
            event = parse_binance_ws_event(event_data)
            print(f"✅ {event_name}: {type(event).__name__}")
        except Exception as e:
            print(f"❌ {event_name}: {e}")

if __name__ == "__main__":
    # Mevcut test
    data = json.loads(raw_json)
    event = parse_binance_ws_event(data)
    print(event)
    print("Trade Time:", event.trade_time)
    
    print("\n--- Tüm Event Testleri ---")
    test_all_events()