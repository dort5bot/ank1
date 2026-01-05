# utils/binance_api/binance_websocket.py    (v9 - acıklamalar en sonda >

import aiohttp
import asyncio
import time
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable, AsyncGenerator, Union
from urllib.parse import urlencode

# utils/binance/binance_constants.py'den import
from .binance_constants import (
    WS_BASE_URL,
    WS_FUTURES_URL,
    WS_TESTNET_URL,
    WS_TESTNET_FUTURES_URL,
    WS_STREAMS,
    WS_STREAMS_REVERSE
)

from .binance_exceptions import BinanceWebSocketError
#from utils.apikey_manager import APIKeyManager kod içinde
try:
    from utils.apikey_manager import APIKeyManager
except ImportError:
    APIKeyManager = None
    logger.warning("APIKeyManager not available")


logger = logging.getLogger(__name__)

# Constants
KEEPALIVE_INTERVAL = 30 * 60  # 30 dakika
RECEIVE_TIMEOUT = 1.0

class BinanceWSClient:

    def __init__(
        self,
        futures: bool = False,
        testnet: bool = False,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        user_id: Optional[int] = None,
        reconnect_interval: int = 5,
        max_reconnect_attempts: int = 10,
        ping_interval: int = 30,
        on_message: Optional[Callable[[dict], Awaitable[None]]] = None,
        on_error: Optional[Callable[[Exception], Awaitable[None]]] = None,
        on_connect: Optional[Callable[[], Awaitable[None]]] = None,
        on_disconnect: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.futures = futures
        self.testnet = testnet
        self.user_id = user_id
        self.api_key = api_key
        self.secret_key = secret_key
        
        self._reconnect_interval = reconnect_interval
        self._max_reconnect_attempts = max_reconnect_attempts
        self._ping_interval = ping_interval
        
        self._on_message = on_message
        self._on_error = on_error
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._reconnect_attempts = 0
        self._connected = False
        self._receive_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._subscribed_streams = set()
        self._message_queue = asyncio.Queue()
        self._last_message_time = 0
        
        self._listen_key = None
        self._user_stream_task: Optional[asyncio.Task] = None
        
        self._set_base_urls()


    async def connect_user_data_stream(self) -> bool:
        """
        🔐 User data stream için listen key al ve bağlan - DÜZELTİLDİ
        """
        try:
            # API key yoksa ama user_id varsa, manager'dan al
            if not self.api_key and self.user_id:
                await self._load_credentials_from_manager()
            
            if not self.api_key or not self.secret_key:
                logger.error("❌ No credentials available for user data stream")
                return False
            
            # Listen key al
            self._listen_key = await self._get_listen_key()
            if not self._listen_key:
                return False
            
            # DÜZELTME: Futures'ta da stream ismi aynı
            stream_name = f"{self._listen_key}@userData"
            await self.subscribe([stream_name])
            
            # Listen key'i periyodik olarak yenile
            self._user_stream_task = asyncio.create_task(self._keepalive_listen_key())
            
            logger.info(f"✅ User data stream connected for user {self.user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ User data stream connection failed: {e}")
            return False
            


    async def _load_credentials_from_manager(self):
        """APIKeyManager'dan credential'ları yükle - DÜZELTİLDİ"""
        if APIKeyManager is None:
            logger.error("APIKeyManager not installed")  # ✅ DÜZELTİLDİ
            return
        try:
            # DÜZELTME: Absolute import
            from utils.apikey_manager import APIKeyManager
            
            api_manager = APIKeyManager.get_instance()
            creds = await api_manager.get_apikey(self.user_id)
            
            if creds:
                self.api_key, self.secret_key = creds
                logger.info(f"✅ Loaded credentials from manager for user {self.user_id}")
            else:
                logger.warning(f"⚠️ No credentials found in manager for user {self.user_id}")
                
        except ImportError as e:
            logger.error(f"❌ apikey_manager not available: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to load credentials: {e}")
            
    #listen     

    async def _cancel_task(self, task: Optional[asyncio.Task], task_name: str = "") -> bool:
        """Task'ı güvenli şekilde iptal et"""
        if task and not task.done():
            logger.debug(f"Cancelling task: {task_name}")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return True
        return False

    def _get_rest_base_url(self) -> str:
        """REST API base URL'ini döndür"""
        if self.futures:
            return "https://testnet.binancefuture.com" if self.testnet else "https://fapi.binance.com"
        else:
            return "https://testnet.binance.vision" if self.testnet else "https://api.binance.com"
            
    
    def _get_listen_key_endpoint(self) -> str:
        """Listen key endpoint'ini döndür"""
        return "/fapi/v1/listenKey" if self.futures else "/api/v3/userDataStream"

    
   
    async def _get_listen_key(self) -> Optional[str]:
        """Binance'ten listen key al - DÜZELTİLDİ"""
        try:
            base_url = self._get_rest_base_url()
            endpoint = self._get_listen_key_endpoint()
            url = base_url + endpoint
            
            async with aiohttp.ClientSession() as session:
                headers = {"X-MBX-APIKEY": self.api_key}
                
                async with session.post(url, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        listen_key = data.get('listenKey')
                        logger.info(f"✅ Listen key obtained: {listen_key[:20]}...")
                        return listen_key
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Failed to get listen key: {response.status} - {error_text}")
                        return None
                        
        except Exception as e:
            logger.error(f"❌ Error getting listen key: {e}")
            return None

    async def _keepalive_listen_key(self):
        """Listen key'i periyodik olarak yenile"""
        while self._connected and self._listen_key:
            try:
                await asyncio.sleep(30 * 60)  # 30 dakikada bir
                
                if not self._connected:
                    break
                    
                await self._renew_listen_key()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Listen key keepalive error: {e}")

   
    async def _renew_listen_key(self):
        """Listen key'i yenile - DÜZELTİLDİ"""
        try:
            base_url = self._get_rest_base_url()
            endpoint = self._get_listen_key_endpoint()
            url = base_url + endpoint
            
            params = {"listenKey": self._listen_key}
            
            async with aiohttp.ClientSession() as session:
                headers = {"X-MBX-APIKEY": self.api_key}
                
                async with session.put(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        logger.debug("✅ Listen key renewed")
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Failed to renew listen key: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"❌ Error renewing listen key: {e}")
    

    async def _close_listen_key(self):
        """Listen key'i kapat - DÜZELTİLDİ"""
        if not self._listen_key:
            return
            
        try:
            base_url = self._get_rest_base_url()
            endpoint = self._get_listen_key_endpoint()
            url = base_url + endpoint
            
            params = {"listenKey": self._listen_key}
            
            async with aiohttp.ClientSession() as session:
                headers = {"X-MBX-APIKEY": self.api_key}
                
                async with session.delete(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        logger.info("✅ Listen key closed")
                    else:
                        error_text = await response.text()
                        logger.warning(f"⚠️ Failed to close listen key: {response.status} - {error_text}")
                        
        except Exception as e:
            logger.error(f"❌ Error closing listen key: {e}")
        finally:
            self._listen_key = None



    def _set_base_urls(self):
        """URL'leri constants'tan al"""
        if self.futures:
            self.BASE_WS_URL = WS_TESTNET_FUTURES_URL if self.testnet else WS_FUTURES_URL
            self.COMBINED_STREAM_URL = WS_TESTNET_FUTURES_URL if self.testnet else WS_FUTURES_URL
        else:
            self.BASE_WS_URL = WS_TESTNET_URL if self.testnet else WS_BASE_URL
            self.COMBINED_STREAM_URL = WS_TESTNET_URL if self.testnet else WS_BASE_URL

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
            logger.debug("Created new aiohttp ClientSession for WebSocket")
        return self._session

    async def connect(self):
        """
        Connect to Binance WebSocket endpoint.
        Eski method signature korundu, içeriği geliştirildi.
        """
        try:
            # Yeni: Combined stream URL desteği
            base_url = self.BASE_WS_URL
            url = base_url
            
            if self._subscribed_streams:
                streams_param = '/'.join(self._subscribed_streams)
                combined_url = self.COMBINED_STREAM_URL
                url = f"{combined_url}?streams={streams_param}"

            session = await self._get_session()
            
            # Yeni: heartbeat ve autoping eklendi
            self._ws = await session.ws_connect(
                url,
                heartbeat=self._ping_interval,
                autoping=True
            )
            
            self._connected = True
            self._reconnect_attempts = 0  # Yeni: reconnect sayacı sıfırlandı
            self._stop_event.clear()
            
            # Yeni: Ping loop eklendi
            self._receive_task = asyncio.create_task(self._receive_loop())
            self._ping_task = asyncio.create_task(self._ping_loop())
            
            logger.info(f"✅ Connected to Binance WebSocket at {url}")
            
            # Yeni: on_connect callback
            if self._on_connect:
                await self._on_connect()
                
            return True
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed: {e}")
            # Yeni: on_error callback
            if self._on_error:
                await self._on_error(e)
            return False

          
    async def _ping_loop(self):
        """Yeni: Send periodic ping messages to keep connection alive."""
        while self._connected and not self._stop_event.is_set():
            try:
                await asyncio.sleep(self._ping_interval)
                if self._ws and not self._ws.closed:
                    await self._ws.ping()
                    logger.debug("Sent WebSocket ping")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Ping error: {e}")
                break




    async def disconnect(self):
            """WebSocket bağlantısını kapat - İYİLEŞTİRİLDİ"""
            self._connected = False
            self._stop_event.set()
            
            # Task'ları merkezi method ile iptal et
            await self._cancel_task(self._user_stream_task, "user_stream_task")
            await self._cancel_task(self._ping_task, "ping_task")
            await self._cancel_task(self._receive_task, "receive_task")
            
            self._user_stream_task = None
            self._ping_task = None
            self._receive_task = None
            
            # Listen key'i kapat
            if self._listen_key:
                await self._close_listen_key()
            
            # WebSocket'i kapat
            if self._ws and not self._ws.closed:
                await self._ws.close()
                logger.info("WebSocket connection closed")

            # Session'ı kapat
            if self._session and not self._session.closed:
                await self._session.close()
                logger.info("aiohttp session closed")
                
            # on_disconnect callback
            if self._on_disconnect:
                await self._on_disconnect()
                

    async def _receive_loop(self):
        """DÜZELTİLDİ: Tek bir _receive_loop methodu"""
        while not self._stop_event.is_set():
            try:
                async for msg in self._ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        self._last_message_time = time.time()
                        
                        # Metrics kaydı - DÜZELTİLDİ
                        try:
                            from .binance_metrics import record_response
                            await record_response(
                                endpoint="websocket",
                                status_code=200,
                                response_body=data,
                                start_time=self._last_message_time
                            )
                        except ImportError:
                            logger.debug("Metrics module not available")  # DÜZELTİLDİ
                        
                        if self._on_message:
                            await self._on_message(data)
                        else:
                            await self._message_queue.put(data)
                            
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        logger.warning("WebSocket closed by server")
                        await self._record_websocket_error("closed_by_server")
                        break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"WebSocket error: {msg.data}")
                        await self._record_websocket_error("websocket_error", str(msg.data))
                        break
                    elif msg.type == aiohttp.WSMsgType.PONG:
                        logger.debug("Received WebSocket pong")
                        
                if not self._stop_event.is_set():
                    await self._reconnect()
                    
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                await self._record_websocket_error("receive_loop_error", str(e))
                
                if self._on_error:
                    await self._on_error(e)
                    
                if not self._stop_event.is_set():
                    await self._reconnect()
             

    async def _record_websocket_error(self, error_type: str, message: str = ""):
        """ WebSocket hatalarını metrics'a kaydet"""
        try:
            from .binance_metrics import record_response
            await record_response(
                endpoint="websocket",
                status_code=None,
                response_body=None,
                error=BinanceWebSocketError(f"{error_type}: {message}")
            )
        except ImportError:
            logger.debug("Metrics module not available")  # Metrics modülü yoksa sessizce devam et

    async def _reconnect(self):
        """
        Yeni: Exponential backoff ile akıllı yeniden bağlanma
        Eski basit reconnect yerine gelişmiş versiyon
        """
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            if self._on_error:
                await self._on_error(ConnectionError("Max reconnection attempts reached"))
            return

        self._reconnect_attempts += 1
        # Yeni: Exponential backoff
        delay = min(self._reconnect_interval * (2 ** (self._reconnect_attempts - 1)), 300)
        
        logger.warning(f"🔄 Reconnecting in {delay}s (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts})")
        await asyncio.sleep(delay)
        
        try:
            await self.connect()
            # Yeni: Önceki stream'leri yeniden subscribe et
            if self._subscribed_streams:
                await self.subscribe(list(self._subscribed_streams))
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")

    async def send(self, message: dict):
        """
        Send a JSON message to WebSocket.
        Eski method signature korundu, hata yönetimi geliştirildi.
        """
        if not self._connected or self._ws is None:
            raise ConnectionError("WebSocket is not connected")
        
        try:
            await self._ws.send_json(message)
            logger.debug(f"Sent WS message: {message}")
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            # Yeni: on_error callback
            if self._on_error:
                await self._on_error(e)
            raise

    def validate_stream_name(self, stream: str) -> bool:
        """
        Validate if stream name is in supported streams.
        YENİ: Stream ismi doğrulama
        """
        stream_parts = stream.split('@')
        if len(stream_parts) != 2:
            return False
        
        stream_type = stream_parts[1].lower()
        return stream_type in WS_STREAMS_REVERSE

    async def subscribe(self, streams: list[str]):
        """
        Subscribe to one or more streams.
        GÜNCELLENDİ: Stream validation eklendi
        """
        streams = [s.lower() for s in streams]
        
        # YENİ: Stream validation
        invalid_streams = [s for s in streams if not self.validate_stream_name(s)]
        if invalid_streams:
            logger.warning(f"Invalid stream names: {invalid_streams}")
            # Hata callback'i çağır
            if self._on_error:
                await self._on_error(ValueError(f"Invalid stream names: {invalid_streams}"))
        
        valid_streams = [s for s in streams if self.validate_stream_name(s)]
        self._subscribed_streams.update(valid_streams)
        
        if not valid_streams:
            logger.warning("No valid streams to subscribe")
            return
        
        msg = {
            "method": "SUBSCRIBE",
            "params": valid_streams,
            "id": int(time.time() * 1000)
        }
        
        await self.send(msg)
        logger.info(f"✅ Subscribed to streams: {valid_streams}")

    async def unsubscribe(self, streams: list[str]):
        """
        Unsubscribe from streams.
        Eski method signature korundu.
        """
        streams = [s.lower() for s in streams]
        for s in streams:
            self._subscribed_streams.discard(s)
            
        msg = {
            "method": "UNSUBSCRIBE",
            "params": streams,
            "id": int(time.time() * 1000)  # Yeni: unique ID
        }
        
        await self.send(msg)
        logger.info(f"Unsubscribed from streams: {streams}")

    # Backward compatibility uygun
    async def list_subscriptions(self) -> List[str]:
        """Yeni: Get list of currently subscribed streams."""
        return list(self._subscribed_streams)

    def is_connected(self) -> bool:
        """Yeni: Check if WebSocket is connected."""
        return self._connected and self._ws and not self._ws.closed

    def get_last_message_time(self) -> float:
        """Yeni: Get timestamp of last received message."""
        return self._last_message_time

    def get_reconnect_attempts(self) -> int:
        """Yeni: Get current reconnect attempt count."""
        return self._reconnect_attempts

    # Eski async iterator desteği korundu
    def __aiter__(self):
        """
        Async iterator over incoming messages (if no callback provided).
        Eski method aynen korundu.
        """
        if self._on_message is not None:
            raise RuntimeError("Cannot use iterator and callback simultaneously")
        return self._message_generator()

    async def _message_generator(self):
        """
        Yalnızca __aiter__ için mesajları queue'ya koyup yield eder.
        Eski method geliştirildi - queue kullanılıyor.
        """
        while self._connected or not self._message_queue.empty():
            try:
                # Yeni: timeout ile daha güvenli
                message = await asyncio.wait_for(
                    self._message_queue.get(), 
                    timeout=1.0
                )
                yield message
                self._message_queue.task_done()
            except asyncio.TimeoutError:
                if not self._connected:
                    break
                continue

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        

    async def get_connection_health(self) -> Dict[str, Any]:
        """
        🔧 DÜZELTİLDİ: WebSocket bağlantı sağlık durumunu döndür
        """
        current_time = time.time()
        time_since_last_msg = current_time - self._last_message_time if self._last_message_time > 0 else float('inf')
        
        base_health = {
            "connected": self.is_connected(),
            "reconnect_attempts": self._reconnect_attempts,
            "subscribed_streams_count": len(self._subscribed_streams),
            "time_since_last_message": time_since_last_msg,
            "message_queue_size": self._message_queue.qsize(),
            "last_message_time": self._last_message_time,
            "ping_interval": self._ping_interval,
            # YENİ: User stream durumu
            "user_data_stream_connected": self._listen_key is not None,
            "has_credentials": bool(self.api_key and self.secret_key),
            "user_id": self.user_id,
            "listen_key": self._listen_key[:20] + "..." if self._listen_key else None
        }
        return base_health

# WSManager eklendi - User Bazlı islemler - eski ad: MultiUserWebSocketManager
class BinanceWSManager:
    def __init__(self):
        self._connections: Dict[int, BinanceWSClient] = {}  # user_id -> client mapping
        self._user_streams: Dict[int, BinanceWSClient] = {}  # user-specific streams
    
    async def create_user_connection(
        self,
        user_id: int,
        streams: List[str],
        futures: bool = False,
        on_message: Optional[Callable[[dict], Awaitable[None]]] = None,
        connect_user_stream: bool = True,  # YENİ: User data stream seçeneği
        **kwargs
    ) -> BinanceWSClient:
        """Create WebSocket connection for specific user"""
        # Mevcut bağlantıyı kontrol et
        if user_id in self._connections:
            existing_ws = self._connections[user_id]
            if existing_ws.is_connected():
                logger.info(f"🔄 Using existing connection for user {user_id}")
                return existing_ws
            else:
                # Bağlantı kopmuş, temizle
                await self._cleanup_user_connection(user_id)
        
        # Yeni bağlantı oluştur
        ws = BinanceWSClient(
            user_id=user_id,
            futures=futures, 
            on_message=on_message,
            **kwargs
        )
        
        await ws.connect()
        
        # Public streams
        if streams:
            await ws.subscribe(streams)
            
        # User data stream (opsiyonel)
        if connect_user_stream:
            user_stream_connected = await ws.connect_user_data_stream()
            if user_stream_connected:
                self._user_streams[user_id] = ws
                logger.info(f"✅ User data stream activated for user {user_id}")
        
        self._connections[user_id] = ws
        return ws
    
    async def get_user_connection(self, user_id: int) -> Optional[BinanceWSClient]:
        """Get WebSocket connection for user"""
        return self._connections.get(user_id)
    
    async def close_user_connection(self, user_id: int):
        """Close specific user's WebSocket connection"""
        if user_id in self._connections:
            ws = self._connections[user_id]
            if ws.is_connected():
                await ws.disconnect()
            del self._connections[user_id]
            
        if user_id in self._user_streams:
            del self._user_streams[user_id]
    
    async def _cleanup_user_connection(self, user_id: int):
        """Cleanup disconnected user connections"""
        if user_id in self._connections:
            ws = self._connections[user_id]
            if not ws.is_connected():
                del self._connections[user_id]
                if user_id in self._user_streams:
                    del self._user_streams[user_id]
    
    async def close_all(self):
        """Close all WebSocket connections"""
        for user_id, ws in list(self._connections.items()):
            if ws.is_connected():
                await ws.disconnect()
        self._connections.clear()
        self._user_streams.clear()


"""
v9 - Enhanced Binance WebSocket Client

✅ ÖZELLİKLER:
- Combined stream support  
- Exponential backoff reconnection
- Ping/pong mechanism
- Better error handling and callbacks
- Message queue for async iteration
- Backward compatible with existing code
- Multi-user support with BinanceWSManager
- APIKeyManager integration for secure credentials
- User data streams for private data
- Comprehensive health monitoring

✅ KULLANIM:
# Single connection
async with BinanceWSClient() as ws:
    await ws.subscribe(['btcusdt@ticker'])
    async for msg in ws:
        print(msg)

# Multi-user management  
manager = BinanceWSManager()
user_ws = await manager.create_user_connection(
    user_id=123,
    streams=['btcusdt@ticker'],
    connect_user_stream=True
)
"""

# TEST NOT: Çoklu kullanıcı testleri için tests/ dizininde 
# test_binance_websocket.py dosyasını oluşturun.