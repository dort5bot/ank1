
spotclient.py → private_spot + public_spot
futuresclient.py → private_futures + public_futures
4 farklı YAML bölümü için 2 farklı client dosyası gerekiyor. 
Private/public ayrımı sadece endpoint türünü belirtiyor, client tipini değil.

====**====
* .env de tanımlı api ile işlemleri yapar
* /apikey ile api + secret girilmiş ise kişisel işlemleri bu api ile yapar
(alarm, trade, ...)
alarm, trade için TG_İD KULLANILSA NE OLUR

====**====
dosyayı incele
dosya sadece publick api ile veri alacak endpointleri içerecek şekilde tasarladı
- gelişimi korunarak, geliştirilecek, mevcut adlandırmaları kornacak

- öneri ve geliştirme varsa belirt
- eksik endpoint var mı
- import eksiklikleri/ sorunları var mı

Kod tekrarının yok + Sıralama sorunu yok + çakışmayan + açılış/ kapanışları tam
Hafıza sızıntısı olmayan +  beklenmedik davranışlar veya performans sorunları olmayan
geliştirici dostu Kod yorumları oldukça açıklamaları

DIContainer ile global nesne erişimi çözülmüş yapı


* BOT GENEL ÖZELLİKLERİ
- multi user destekli
import/lock güvenliği, 
aiogram 3.x e uygun + Router pattern + logging yapı
Type hint'leri iyileştir - Any kullanımı en az sayıda
imports - ABSOLUTE  

Code Quality:PEP8 compliance, Type hints, Docstrings, Singleton pattern, modern Async/await pattern
Error Handling: Comprehensive try-catch blocks, Binance API validation, 
Security Improvements: API key format validation, Input sanitization, 
Thread Safety: asyncio.Lock eklendi, Connection pool management, Cache synchronization, Key masking for logging
Performance:  Connection pooling, Cache management, Index optimization

** gerekiyorsa - botta bunu gerektiren dosyalar var
Database transaction rollback
cache cleanup,
Periodic cleanup tasks, 
Secure message deletion
==============================
binance_a.py
from .b_map_validator import BMapValidator, MapValidationError

======ANALİZ=====
veri kaynağı:

utils/
└── binance_api/
    ├── __init__.py
    ├── binance_a.py              # 🧠 map tabanlı Ana aggregator (public/private sınıflar dahil)
    ├── b-map_private.yaml        #←  Private API endpoint’leri
    ├── b-map_public.yaml         #←  Public API endpoint’leri
    ├── b_map_validator.py        #←  YAML doğrulama (BMapValidator)
    ├── futuresclient.py
    ├── spotclient.py
    │
    ├── binance_multi_user.py        # Çoklu kullanıcı (Multi API key yönetimi)
    │  
    ├── binance_client.py            # HTTP + Auth yönetimi
    ├── binance_circuit_breaker.py   # Rate limit & error kontrol
    ├── binance_request.py           # Asenkron istek yöneticisi (aiohttp tabanlı)
    ├── binance_constants.py         # Sabitler ve base URL'ler
    ├── binance_exceptions.py        # Hata tipleri (ör. BinanceAPIError)
    ├── binance_types.py             # Type hint'ler (TypedDict’ler)
    ├── binance_metrics.py           # Opsiyonel: API performans ölçüm / latency log
    ├── binance_websocket.py         # Stream API (gerçek zamanlı)
    ├── binance_ws_pydantic.py       # WebSocket için model doğrulama (opsiyonel)
    └── README.md

│   │   ├── futuresclient.py
│   │   ├── rate_limiter.py
│   │   └── spotclient.py

HEDEF: analiz modülleri oluşturmak

MODÜL AYRINTILARI:
   Modül adı -  .py dosya adı -  Endpoint(ler) -  API Türü -  eklenecek Zorunlu Klasik Metrikler -  eklenecek Profesyonel Metrikler (öncelik: * yüksek / ** orta / *** düşük) -  Amaç -  Çıktı Türü -  Komut - Metot -  İş Tipi -  Paralel işlem Türü -  Gerekçe / Katkı
A. Trend & Momentum (TA) -  trend_moment.py -  /api/v3/klines, /api/v3/ticker/24hr, /api/v3/avgPrice ➕ /fapi/v1/continuousKlines (futures trend uyumu için) -  Spot Public -  EMA, RSI, MACD, Bollinger Bands, ATR + ADX (Directional Index), Stochastic RSI, Momentum Oscillator -  *Kalman Filter Trend, *Z-Score Normalization, **Wavelet Transform, **Hilbert Transform Slope, **Fractal Dimension Index (FDI) -  Fiyat yönü & momentum gücü -  Trend Score (0–1) -  /trend, /t - GET -  CPU-bound -  Batch -  Spot & futures trend uyuşması, daha sağlam momentum skoru
B. Piyasa Rejimi (Volatilite & Yapı) -  volat_regime.py -  /api/v3/klines, /fapi/v1/markPrice, /fapi/v1/fundingRate ➕ /fapi/v1/premiumIndex, /fapi/v1/indexPriceKlines -  Spot + Futures Public -  Historical Volatility, ATR, Bollinger Width + Variance Ratio Test, Range Expansion Index -  *GARCH(1,1), *Entropy Index, **Hurst Exponent, ***Regime Switching Model -  Trend / Range modu ayrımı -  Regime Label (Trend, Range) -  /regime, /rg - GET -  CPU-bound -  Batch -  Volatilite rejimi için implied–realized farkı ve premium yapısı gerek
...
   
   
1.yöntem
├── analysis
│   ├── __init__.py
│   ├── analysis_base_module.py      # (Base class + legacy decorator + OHLCV fetch)
│   ├── analysis_core.py             # (Aggregator, run_all/run_single_analysis, circuit breaker, cleanup, health check, user & performance optimize sınıfları)
│   ├── analysis_metric_schema.yaml  # endpoint, komut, metrik, çıktı tipi, amaç, vb. tanımlarını içerir.
│   ├── analysis_router.py           # FastAPI router
│   ├── analysis_schema_manager.py   # Schema yöneticisi
│   ├── corr_lead.py
│   ├── deriv_sentim.py
│   ├── microalpha.py
│   ├── onchain.py
│   ├── order_micros.py
│   ├── port_alloc.py
│   ├── regime_anomal.py
│   ├── risk_expos.py
│   ├── trend_moment.py
│   └── volat_regime.py
│   │  
│   ├── config
│   │   ├── c_corr.py
│   │   ├── c_deriv.py
│   │   ├── c_micro.py
│   │   ├── c_onchain.py
│   │   ├── c_order.py
│   │   ├── c_portalloc.py
│   │   ├── c_risk.py
│   │   ├── c_trend.py
│   │   ├── c_volat.py
│   │   ├── cm_base.py # modül base
│   │   └── cm_loader.py # modül config loader





├── analysis
│   ├── config
│   │   ├── c_deriv.py
│   │   ├── c_micro.py
│   │   ├── c_onchain.py
│   │   ├── c_order.py
│   │   ├── c_portalloc.py
│   │   ├── c_risk.py
│   │   ├── c_trend.py
│   │   ├── c_volat.py
│   │   ├── cm_base.py  # modül base.py
│   │   └── cm_loader.py    # modül config loader > loader.py
│   ├── __init__.py
│   ├── analysis_base_module.py
│   ├── analysis_core.py    
│   ├── analysis_metric_schema.yaml
│   ├── analysis_router.py
│   ├── analysis_schema_manager.py
│   ├── deriv_sentim.py
│   ├── microalpha.py
│   ├── onchain.py
│   ├── order_micros.py
│   ├── regime_anomal.py
│   ├── risk_expos.py
│   ├── trend_moment.py
│   └── volat_regime.py

örnek:
    trend_moment.py
    - içerik sadece modül ayrıntılarındaki metrikler olur
    - analysis_metric_schema.yaml içeriisnde metrikler için gerekli bilgiler yer alır
    örnek:
       
      - name: "Trend & Momentum"
        file: "trend_moment.py"
        command: "/trend"
        api_type: "Spot Public"
        endpoints: ["/api/v3/klines", "/api/v3/ticker/24hr", "/api/v3/avgPrice"]
        methods: ["GET"]

        classical_metrics:
          - EMA
          - RSI
          - MACD
          - Bollinger Bands
          - ATR

        professional_metrics:
          - { name: "Kalman Filter Trend", priority: "*" }
          - { name: "Z-Score Normalization", priority: "*" }
          - { name: "Wavelet Transform", priority: "**" }
          - { name: "Hilbert Transform Slope", priority: "**" }

        composite_metrics:
          - "0.25*MACD_Histogram + 0.20*RSI + 0.20*EMA_21 + 0.15*EMA_9 + 0.10*EMA_50 + 0.10*Kalman_Trend"

        objective: "Fiyat yönü & momentum gücü"
        output_type: "Trend Score (0–1)"
        
    - modül metrikleri paralel şekilde çalıştırır, en hızlı şekilde sonuçları üretir
    - analysis_core.py    agredator ile sonuçlar çağrılıp kullanılır, yeni skorlar oluşturulur
    - config gerekiyorsa kendi sayfası içinde olur, kısmende mümkünse analysis_metric_schema.yaml içine eklenir
    - modüle ait dosyalar çok parçalanmadan ama mevcut dosyada şişmeden 800-1000 satır olmadan işlemini yapar
    
2. yöntem
- modüle ait .py dosyası olur,
- gerekli endpoitler vb burda olur
- gerekli işlemleri üretir
 - analysis_core.py    agredator ile sonuçlar çağrılıp kullanılır, yeni skorlar oluşturulur




Örnek kısa zincir (tam iş akışı)
user -> /trend komutu -> tremo.py -> SpotClient.get_symbol_price()
→ SpotClient -> http_client.send_request("GET", "/api/v3/ticker/price")
→ BinanceHTTPClient._request() -> aiohttp session + retry + metrics
→ Binance API
→ response return
