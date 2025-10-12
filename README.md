utils/binance_api/binance_a.py
maps tabanlıdır

---
.yaml çağrı şekli 	GET
örnek:
çağrılacak metot: ticker_24h

get_ticker_24h:
    client: SpotClient
    method: get_ticker_24h
    path: /api/v3/ticker/24hr
    http_method: GET
    signed: false
    scope: market
    base: spot
    weight: 1
    rate_limit_type: IP
    multi_user_support: true
    cache_ttl: 5
    job_type: io
    purpose: "Retrieve 24-hour price change statistics for trend strength metrics"
    tags: [public, ticker, volatility, analysis]
    enabled: true
    version: v3
    return_type: dict
	
---
