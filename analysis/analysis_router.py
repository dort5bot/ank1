# analysis/analysis_router.py

"""
analiz modüllerini otomatik olarak yükleyen ve route (endpoint) oluşturur
Uygulama: main.py
# main.py
from fastapi import FastAPI
from api.analysis_router import router as analysis_router
app = FastAPI(title="Modüler Analiz API")
app.include_router(analysis_router)
# Swagger: http://localhost:8000/docs


| Özellik                     | Açıklama                                                       |
| --------------------------- | -------------------------------------------------------------- |
| ⚡ Otomatik endpoint üretimi | Yeni analiz modülü = otomatik yeni route                       |
| 🔗 Swagger desteği          | Her modül dokümantasyonda listelenir                           |
| ♻️ Kolay bakım              | Yeni route eklemek için sadece `.yaml` + `.py` dosyası yeterli |
| 🧩 Modüler mimari           | Her modül kendi dosyasında, bağımsız `run()` ile               |


"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional
from .analysis_schema_manager import load_analysis_schema, load_module_run_function


router = APIRouter()
schema = load_analysis_schema()

# Tüm modülleri yükleyip route oluştur
for module in schema.modules:
    route_path = module.command  # örn. "/trend"
    module_file = module.file    # örn. "trend_moment.py"
    run_function = load_module_run_function(module_file)

    async def endpoint(symbol: str, priority: Optional[str] = Query(None), _run=run_function):
        try:
            result = await _run(symbol=symbol, priority=priority)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Route'u FastAPI'ye ekle
    router.add_api_route(
        path=route_path,
        endpoint=endpoint,
        methods=["GET"],
        summary=module.name,
        description=f"{module.objective or ''} (API type: {module.api_type})",
        tags=["analysis"]
    )
