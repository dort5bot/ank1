# analysis/analysis_router.py

"""
 Hybrid API Management
API Key Kullanım Senaryoları:Senaryo	API Key Tipi	Kullanım Amacı	Limit Etkisi
1. Sistem/Bot API
(.env'deki)	Binance API+Secret	Tüm kullanıcılar için ortak analiz	Tek IP → Tüm kullanıcılar için ortak limit! ⚠️
2. Kişisel API
(api_manager.py)	Kullanıcıya özel	Trade, bakiye, kişisel işlemler	Kullanıcı başına ayrı limit ✅

 
"""
# analysis/analysis_router.py - GÜNCELLENMİŞ

from fastapi import APIRouter, Query, HTTPException, Security, Depends, Header
from typing import Optional, Dict
import os
from dotenv import load_dotenv

# ✅ YENİ: APIKeyManager import
from utils.apikey_manager import APIKeyManager, initialize_managers

load_dotenv()
router = APIRouter()

# ==========================================================
# 🔐 HYBRID API KEY YÖNETİMİ - APIKeyManager ENTEGRASYONU
# ==========================================================

# Sistem API'si (fallback için - .env'den)
SYSTEM_BINANCE_API = {
    "api_key": os.getenv("BINANCE_API_KEY"),
    "api_secret": os.getenv("BINANCE_API_SECRET"),
    "source": "system"
}

async def get_user_credentials(
    x_user_id: Optional[int] = Header(None, description="Kullanıcı ID"),
    x_api_key: Optional[str] = Header(None, description="Binance API Key (opsiyonel)"),
    x_api_secret: Optional[str] = Header(None, description="Binance API Secret (opsiyonel)")
) -> Dict[str, str]:
    """
    🔄 HYBRID CREDENTIAL MANAGEMENT:
    1. Önce header'dan gelen API key/secret
    2. Sonra APIKeyManager'dan kullanıcının kayıtlı API'si  
    3. En son sistem API'si (fallback)
    """
    
    # ✅ 1. HEADER'DAN GELEN API (en yüksek öncelik)
    if x_api_key and x_api_secret:
        return {
            "api_key": x_api_key,
            "api_secret": x_api_secret, 
            "source": "header",
            "user_id": x_user_id
        }
    
    # ✅ 2. APIKeyManager'DAN KULLANICI API'Sİ
    if x_user_id:
        try:
            api_manager = APIKeyManager.get_instance()
            user_creds = await api_manager.get_apikey(x_user_id)
            
            if user_creds:
                api_key, secret_key = user_creds
                return {
                    "api_key": api_key,
                    "api_secret": secret_key,
                    "source": "apikey_manager", 
                    "user_id": x_user_id
                }
        except Exception as e:
            print(f"⚠️ APIKeyManager error for user {x_user_id}: {e}")
            # Fallback to system API
    
    # ✅ 3. SİSTEM API'Sİ (FALLBACK - RATE LİMİT RİSKLİ!)
    if SYSTEM_BINANCE_API["api_key"]:
        return {
            **SYSTEM_BINANCE_API,
            "user_id": x_user_id,
            "warning": "Using shared system API - rate limits may apply"
        }
    
    # ❌ Hiçbir API bulunamadı
    raise HTTPException(
        status_code=400, 
        detail="""
        No Binance API credentials found. Provide either:
        - x-user-id + registered API keys in APIKeyManager OR  
        - x-api-key + x-api-secret headers OR
        - System API keys in .env file
        """
    )


# analysis_router.py'ye ek güvenlik katmanları

async def validate_and_get_credentials(
    user_id: Optional[int] = None,
    api_key: Optional[str] = None, 
    api_secret: Optional[str] = None
) -> Dict[str, str]:
    """
    Gelişmiş credential validation with APIKeyManager
    """
    api_manager = APIKeyManager.get_instance()
    
    # ✅ API Key validation with Binance
    if user_id:
        is_valid = await api_manager.validate_binance_credentials(user_id)
        if is_valid:
            creds = await api_manager.get_apikey(user_id)
            return {
                "api_key": creds[0],
                "api_secret": creds[1], 
                "source": "apikey_manager_validated",
                "user_id": user_id
            }
    
    # Fallback to other methods...
    return await get_user_credentials(user_id, api_key, api_secret)


# ==========================================================
# 📦 MODÜL YÜKLEYİCİ - CREDENTIAL AWARE
# ==========================================================

def create_endpoint(run_func):
    async def endpoint(
        symbol: str,
        priority: Optional[str] = Query(None),
        credentials: Dict = Depends(get_user_credentials)  # ✅ Credentials injection
    ):
        try:
            # Input validation
            if not symbol or not symbol.strip():
                raise HTTPException(status_code=400, detail="Symbol parameter is required")
            
            symbol = symbol.strip().upper()
            
            # 🎯 CREDENTIAL-AWARE ANALİZ
            result = await run_func(
                symbol=symbol, 
                priority=priority,
                credentials=credentials  # ✅ API bilgilerini modüle iletiyoruz
            )
            
            # Result validation
            if not isinstance(result, dict):
                raise HTTPException(status_code=500, detail="Invalid response format from analysis module")
            
            # ✅ API source bilgisini response'a ekle (debug için)
            result["api_source"] = credentials.get("source", "unknown")
            result["user_id"] = credentials.get("user_id")
                
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Analysis error for {symbol}: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Analysis failed: {str(e)}"
            )
    return endpoint

# ==========================================================
# 🔁 DİNAMİK ROUTE OLUŞTURMA
# ==========================================================

# ✅ APIKeyManager initialization
@router.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında APIKeyManager'ı başlat"""
    success = await initialize_managers()
    if not success:
        print("❌ APIKeyManager initialization failed!")
    else:
        print("✅ APIKeyManager initialized successfully")

schema = load_analysis_schema()

for module in schema.modules:
    try:
        route_path = module.command
        module_file = module.file
        run_function = load_module_run_function(module_file)
        endpoint = create_endpoint(run_function)

        router.add_api_route(
            path=route_path,
            endpoint=endpoint,
            methods=["GET"],
            summary=module.name,
            description=f"{module.objective or ''} (API: {module.api_type})",
            tags=["analysis"],
            # ✅ Response model eklenebilir
            # response_model=AnalysisResponse
        )
        
        print(f"✅ Route created: {route_path} -> {module.file}")
        
    except Exception as e:
        print(f"❌ Failed to create route for {module.name}: {str(e)}")