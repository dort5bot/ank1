# utils/binance_api/bi_advanced_validator.py
"""
# Gelişmiş versiyon
python utils/binance_api/bit_advanced_validator.py


✅ Otomatik YAML parsing - Manuel liste gerekmez
✅ Smart parameter selection - Endpoint tipine göre parametre
✅ Async testing - Hızlı sonuç
✅ Detailed reporting - Gruplandırılmış sonuçlar
✅ Error handling - Timeout ve hata yönetimi
✅ Section grouping - Public/private ayrımı

public + 
"""
# utils/binance_api/bi_advanced_validator.py
import requests
import yaml
import asyncio
import aiohttp
from typing import Dict, List, Any
import os
import json
from pathlib import Path

class BinanceEndpointValidator:
    def __init__(self):
        self.base_urls = {
            "spot": "https://api.binance.com",
            "futures": "https://fapi.binance.com"
        }
        self.results = []
        self.corrections_needed = []
        
        # Doğru path mapping'i
        self.correct_paths = {
            # Futures data endpoints
            "/fapi/v1/globalLongShortAccountRatio": "/futures/data/globalLongShortAccountRatio",
            "/fapi/v1/topLongShortAccountRatio": "/futures/data/topLongShortAccountRatio", 
            "/fapi/v1/topLongShortPositionRatio": "/futures/data/topLongShortPositionRatio",
            "/fapi/v1/takerlongshortRatio": "/futures/data/takerlongshortRatio",
            "/fapi/v1/openInterestHist": "/futures/data/openInterestHist",
            "/fapi/v1/takerBuySellVol": "/futures/data/takerBuySellVol",
            
            # Diğer düzeltmeler
            "/fapi/v1/openInterest": "/fapi/v1/openInterest",  # Bu doğru, parametre sorunu
            "/fapi/v1/allForceOrders": "/fapi/v1/allForceOrders",  # Bu doğru, parametre sorunu
            "/fapi/v1/adlQuantile": "/fapi/v1/adlQuantile",  # Bu doğru, parametre sorunu
            "/fapi/v1/historicalTrades": "/fapi/v1/historicalTrades"  # Bu doğru, parametre sorunu
        }
    
    def load_yaml_endpoints(self, file_path: str) -> Dict[str, Any]:
        """YAML dosyasından endpoint'leri yükle"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def extract_testable_endpoints(self, yaml_data: Dict[str, Any], file_path: str) -> List[Dict]:
        """YAML'den test edilebilir endpoint'leri çıkar"""
        endpoints = []
        
        for section_name, section_data in yaml_data.items():
            if section_name == "meta":
                continue
                
            if isinstance(section_data, dict):
                for endpoint_name, endpoint_info in section_data.items():
                    # Sadece public ve GET endpoint'leri test et
                    if (endpoint_info.get('signed', True) == False and 
                        endpoint_info.get('http_method') == 'GET' and
                        endpoint_info.get('enabled', True)):
                        
                        endpoints.append({
                            'name': endpoint_name,
                            'path': endpoint_info['path'],
                            'base': endpoint_info.get('base', 'spot'),
                            'section': section_name,
                            'file': os.path.basename(file_path),
                            'full_info': endpoint_info  # Tam bilgiyi sakla
                        })
        
        return endpoints
    

    async def test_endpoint_async(self, endpoint: Dict, session: aiohttp.ClientSession):
        """Async endpoint testi"""
        base_url = self.base_urls[endpoint['base']]
        current_path = endpoint['path']
        corrected_path = self.correct_paths.get(current_path, current_path)
        needs_correction = current_path != corrected_path
        
        url = f"{base_url}{current_path}"
        params = self._get_test_params(current_path)
        
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                response_text = await response.text()  # ✅ Response text'ini al
                status = response.status
                
                # ✅ YENİ: KALDIRILMIŞ ENDPOINT KONTROLÜ
                if status == 400 and "out of maintenance" in response_text:
                    return {
                        'endpoint': endpoint['name'],
                        'current_path': current_path,
                        'corrected_path': corrected_path,
                        'status': '❌ KALDIRILMIŞ',
                        'status_code': status,
                        'base': endpoint['base'],
                        'section': endpoint['section'],
                        'file': endpoint['file'],
                        'needs_correction': needs_correction,
                        'note': 'Binance bu endpointi kaldırdı'
                    }
                
                if status == 200:
                    result = {
                        'endpoint': endpoint['name'],
                        'current_path': current_path,
                        'corrected_path': corrected_path,
                        'status': '✅ BAŞARILI',
                        'status_code': status,
                        'base': endpoint['base'],
                        'section': endpoint['section'],
                        'file': endpoint['file'],
                        'needs_correction': needs_correction,
                        'test_url': f"{base_url}{current_path}",
                        'params_used': params
                    }
                    
                    if needs_correction:
                        self.corrections_needed.append(result)
                    
                    return result
                else:

                    # Hata durumunda düzeltilmiş path'i test et
                    if needs_correction:
                        corrected_url = f"{base_url}{corrected_path}"
                        try:
                            async with session.get(corrected_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as corrected_response:
                                if corrected_response.status == 200:
                                    result = {
                                        'endpoint': endpoint['name'],
                                        'current_path': current_path,
                                        'corrected_path': corrected_path,
                                        'status': '🔄 DÜZELTİLEBİLİR',
                                        'status_code': corrected_response.status,
                                        'base': endpoint['base'],
                                        'section': endpoint['section'],
                                        'file': endpoint['file'],
                                        'needs_correction': True,
                                        'test_url': corrected_url,
                                        'params_used': params,
                                        'note': 'DÜZELTİLMİŞ PATH ÇALIŞIYOR'
                                    }
                                    self.corrections_needed.append(result)
                                    return result
                        except:
                            pass
                    
                    return {
                        'endpoint': endpoint['name'],
                        'current_path': current_path,
                        'corrected_path': corrected_path,
                        'status': '❌ HATALI',
                        'status_code': status,
                        'base': endpoint['base'],
                        'section': endpoint['section'],
                        'file': endpoint['file'],
                        'needs_correction': needs_correction,
                        'test_url': url,
                        'params_used': params,
                        'error': f"HTTP {status}"
                    }
        except asyncio.TimeoutError:
            return {
                'endpoint': endpoint['name'],
                'current_path': current_path,
                'corrected_path': corrected_path,
                'status': '⏰ TIMEOUT',
                'status_code': 0,
                'base': endpoint['base'],
                'section': endpoint['section'],
                'file': endpoint['file'],
                'needs_correction': needs_correction
            }
        except Exception as e:
            return {
                'endpoint': endpoint['name'],
                'current_path': current_path,
                'corrected_path': corrected_path,
                'status': '💥 HATA',
                'status_code': 0,
                'base': endpoint['base'],
                'section': endpoint['section'],
                'file': endpoint['file'],
                'needs_correction': needs_correction,
                'error': str(e)
            }
    
    def _get_test_params(self, path: str) -> Dict[str, str]:
        """Endpoint'e göre test parametreleri oluştur"""
        params = {}
        
        # Spot endpoints
        if 'ticker' in path or 'price' in path:
            params['symbol'] = 'BTCUSDT'
        elif 'klines' in path:
            params['symbol'] = 'BTCUSDT'
            params['interval'] = '1m'
            params['limit'] = '1'
        elif 'depth' in path:
            params['symbol'] = 'BTCUSDT'
            params['limit'] = '5'
        elif 'trades' in path:
            params['symbol'] = 'BTCUSDT'
            params['limit'] = '1'
        elif 'exchangeInfo' in path:
            pass
        elif 'avgPrice' in path:
            params['symbol'] = 'BTCUSDT'
        elif 'historicalTrades' in path:
            params['symbol'] = 'BTCUSDT'
            params['limit'] = '1'
        
        # Futures data endpoints
        elif 'globalLongShortAccountRatio' in path:
            params['symbol'] = 'BTCUSDT'
            params['period'] = '5m'
            params['limit'] = '1'
        elif 'topLongShortAccountRatio' in path:
            params['symbol'] = 'BTCUSDT'
            params['period'] = '5m'
            params['limit'] = '1'
        elif 'topLongShortPositionRatio' in path:
            params['symbol'] = 'BTCUSDT'
            params['period'] = '5m'
            params['limit'] = '1'
        elif 'takerlongshortRatio' in path:
            params['symbol'] = 'BTCUSDT'
            params['period'] = '5m'
            params['limit'] = '1'
        elif 'openInterestHist' in path:
            params['symbol'] = 'BTCUSDT'
            params['period'] = '5m'
            params['limit'] = '1'
        elif 'takerBuySellVol' in path:
            params['symbol'] = 'BTCUSDT'
            params['period'] = '5m'
            params['limit'] = '1'
        
        # Futures API endpoints
        elif 'allForceOrders' in path:
            params['symbol'] = 'BTCUSDT'
            params['limit'] = '1'
        elif 'adlQuantile' in path:
            params['symbol'] = 'BTCUSDT'
        elif 'openInterest' in path:
            params['symbol'] = 'BTCUSDT'
        
        # YENİ EKLENEN PARAMETRELER:
        elif 'continuousKlines' in path:
            params['pair'] = 'BTCUSDT'
            params['contractType'] = 'PERPETUAL' 
            params['interval'] = '1m'
            params['limit'] = '1'
        elif 'indexPriceKlines' in path:
            params['pair'] = 'BTCUSDT'
            params['interval'] = '1m'
            params['limit'] = '1'
        elif 'markPriceKlines' in path:
            params['symbol'] = 'BTCUSDT'
            params['interval'] = '1m'
            params['limit'] = '1'
        elif 'aggTrades' in path:
            params['symbol'] = 'BTCUSDT'
            params['limit'] = '1'
        elif 'allForceOrders' in path:
            params['symbol'] = 'BTCUSDT'
            params['limit'] = '1'
        elif 'takerlongshortRatio' in path:  # DÜZELTİLDİ
            params['symbol'] = 'BTCUSDT'
            params['period'] = '5m'
            params['limit'] = '10'
        
        elif 'allForceOrders' in path:
            params['symbol'] = 'BTCUSDT'
            params['limit'] = '1'
            # İsteğe bağlı: params['startTime'] = str(int(time.time() * 1000) - 3600000)  # 1 saat önce
            
            
            
        return params
        
        

    
    async def validate_yaml_files(self, yaml_files: List[str]):
        """Tüm YAML dosyalarını validate et"""
        print("🔍 Binance Endpoint Validator Başlatılıyor...\n")
        
        all_endpoints = []
        
        # Tüm YAML dosyalarından endpoint'leri topla
        for yaml_file in yaml_files:
            try:
                if not os.path.exists(yaml_file):
                    print(f"❌ {yaml_file} dosyası bulunamadı!")
                    continue
                    
                data = self.load_yaml_endpoints(yaml_file)
                endpoints = self.extract_testable_endpoints(data, yaml_file)
                all_endpoints.extend(endpoints)
                print(f"📁 {yaml_file}: {len(endpoints)} endpoint bulundu")
            except Exception as e:
                print(f"❌ {yaml_file} yüklenirken hata: {e}")
        
        print(f"\n🎯 Toplam {len(all_endpoints)} endpoint test edilecek...\n")
        
        if not all_endpoints:
            print("⚠️  Test edilecek endpoint bulunamadı!")
            return
        
        # Async testleri başlat
        async with aiohttp.ClientSession() as session:
            tasks = []
            for endpoint in all_endpoints:
                task = self.test_endpoint_async(endpoint, session)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # Sonuçları göster
            self.display_detailed_results(results)
    
    def display_detailed_results(self, results: List[Dict]):
        """Detaylı sonuçları göster"""
        successful = [r for r in results if '✅' in r['status']]
        correctable = [r for r in results if '🔄' in r['status']]
        failed = [r for r in results if '❌' in r['status'] or '💥' in r['status'] or '⏰' in r['status']]
        
        print("\n" + "="*80)
        print("🎯 DETAYLI VALIDATION SONUÇLARI")
        print("="*80)
        
        print(f"✅ BAŞARILI: {len(successful)}")
        print(f"🔄 DÜZELTİLEBİLİR: {len(correctable)}") 
        print(f"❌ HATALI: {len(failed)}")
        print(f"📝 TOPLAM DÜZELTME GEREKEN: {len(self.corrections_needed)}")
        
        # DÜZELTİLEBİLİR endpoint'leri göster
        if correctable:
            print(f"\n🟡 DÜZELTİLEBİLİR ENDPOINT'LER:")
            print("-" * 80)
            for result in correctable:
                print(f"🔄 {result['endpoint']}")
                print(f"   ESKİ Path: {result['current_path']}")
                print(f"   YENİ Path: {result['corrected_path']}")
                print(f"   Section: {result['section']}")
                if 'note' in result:
                    print(f"   Not: {result['note']}")
                print()
        
        # HATALI endpoint'leri göster
        if failed:
            print(f"\n🔴 HATALI ENDPOINT'LER:")
            print("-" * 80)
            for result in failed:
                print(f"{result['status']} {result['endpoint']}")
                print(f"   Path: {result['current_path']}")
                print(f"   File: {result['file']}")
                print(f"   Section: {result['section']}")
                if 'error' in result:
                    print(f"   Hata: {result['error']}")
                print()
        
        # BAŞARILI endpoint'leri göster (düzeltme gerekenlerle birlikte)
        if successful:
            print(f"\n🟢 BAŞARILI ENDPOINT'LER:")
            print("-" * 80)
            
            for result in successful:
                status_icon = "🔧" if result['needs_correction'] else "✅"
                print(f"{status_icon} {result['endpoint']}")
                print(f"   Path: {result['current_path']}")
                if result['needs_correction']:
                    print(f"   ⚠️  DÜZELTME ÖNERİLEN: {result['corrected_path']}")
                print(f"   Section: {result['section']}")
                print()
        
        # Düzeltme önerilerini JSON'a kaydet
        if self.corrections_needed:
            self.save_corrections_to_file()
    
    def save_corrections_to_file(self):
        """Düzeltme önerilerini JSON dosyasına kaydet"""
        corrections_data = {
            'timestamp': time.time(),
            'total_corrections': len(self.corrections_needed),
            'corrections': self.corrections_needed
        }
        
        output_file = "binance_path_corrections.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(corrections_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Düzeltme önerileri '{output_file}' dosyasına kaydedildi")
        
        # Otomatik düzeltme seçeneği sun
        self.offer_auto_correction()
    
    def offer_auto_correction(self):
        """Otomatik düzeltme teklifi"""
        if not self.corrections_needed:
            return
        
        print(f"\n🔄 OTOMATİK DÜZELTME İŞLEMİ")
        print("-" * 50)
        print(f"Toplam {len(self.corrections_needed)} endpoint düzeltilecek.")
        
        response = input("YAML dosyalarını otomatik düzeltmek istiyor musunuz? (e/h): ")
        if response.lower() in ['e', 'y', 'yes', 'evet']:
            self.apply_corrections()
        else:
            print("Düzeltme işlemi iptal edildi.")
    
    def apply_corrections(self):
        """YAML dosyalarını otomatik düzelt"""
        corrections_by_file = {}
        
        # Düzeltmeleri dosyaya göre grupla
        for correction in self.corrections_needed:
            file_name = correction['file']
            if file_name not in corrections_by_file:
                corrections_by_file[file_name] = []
            corrections_by_file[file_name].append(correction)
        
        # Her dosyayı düzelt
        for file_name, corrections in corrections_by_file.items():
            file_path = f"utils/binance_api/{file_name}"
            
            if not os.path.exists(file_path):
                print(f"❌ {file_path} dosyası bulunamadı!")
                continue
            
            try:
                # YAML dosyasını yükle
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                
                # Düzeltmeleri uygula
                corrections_applied = 0
                for correction in corrections:
                    section = correction['section']
                    endpoint_name = correction['endpoint']
                    new_path = correction['corrected_path']
                    
                    if section in data and endpoint_name in data[section]:
                        old_path = data[section][endpoint_name]['path']
                        if old_path != new_path:
                            data[section][endpoint_name]['path'] = new_path
                            corrections_applied += 1
                            print(f"   🔄 {endpoint_name}: {old_path} → {new_path}")
                
                # Düzeltilmiş dosyayı kaydet
                if corrections_applied > 0:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, indent=2)
                    
                    print(f"✅ {file_name}: {corrections_applied} düzeltme uygulandı")
                else:
                    print(f"ℹ️  {file_name}: Düzeltme gerekmiyor")
                    
            except Exception as e:
                print(f"❌ {file_name} düzeltilirken hata: {e}")

# Kullanım örneği
async def main():
    validator = BinanceEndpointValidator()
    
    # Test edilecek YAML dosyaları
    yaml_files = [
        "utils/binance_api/b-map_public.yaml",
        "utils/binance_api/b-map_private.yaml"
    ]
    
    await validator.validate_yaml_files(yaml_files)

if __name__ == "__main__":
    import time
    asyncio.run(main())