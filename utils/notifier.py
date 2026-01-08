# utils/notifier.py
# telegram bildirim modülü

import os
import aiohttp
import logging
import time

class TelegramNotifier:
    def __init__(self):
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.admin_id = os.getenv("ADMIN_IDS")
        self.api_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        
        # --- SPAM FİLTRESİ AYARLARI ---
        self.last_sent = {}  # { "BTC": 1704712345, "SOL": 1704715678 }
        self.cooldown = 3600 # Saniye cinsinden bekleme süresi (3600s = 1 Saat)

    async def send_notification(self, text: str, symbol: str = None, parse_mode: str = "HTML"):
        """
        text: Gönderilecek mesaj
        symbol: Eğer bir coin bildirimi ise sembol ismi (Spam filtresi için)
        """
        if not self.token or not self.admin_id:
            logging.warning("Bildirim gönderilemedi: Token veya Chat ID eksik.")
            return False

        # --- SPAM KONTROLÜ ---
        now = time.time()
        if symbol:
            last_time = self.last_sent.get(symbol, 0)
            if now - last_time < self.cooldown:
                logging.info(f"🚫 {symbol} için bekleme süresi dolmadı. Bildirim atlanıyor.")
                return False
            
            # Gönderim başarılı olursa süreyi güncelleyeceğiz
            self.last_sent[symbol] = now

        payload = {
            "chat_id": self.admin_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(self.api_url, json=payload) as response:
                    success = response.status == 200
                    if not success and symbol in self.last_sent:
                        # Eğer gönderim başarısız olursa, süreyi sıfırla ki tekrar denesin
                        del self.last_sent[symbol]
                    return success
            except Exception as e:
                logging.error(f"Bildirim Hatası: {e}")
                if symbol in self.last_sent: del self.last_sent[symbol]
                return False