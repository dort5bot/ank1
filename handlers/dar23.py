# handlers/dar_handler.py
"""
v3.1 - Güncellenmiş dosya filtresi
# Aiogram 3.x uyumlu
# Proje yedekleme ve komut tarama yardımcı handler
. ile başlayan dosyalar ve __pycache__ gibi klasörler yok sayılır.
/dar → proje ağaç yapısını mesaj olarak gösterir.
/dar k → tüm @router.message(Command(...)) komutlarını bulur
/dar t → proje ağaç yapısı + dosya içeriğini birleştirip, her dosya için başlık ekleyerek mesaj halinde .txt dosyası olarak gönderir.
/dar Z → tüm proje klasörünü .zip dosyası olarak gönderir.
# zaman format: mbot1_0917_2043 (aygün_saaddkika) ESKİ: "%Y%m%d_%H%M%S" = YılAyGün_SaatDakikaSaniye
"""

import os
import re
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime

from aiogram import Router
from aiogram.types import Message, FSInputFile
from aiogram.filters import Command

# Router
router = Router()

# Kök dizin (proje kökü)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Geçici dosya dizini (Render uyumlu)
TMP_DIR = Path(tempfile.gettempdir())
TMP_DIR.mkdir(parents=True, exist_ok=True)

TELEGRAM_NAME = os.getenv("TELEGRAM_NAME", "hbot")
TELEGRAM_MSG_LIMIT = 4000

# İzin verilen dosya uzantıları (sadece bu dosyalar dahil edilecek)
ALLOWED_EXTENSIONS = {
    '.py', '.txt', '.md', '.yaml', '.yml', '.json', '.ini', '.cfg', '.conf',
    'Dockerfile', '.dockerignore', 'docker-compose.yml', 'docker-compose.yaml',
    '.env', '.gitignore', 'requirements.txt', 'setup.py', 'pyproject.toml'
}

# İzin verilen dosya isimleri (uzantısız önemli dosyalar)
ALLOWED_FILENAMES = {
    'Dockerfile', 'docker-compose.yml', 'docker-compose.yaml', 
    'requirements.txt', '.env', '.gitignore', 'setup.py'
}


# -------------------------------
# 📂 Proje ağaç yapısı üretici
# -------------------------------
def generate_tree(path: Path, prefix: str = "") -> str:
    tree = ""
    entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
    for idx, entry in enumerate(entries):
        if entry.name.startswith(".") or entry.name in ["__pycache__"]:
            continue
        connector = "└── " if idx == len(entries) - 1 else "├── "
        tree += f"{prefix}{connector}{entry.name}\n"
        if entry.is_dir():
            extension = "    " if idx == len(entries) - 1 else "│   "
            tree += generate_tree(entry, prefix + extension)
    return tree


# -------------------------------
# 🔍 handlers içindeki komut tarayıcı
# -------------------------------
def scan_handlers_for_commands():
    commands = {}
    handler_dir = PROJECT_ROOT / "handlers"

    pattern = re.compile(r'@router\.message\(.*Command\(["\'](\w+)["\']')
    for fname in os.listdir(handler_dir):
        if not fname.endswith(".py") or fname.startswith("__"):
            continue
        fpath = handler_dir / fname
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
            matches = pattern.findall(content)
            for cmd in matches:
                commands[f"/{cmd}"] = f"({fname})"
        except Exception:
            continue
    return commands


# -------------------------------
# 🎯 Dosya Filtreleme Fonksiyonu
# -------------------------------
def should_include_file(file_path: Path) -> bool:
    """Dosyanın dahil edilip edilmeyeceğine karar verir"""
    filename = file_path.name
    
    # Gizli dosyaları atla
    if filename.startswith('.'):
        return False
    
    # Önbellek ve geçici dizinleri atla
    if filename in ['__pycache__', '__pycache__', 'node_modules', '.git']:
        return False
    
    # İzin verilen dosya isimlerini kontrol et
    if filename in ALLOWED_FILENAMES:
        return True
    
    # İzin verilen uzantıları kontrol et
    if file_path.suffix in ALLOWED_EXTENSIONS:
        return True
    
    return False


# -------------------------------
# 🎯 Komut Handler
# -------------------------------
@router.message(Command("dar"))
async def dar_command(message: Message):
    args = message.text.strip().split()[1:]
    mode = args[0].lower() if args else ""

    # Yeni format: mbot1_0917_2043 (aygün_saaddkika) ESKİ: "%Y%m%d_%H%M%S" = YılAyGün_SaatDakikaSaniye
    timestamp = datetime.now().strftime("%m%d_%H%M%S")

    # --- Komut Tarama (/dar k)
    if mode == "k":
        scanned = scan_handlers_for_commands()
        lines = [f"{cmd} → {desc}" for cmd, desc in sorted(scanned.items())]
        text = "\n".join(lines) if lines else "❌ Komut bulunamadı."
        await message.answer(f"<pre>{text}</pre>", parse_mode="HTML")
        return

    # --- TXT Kod Birleştir (/dar t) - GÜNCELLENDİ: FİLTRELEME EKLENDİ

    # --- TXT Kod Birleştir (/dar t) - GÜNCELLENDİ: FİLTRELEME + HEDEF DESTEKLİ
    if mode == "t":
        # Hedef argüman (ör: "ayva" veya "elma.py")
        target = args[1].strip() if len(args) > 1 else None

        content_blocks = []

        # Önce proje ağaç yapısını ekle
        tree_str = generate_tree(PROJECT_ROOT)
        content_blocks.append("📁 PROJE AĞAÇ YAPISI\n")
        content_blocks.append(tree_str)
        content_blocks.append("\n" + "="*50 + "\n")
        content_blocks.append("📄 DOSYA İÇERİKLERİ\n")
        content_blocks.append("="*50 + "\n")

        # 🔍 Dosya hedefi varsa yol belirle
        target_path = None
        if target:
            possible_path = PROJECT_ROOT / target
            if possible_path.exists():
                target_path = possible_path
            else:
                # Alt klasörlerde arama (örnek: ayva klasörü)
                for root, dirs, files in os.walk(PROJECT_ROOT):
                    if target in dirs or target in files:
                        target_path = Path(root) / target
                        break

        # 🔁 Dosyaları dolaş
        for dirpath, _, filenames in os.walk(PROJECT_ROOT):
            dir_path = Path(dirpath)

            # Eğer hedef klasör belirlenmişse, sadece o klasör içinde ara
            if target_path and target_path.is_dir() and target_path not in dir_path.parents and dir_path != target_path:
                continue

            for fname in sorted(filenames):
                file_path = dir_path / fname

                # Eğer hedef dosya belirlenmişse sadece o dosya
                if target_path and target_path.is_file() and file_path != target_path:
                    continue

                if not should_include_file(file_path):
                    continue

                rel_path = file_path.relative_to(PROJECT_ROOT)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                except Exception:
                    continue

                block = (
                    "\n" + "=" * 30 + "\n"
                    f"|| {rel_path.as_posix()} ||\n"
                    + "=" * 30 + "\n"
                    + file_content.strip() + "\n"
                )
                content_blocks.append(block)

        # 🔖 Dosya adı formatı
        name_part = target if target else "full"
        txt_path = TMP_DIR / f"{TELEGRAM_NAME}_{name_part}_{timestamp}.txt"

        full_content = "\n".join(content_blocks)

        # Uzunsa dosya gönder
        if len(full_content) > TELEGRAM_MSG_LIMIT:
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(full_content)
                await message.answer_document(FSInputFile(str(txt_path)))
            except Exception as e:
                await message.answer(f"Hata oluştu: {e}")
            finally:
                if txt_path.exists():
                    txt_path.unlink()
        else:
            await message.answer(f"<pre>{full_content}</pre>", parse_mode="HTML")

        return

   
    
    
    
    # --- ZIP Yedek (/dar Z) - GÜNCELLENDİ: FİLTRELEME EKLENDİ
    if mode.upper() == "Z":
        zip_path = TMP_DIR / f"{TELEGRAM_NAME}_{timestamp}.zip"
        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(PROJECT_ROOT):
                    for file in files:
                        file_path = Path(root) / file
                        
                        # Dosya filtresini uygula
                        if not should_include_file(file_path):
                            continue
                            
                        rel_path = file_path.relative_to(PROJECT_ROOT)
                        try:
                            zipf.write(file_path, rel_path)
                        except Exception:
                            continue
            await message.answer_document(FSInputFile(str(zip_path)))
        except Exception as e:
            await message.answer(f"Hata oluştu: {e}")
        finally:
            if zip_path.exists():
                zip_path.unlink()
        return

    # --- Varsayılan (/dar → ağaç mesaj)
    tree_str = generate_tree(PROJECT_ROOT)
    if len(tree_str) > TELEGRAM_MSG_LIMIT:
        txt_path = TMP_DIR / f"{TELEGRAM_NAME}_{timestamp}.txt"
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(tree_str)
            await message.answer_document(FSInputFile(str(txt_path)))
        except Exception as e:
            await message.answer(f"Hata oluştu: {e}")
        finally:
            if txt_path.exists():
                txt_path.unlink()
    else:
        await message.answer(f"<pre>{tree_str}</pre>", parse_mode="HTML")