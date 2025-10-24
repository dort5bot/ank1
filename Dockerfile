# =========================
# Dockerfile v24
# =========================
# numpy sürümü >=1.23.2,<2.0 olarak sabitlendi → pandas ve PyWavelets ile uyumlu.
# -------------------------
# Build aşaması
# -------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Build bağımlılıklarını kur
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    python3-dev \
    libffi-dev \
    libssl-dev \
    libopenblas-dev \
    liblapack-dev \
    rustc \
    cargo \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pip, setuptools, wheel güncelle
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# requirements.txt kopyala
COPY requirements.txt .

# Önce uyumlu numpy sürümünü yükle
RUN pip install --no-cache-dir "numpy>=1.23.2,<2.0"

# Kalan paketleri wheel olarak derle
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# requirements.txt'yi de wheels klasörüne kopyala
RUN cp requirements.txt /app/wheels/

# -------------------------
# Runtime aşaması
# -------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Sadece runtime için gerekli paketler
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Uygulama kullanıcısı oluştur
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

# Python optimizasyonları
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX=/tmp \
    PIP_NO_CACHE_DIR=1

# Wheel'ları ve requirements.txt'yi kopyala
COPY --from=builder /app/wheels /wheels

# Wheel'lardan paketleri yükle
RUN pip install --no-index --find-links=/wheels -r /wheels/requirements.txt \
    && rm -rf /wheels

# Uygulama kodunu kopyala
COPY --chown=appuser:appgroup . .

# Healthcheck ve port
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Çalışma kullanıcısını ayarla
USER appuser

# Başlatma komutu
CMD ["python", "main.py"]
