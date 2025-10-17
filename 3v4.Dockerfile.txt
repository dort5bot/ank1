#v4
#Dockerfile

#	python:3.11-slim tabanlı
#	Multi-stage build
#	--no-cache-dir, --no-install-recommends kullanımı
#	entrypoint.sh ile esnek başlangıç
#	.dockerignore uyumlu
#	BuildKit uyumlu
#	Minimal ve güvenli

#Dockerfile
# ----------------------
# Build Aşaması
# ----------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Build bağımlılıklarını kur (minimal ve temiz)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    libc-dev \
    libffi-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Pip araçlarını güncelle (cache’siz)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Bağımlılık dosyasını kopyala
COPY requirements.txt .

# Wheel olarak bağımlılıkları derle
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# requirements.txt'yi de kopyala (runtime için)
RUN cp requirements.txt /app/wheels/


# ----------------------
# Runtime Aşaması
# ----------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Sadece gerekli bağımlılıkları kur
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Uygulama için kullanıcı oluştur
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /usr/sbin/nologin --create-home appuser

# Python ayarları
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPYCACHEPREFIX=/tmp \
    PIP_NO_CACHE_DIR=1

# Build aşamasından wheel'ları kopyala
COPY --from=builder /app/wheels /wheels

# Bağımlılıkları kur ve wheel'ları temizle
RUN pip install --no-index --find-links=/wheels -r /wheels/requirements.txt \
    && rm -rf /wheels

# Gerekli uygulama dosyalarını kopyala (sadece gerekenleri .dockerignore ile uyumlu)
COPY --chown=appuser:appgroup main.py . 
COPY --chown=appuser:appgroup app/ ./app/
COPY --chown=appuser:appgroup entrypoint.sh /entrypoint.sh

# Entrypoint scriptini çalıştırılabilir yap
RUN chmod +x /entrypoint.sh

# Metadata bilgileri (isteğe bağlı ama faydalı)
LABEL maintainer="sen@example.com" \
      org.opencontainers.image.source="https://github.com/kullanici/proje" \
      org.opencontainers.image.licenses="MIT"

# Health check (isteğe bağlı endpoint varsa)
EXPOSE 3000
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Uygulama kullanıcısına geç
USER appuser

# Entrypoint ile başlat
ENTRYPOINT ["/entrypoint.sh"]
