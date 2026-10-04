# PgmForge 容器镜像
#
#   docker build -t pgmforge:0.1.0 .
#   docker run --rm pgmforge:0.1.0 python cli.py demo
#
# 作者：晨星

FROM python:3.13-slim

LABEL org.opencontainers.image.title="PgmForge" \
      org.opencontainers.image.description="Probabilistic graphical model inference with exact + approximate engines" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.authors="晨星" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY . .

# 非 root 运行
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

ENV PATH="/app:${PATH}"

ENTRYPOINT ["python", "cli.py"]
CMD ["demo"]
