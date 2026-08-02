# Iron-Spyder — CPU-only production image.
#
# No CUDA base image, no PyTorch/TensorFlow. The live workload is
# NumPy/SciPy/pandas/scikit-learn plus FastAPI/Uvicorn/Streamlit. A GPU would
# sit idle on this image; do not add one to the host either.

FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="iron-spyder" \
      org.opencontainers.image.description="SPY defined-risk options intelligence (CPU-only)" \
      com.iron-spyder.compute="cpu"

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IRON_SPYDER_STATE_ROOT=/var/lib/iron-spyder \
    # Explicit: never prefer a GPU/CUDA runtime even if one appears on the host.
    CUDA_VISIBLE_DEVICES="" \
    NVIDIA_VISIBLE_DEVICES=void

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY spy_der ./spy_der
COPY scripts ./scripts

RUN pip install --no-cache-dir -e ".[dashboard]" \
    && useradd --system --uid 10001 --home-dir /app --shell /usr/sbin/nologin iron-spyder \
    && mkdir -p /var/lib/iron-spyder \
    && chown -R iron-spyder:iron-spyder /app /var/lib/iron-spyder

USER iron-spyder

EXPOSE 8000 8501 8788

VOLUME ["/var/lib/iron-spyder"]

# Healthchecks live on the compose `api` service — this image is reused by
# supervisor / status / dashboard containers that do not bind :8000.

CMD ["uvicorn", "spy_der.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
