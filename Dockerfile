FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY spy_der ./spy_der
COPY scripts ./scripts

RUN pip install --no-cache-dir -e ".[dashboard]"

EXPOSE 8000 8501

CMD ["uvicorn", "spy_der.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
