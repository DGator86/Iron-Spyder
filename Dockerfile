FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY app app
COPY analytics analytics
COPY backtest backtest
COPY config config
COPY data data
COPY execution execution
COPY features features
COPY models models
COPY monitoring monitoring
COPY optimizer optimizer
COPY risk risk
COPY scripts scripts
COPY simulation simulation
COPY strategies strategies
COPY tests tests
RUN pip install --no-cache-dir -e .[dev]
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
