# Iron-Spyder (SPY-DER Foundation)

Production-oriented **SPY-only, defined-risk options** foundation for research, backtesting, paper trading, and future execution interfaces.

## Architecture

- `app/` FastAPI API + Streamlit dashboard
- `config/` YAML mode configs (`research`, `paper`, `live`)
- `data/` typed schemas, synthetic chain generator, persistence scaffolding
- `analytics/` Black-Scholes + Greeks + market analytics
- `features/` feature registry/versioning seed
- `models/` rule-based market-state classifier
- `simulation/` path simulation seed
- `strategies/` strategy templates, payoff metrics, deterministic risk validator
- `optimizer/` market-state strategy filtering and ranking vs no-trade
- `risk/` risk limits + kill switch
- `execution/` broker abstraction + paper broker
- `backtest/`, `monitoring/`, `scripts/`, `tests/` foundations

## Requirements

- Python 3.12
- FastAPI, Streamlit, Pydantic, SQLAlchemy, SQLite-compatible design
- NumPy, pandas, SciPy, scikit-learn
- pytest, Ruff, mypy
- Docker + Docker Compose

## Quickstart (Docker)

```bash
docker compose up --build
```

API: `http://localhost:8000`  
Dashboard: `http://localhost:8501`

## Quickstart (No Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.api:app --reload
```

## Demo commands

```bash
python -m scripts.seed
python -m scripts.demo
```

## Safety constraints

- SPY options only
- No naked short calls/puts
- No unlimited-risk structures
- No share-required strategies
- Every accepted strategy must have finite, deterministic max loss before entry
- Paper trading only by default (no unattended live execution)
