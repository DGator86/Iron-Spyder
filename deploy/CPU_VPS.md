# Iron-Spyder dedicated CPU VPS

**Authoritative deployment guidance.** Iron-Spyder runs on its own CPU machine.
It does not share a host with SPY-DER, 0DTE, or a GPU server.

## Why CPU, not GPU

The live stack depends on NumPy, SciPy, pandas, scikit-learn, FastAPI, Uvicorn,
and Streamlit. It does **not** include PyTorch, TensorFlow, CUDA, GPU-XGBoost,
JAX, or another GPU framework.

The workload is:

- Options-chain ingestion and validation
- Greeks, exposure, volatility-surface, and regime calculations
- Monte Carlo path simulation (CPU)
- Strategy generation and payoff evaluation
- FastAPI / Streamlit services
- SQLite state persistence
- Periodic decision cycles

Those are CPU, memory, storage, and network-reliability problems. A GPU on this
host would sit idle and waste money. The Docker image and Compose files assert
CPU-only (`CUDA_VISIBLE_DEVICES=""`, no nvidia runtime, no device reservations).

## Isolation rule

| System | Host | Paths |
|---|---|---|
| Iron-Spyder (this repo) | **Dedicated CPU VPS** | `/opt/iron-spyder`, `/var/lib/iron-spyder`, `/etc/iron-spyder` |
| Legacy SPY-DER / 0DTE | Separate machine (if still running) | never referenced here |

Do not co-tenant. Separate CPU, separate disk, separate secrets file.

## Live VPS (paper / production)

| Resource | Recommendation |
|---|---|
| CPU | 8 dedicated or high-performance vCPUs |
| RAM | 32 GB |
| Storage | 250–500 GB NVMe SSD |
| OS | Ubuntu 24.04 LTS |
| Architecture | x86-64 |
| Network | Reliable low-latency US-East |
| Deployment | Docker Compose (`docker-compose.yml`) |
| Database | SQLite initially; PostgreSQL later |
| Backups | Daily encrypted state + configuration |
| GPU | **None** |

A 4-vCPU / 16 GB box can boot the app, but is not the primary research-and-
production machine: backtests, Monte Carlo, historical ingestion, API traffic,
and the dashboard will contend for the same cores.

### What runs on the live VPS

- Market-data ingestion
- Real-time inference (CPU)
- Strategy selection and risk controls
- Paper or (later) live execution
- API + dashboard + status API
- State persistence and monitoring

Priority: reliability and deterministic response time — not maximum compute.

```bash
cp .env.example .env          # fill provider keys; leave ALLOW_LIVE=0
docker compose up -d --build
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8788/v1/system
```

Or first-time pull-based install as root:

```bash
curl -fsSL https://raw.githubusercontent.com/DGator86/Iron-Spyder/main/deploy/remote-deploy.sh | bash
```

## Research worker (second CPU machine)

Do **not** run giant historical optimizations on the machine that manages
positions.

| Resource | Recommendation |
|---|---|
| CPU | 16–32 cores |
| RAM | 64–128 GB |
| Storage | 1–2 TB NVMe |
| Form | Temporary / interruptible CPU compute where practical |
| GPU | Still none, until models exist that use it |

Use for walk-forward backtests, parameter sweeps, Monte Carlo calibration,
cost-stress testing, feature research, and model retraining.

```bash
docker compose -f docker-compose.research.yml run --rm research \
    python -m scripts.backtest
```

## When a GPU would become justified

Only after the codebase grows models that can use one:

- PyTorch temporal transformers
- Deep volatility-surface models
- Sequence models on full option-chain tensors
- Large neural ensembles
- GPU-accelerated XGBoost / CatBoost
- CUDA/JAX path simulation
- Reinforcement-learning research

Even then: train on an **on-demand GPU** machine, export the artifact, deploy
**CPU inference** to the live VPS. Do not put a GPU in the live trading server.
For SPY alone, inference should stay lightweight enough for CPU.

## Spend order (before GPU)

1. High-quality historical SPY options data
2. Reliable live options data
3. Sufficient CPU and storage for walk-forward validation
4. Monitoring, backups, and execution safety
5. GPU experimentation only afterward

## Final recommendation

Deploy Iron-Spyder on an **8-vCPU, 32 GB, NVMe, no-GPU** VPS of its own, and
use temporary larger **CPU** machines for research. Do not pay for a GPU yet.
