from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def geometric_brownian_paths(spot: float, vol: float, steps: int, n_paths: int, dt: float = 1 / 252) -> NDArray[np.float64]:
    shocks = np.random.normal(0, 1, size=(n_paths, steps))
    increments = (-(vol**2) / 2) * dt + vol * np.sqrt(dt) * shocks
    return np.asarray(spot * np.exp(np.cumsum(increments, axis=1)), dtype=np.float64)
