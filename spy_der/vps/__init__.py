"""VPS runtime: state root, heartbeats, live state, and deployable services.

Decision math stays in :mod:`spy_der.pipeline` and :mod:`spy_der.runtime`.
This package owns only what it takes to run that system on a VPS — the state
tree layout, liveness files the dashboard can read without SSH, the long-running
supervisor service, the read-only status API, and the pull-based deploy path.
"""

from spy_der.vps.heartbeat import classify_age, read_heartbeats, write_heartbeat
from spy_der.vps.live_state import build_live_state, write_live_state
from spy_der.vps.paths import DEFAULT_STATE_ROOT, STATE_DIRS, state_paths
from spy_der.vps.service import VpsService, VpsServiceConfig
from spy_der.vps.system_status import build_system_status

__all__ = [
    "DEFAULT_STATE_ROOT",
    "STATE_DIRS",
    "VpsService",
    "VpsServiceConfig",
    "build_live_state",
    "build_system_status",
    "classify_age",
    "read_heartbeats",
    "state_paths",
    "write_heartbeat",
    "write_live_state",
]
