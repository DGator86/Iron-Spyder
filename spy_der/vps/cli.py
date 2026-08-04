"""CLI dispatcher for VPS-facing commands.

    spyder-vps supervisor …      # decision loop + live_state publisher
    spyder-vps dashboard-api …   # read-only status HTTP
    spyder-vps status …          # print system status to stdout
"""

from __future__ import annotations

import json
import sys


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(__doc__)
        print("Commands:")
        print("  supervisor       Run the VPS decision supervisor")
        print("  dashboard-api    Desk/status HTTP API (journal + optimize + live state)")
        print("  status           Print system status JSON for the state root")
        print("  import-spyder    Inspect/load/backtest SPY-DER market recordings")
        return 0 if args else 2

    cmd, rest = args[0], args[1:]

    if cmd in {"supervisor", "service", "run"}:
        from spy_der.vps.service import main as supervisor_main

        return supervisor_main(rest)

    if cmd in {"dashboard-api", "dashboard_api", "dashboard"}:
        from spy_der.vps.dashboard_api import main as dashboard_main

        return dashboard_main(rest)

    if cmd == "status":
        return _status(rest)

    if cmd in {"import-spyder", "import_spyder", "import"}:
        from scripts.import_spyder import main as import_main

        return import_main(rest)

    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


def _status(argv: list[str]) -> int:
    import argparse

    from spy_der.vps.paths import state_paths
    from spy_der.vps.system_status import build_system_status

    parser = argparse.ArgumentParser(description="Print VPS system status")
    parser.add_argument("--state-root", default=None)
    args = parser.parse_args(argv)
    root = args.state_root or state_paths().root
    print(json.dumps(build_system_status(root), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
