from __future__ import annotations

from app.state import STATE

if __name__ == "__main__":
    snapshot = STATE.snapshot()
    print("Market state:", snapshot["market_state"].state)
    print("Top strategy:", snapshot["best"])
