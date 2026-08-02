from __future__ import annotations

from app.state import STATE

if __name__ == "__main__":
    snap = STATE.snapshot()
    print({"spy": snap["quote"].last, "expected_move": snap["expected_move"]})
