from __future__ import annotations


def walk_forward_windows(total: int, train: int, test: int) -> list[tuple[range, range]]:
    windows: list[tuple[range, range]] = []
    i = 0
    while i + train + test <= total:
        windows.append((range(i, i + train), range(i + train, i + train + test)))
        i += test
    return windows
