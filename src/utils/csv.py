"""CSV output shared by benchmark reporting and profiling."""

import csv
from pathlib import Path


def write_csv(path, rows, fieldnames=None, *, append=False, extrasaction="raise"):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        fieldnames
        if fieldnames is not None
        else tuple(dict.fromkeys(key for row in rows for key in row))
    )
    header = not append or not path.exists() or path.stat().st_size == 0
    with path.open("a" if append else "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction=extrasaction)
        if header:
            writer.writeheader()
        writer.writerows(rows)
