"""Day-1 script.

Fetches VNC + descending-neuron subset from male-cns:v1.0, caches to data/,
prints a summary. First run: ~1-2 min network + a few MB of parquet.
Subsequent runs: < 1 s from cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.neuprint_client import fetch_vnc_subset


def main():
    subset = fetch_vnc_subset()
    print(subset.summary())


if __name__ == "__main__":
    main()
