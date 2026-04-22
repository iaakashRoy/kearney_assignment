"""
Benchmark cost data loading and lookup helpers.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from app.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_benchmark() -> list[dict]:
    """Load benchmark cost data once; returns [] if not found."""
    csv_path = settings.benchmark_csv_path
    if csv_path:
        candidates = [Path(csv_path)]
    else:
        # Bundled inside the package (app/data/) and the legacy /-mounted Docker location.
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "data" / "benchmark_component_costs.csv",
            Path("/benchmark_component_costs.csv"),
        ]

    for path in candidates:
        if path.exists():
            logger.info("Loaded benchmark CSV from %s", path)
            with open(path, newline="") as fh:
                return list(csv.DictReader(fh))

    logger.warning("benchmark_component_costs.csv not found — cost lookup disabled")
    return []


def lookup_cost(component_type: str, material: str, size_category: str) -> dict | None:
    benchmark = load_benchmark()
    # Exact match
    for row in benchmark:
        if (
            row.get("component_type") == component_type
            and row.get("material") == material
            and row.get("size_category") == size_category
        ):
            return row
    # Fallback: component_type + material
    for row in benchmark:
        if row.get("component_type") == component_type and row.get("material") == material:
            return row
    # Fallback: component_type only
    for row in benchmark:
        if row.get("component_type") == component_type:
            return row
    return None
