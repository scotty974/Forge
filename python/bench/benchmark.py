"""Mesure du temps et de la RAM d'un traitement découpé en phases.

Équivalent Python de ``forge/src/bench``. Pour chaque phase on relève :

- la durée (``time.perf_counter``),
- le pic de tas alloué pendant la phase, au-delà de ce qui était déjà alloué
  (``tracemalloc``, qui voit aussi les tableaux numpy),
- la mémoire physique du processus (RSS / working set) en fin de phase (``psutil``).

Le rapport s'affiche dans la console et s'écrit en JSON, au même format que le
benchmark Rust, pour pouvoir les comparer avec ``compare.py``.
"""

from __future__ import annotations

import json
import math
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import psutil


@dataclass
class PhaseResult:
    name: str
    duration_s: float
    peak_heap_bytes: int  # pic pendant la phase, en plus de ce qui était alloué avant
    rss_bytes: int  # mémoire physique du processus en fin de phase


def rss_bytes() -> int:
    return psutil.Process().memory_info().rss


class Benchmark:
    def __init__(self, label: str):
        self.label = label
        self.phases: list[PhaseResult] = []
        self.metrics: dict[str, float] = {}

        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self._heap_baseline, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        self._max_heap = self._heap_baseline
        self._max_rss = rss_bytes()
        self._start = time.perf_counter()
        self._end: float | None = None

    @contextmanager
    def phase(self, name: str):
        """Mesure le bloc ``with`` : ``with bench.phase("fit"): ...``."""
        heap_before, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        t = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - t
            _, peak = tracemalloc.get_traced_memory()
            rss = rss_bytes()
            self._max_heap = max(self._max_heap, peak)
            self._max_rss = max(self._max_rss, rss)
            self.phases.append(
                PhaseResult(name, duration, max(0, peak - heap_before), rss)
            )

    def add_metric(self, name: str, value: float) -> None:
        self.metrics[name] = float(value)

    def finish(self) -> None:
        """Arrête le chronomètre global et le suivi mémoire."""
        if self._end is None:
            self._end = time.perf_counter()
            tracemalloc.stop()

    @property
    def total_s(self) -> float:
        end = self._end if self._end is not None else time.perf_counter()
        return end - self._start

    @property
    def peak_heap_bytes(self) -> int:
        return max(0, self._max_heap - self._heap_baseline)

    @property
    def peak_rss_bytes(self) -> int:
        return self._max_rss

    def print_report(self) -> None:
        print(f"\n=== Benchmark : {self.label} ===")
        print(f"{'phase':<16} {'temps (ms)':>12} {'pic tas (Ko)':>14} {'RSS (Mo)':>12}")
        for p in self.phases:
            print(
                f"{p.name:<16} {p.duration_s * 1000:>12.3f} "
                f"{p.peak_heap_bytes / 1024:>14.1f} {p.rss_bytes / 1024**2:>12.2f}"
            )
        print("-" * 57)
        print(
            f"{'TOTAL':<16} {self.total_s * 1000:>12.3f} "
            f"{self.peak_heap_bytes / 1024:>14.1f} {self._max_rss / 1024**2:>12.2f}"
        )

    def to_dict(self, extra: dict[str, str] | None = None) -> dict:
        return {
            "language": "python",
            "label": self.label,
            **(extra or {}),
            "total_ms": self.total_s * 1000,
            "peak_heap_bytes": self.peak_heap_bytes,
            "peak_rss_bytes": self._max_rss,
            "phases": [
                {
                    "name": p.name,
                    "duration_ms": p.duration_s * 1000,
                    "peak_heap_bytes": p.peak_heap_bytes,
                    "rss_bytes": p.rss_bytes,
                }
                for p in self.phases
            ],
            "metrics": {
                k: (v if math.isfinite(v) else None) for k, v in self.metrics.items()
            },
        }

    def save_json(self, path: str | Path, extra: dict[str, str] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(extra), indent=2), encoding="utf-8")
