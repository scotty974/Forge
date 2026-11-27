"""Compare les benchmarks Rust (Forge) et Python (scikit-learn).

Lit les fichiers JSON écrits dans result/ :
    result/bench_rust.json            (cargo run --release, dans forge/)
    result/bench_python_linear.json   (python main.py)
    result/bench_python_sgd.json      (python main.py --model sgd)

Usage (depuis le dossier python/) :
    python compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULT = Path(__file__).resolve().parent.parent / "result"
FILES = ["bench_rust.json", "bench_python_linear.json", "bench_python_sgd.json"]


def load() -> list[dict]:
    runs = []
    for name in FILES:
        path = RESULT / name
        if path.exists():
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            print(f"(absent : {path})")
    return runs


def short(run: dict) -> str:
    return f"{run['language']}:{run.get('model', '?').split(' ')[0]}"


def main() -> None:
    runs = load()
    if not runs:
        print("Aucun benchmark trouvé : lancer d'abord forge (cargo run --release) et python main.py")
        return

    names = [short(r) for r in runs]
    width = max(22, *(len(n) + 2 for n in names))

    def row(label: str, values: list[str]) -> None:
        print(f"{label:<18}" + "".join(f"{v:>{width}}" for v in values))

    print("\n=== Comparaison ===")
    row("", names)
    row("build", [r.get("build", "?") for r in runs])
    print("-" * (18 + width * len(runs)))

    # Temps par phase
    phases = [p["name"] for p in runs[0]["phases"]]
    for phase in phases:
        vals = []
        for r in runs:
            d = {p["name"]: p for p in r["phases"]}.get(phase)
            vals.append(f"{d['duration_ms']:.3f} ms" if d else "-")
        row(f"t {phase}", vals)
    row("TEMPS TOTAL", [f"{r['total_ms']:.3f} ms" for r in runs])
    print("-" * (18 + width * len(runs)))

    row("pic tas", [f"{r['peak_heap_bytes'] / 1024:.1f} Ko" for r in runs])
    row("pic RSS", [f"{r['peak_rss_bytes'] / 1024**2:.2f} Mo" for r in runs])
    print("-" * (18 + width * len(runs)))

    keys = []
    for r in runs:
        keys += [k for k in r.get("metrics", {}) if k not in keys]
    for k in keys:
        vals = []
        for r in runs:
            v = r.get("metrics", {}).get(k)
            vals.append("-" if v is None else (f"{v:.0f}" if k == "epochs" else f"{v:.4f}"))
        row(k, vals)

    # Rapports par rapport à Rust
    rust = next((r for r in runs if r["language"] == "rust"), None)
    if rust:
        print("\nRapport Python / Rust (> 1 : Rust est meilleur)")
        for r in runs:
            if r is rust:
                continue
            t = r["total_ms"] / rust["total_ms"] if rust["total_ms"] else float("nan")
            h = (
                r["peak_heap_bytes"] / rust["peak_heap_bytes"]
                if rust["peak_heap_bytes"]
                else float("nan")
            )
            m = r["peak_rss_bytes"] / rust["peak_rss_bytes"] if rust["peak_rss_bytes"] else float("nan")
            print(f"  {short(r):<20} temps x{t:.2f}   pic tas x{h:.2f}   RSS x{m:.2f}")


if __name__ == "__main__":
    main()
