"""Régression linéaire multiple sur le dataset immobilier avec scikit-learn.

Même pipeline que ``forge/src/main.rs`` (chargement, split 80/20, StandardScaler,
entraînement, prédiction, métriques), mesuré par le module ``bench`` pour
comparer les performances avec Forge.

Usage (depuis le dossier python/) :
    python main.py                # LinearRegression (solution exacte, moindres carrés)
    python main.py --model sgd    # SGDRegressor (descente de gradient stochastique)
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "immobilier.csv"
RESULT = ROOT / "result"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["linear", "sgd"], default="linear")
    parser.add_argument("--no-plot", action="store_true", help="ne pas générer le graphique")
    args = parser.parse_args()

    # Imports hors benchmark : le coût de démarrage de Python/sklearn n'est pas mesuré,
    # comme le chargement du binaire côté Rust.
    import pandas as pd
    from sklearn.linear_model import LinearRegression, SGDRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from bench import Benchmark

    bench = Benchmark(f"immobilier - regression lineaire multiple (sklearn {args.model})")

    with bench.phase("chargement"):
        df = pd.read_csv(DATASET)
        x = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
    print(f"{x.shape[0]} lignes, {x.shape[1]} variables")

    with bench.phase("split"):
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

    with bench.phase("scaling"):
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

    with bench.phase("entrainement"):
        if args.model == "linear":
            model = LinearRegression()
        else:
            # Proche de Forge : descente de gradient, arrêt quand la loss ne baisse plus
            model = SGDRegressor(
                learning_rate="constant", eta0=0.005, max_iter=100_000, tol=1e-6,
                random_state=42,
            )
        model.fit(x_train_scaled, y_train)

    with bench.phase("prediction"):
        pred_train = model.predict(x_train_scaled)
        pred = model.predict(x_test_scaled)

    with bench.phase("metriques"):
        r2_train = r2_score(y_train, pred_train)
        r2_test = r2_score(y_test, pred)
        mae_test = mean_absolute_error(y_test, pred)

    bench.finish()

    print(f"\nR2  train : {r2_train:.4f}")
    print(f"R2  test  : {r2_test:.4f}")
    print(f"MAE test  : {mae_test:.4f} points")

    bench.add_metric("r2_train", r2_train)
    bench.add_metric("r2_test", r2_test)
    bench.add_metric("mae_test", mae_test)
    if hasattr(model, "n_iter_"):
        bench.add_metric("epochs", model.n_iter_)
    bench.print_report()

    out = RESULT / f"bench_python_{args.model}.json"
    bench.save_json(
        out,
        extra={
            "dataset": "../data/immobilier.csv",
            "model": type(model).__name__,
            "build": "python",
            "n_samples": str(x.shape[0]),
        },
    )
    print(f"Benchmark : {out}")

    # Graphique hors benchmark, comme côté Rust
    if not args.no_plot:
        plot_predictions(y_test, pred, RESULT / f"python_predictions_{args.model}.png")


def plot_predictions(y_true, y_pred, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.24, 7.68))
    ax.scatter(y_true, y_pred, s=12, alpha=0.6)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1, label="prédiction parfaite")
    ax.set_xlabel("prix réel (k€)")
    ax.set_ylabel("prix prédit (k€)")
    ax.set_title("Prédictions vs réel (scikit-learn)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"Courbe de prediction : {path}")


if __name__ == "__main__":
    main()
