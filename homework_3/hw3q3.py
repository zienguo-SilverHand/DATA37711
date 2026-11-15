import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, log_loss

HERE = Path(__file__).resolve().parent
NUM_TREES = [1, 10, 100, 300, 1000]
SEED = 0


def find_csv(prefix):
    """Find dset0*.csv / dset1*.csv next to this script (handles names like dset0-1.csv)."""
    hits = sorted(glob.glob(str(HERE / f"{prefix}*.csv")))
    if not hits:
        raise FileNotFoundError(f"no file matching {prefix}*.csv in {HERE}")
    return hits[0]


def run(path):
    df = pd.read_csv(path)
    y = df.iloc[:, 0].to_numpy()                 # first column is the target
    X = df.iloc[:, 1:].to_numpy()
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # binary 0/1 target -> classification, otherwise regression
    is_clf = set(np.unique(y)) <= {0, 1}
    if is_clf:
        model = GradientBoostingClassifier(n_estimators=max(NUM_TREES), random_state=SEED)
    else:
        model = GradientBoostingRegressor(n_estimators=max(NUM_TREES), random_state=SEED)
    model.fit(X_tr, y_tr)

    # staged predictions give the model after 1, 2, ..., 1000 trees from a single fit
    if is_clf:
        stage_tr = list(model.staged_predict_proba(X_tr))
        stage_ts = list(model.staged_predict_proba(X_ts))
        def err(y_true, proba):                  # error rate and log-loss
            return np.mean((proba[:, 1] > 0.5) != y_true), log_loss(y_true, proba, labels=[0, 1])
    else:
        stage_tr = list(model.staged_predict(X_tr))
        stage_ts = list(model.staged_predict(X_ts))
        def err(y_true, pred):
            return (mean_squared_error(y_true, pred),)

    rows = []
    for M in NUM_TREES:
        tr, ts = err(y_tr, stage_tr[M - 1]), err(y_ts, stage_ts[M - 1])
        row = {"trees": M, "train": tr[0], "test": ts[0]}
        if is_clf:
            row.update({"train log-loss": tr[1], "test log-loss": ts[1]})
        rows.append(row)
    table = pd.DataFrame(rows).set_index("trees")

    # full curves for the plot
    curve_tr = [err(y_tr, p)[0] for p in stage_tr]
    curve_ts = [err(y_ts, p)[0] for p in stage_ts]
    best_M = int(np.argmin(curve_ts)) + 1

    name = Path(path).stem
    metric = "error rate" if is_clf else "MSE"
    print(f"\n{name}: n={len(df)}, {'classification' if is_clf else 'regression'} ({metric})")
    print(table.round(4))
    print(f"lowest test {metric} at {best_M} trees: {min(curve_ts):.4f}")
    return name, metric, curve_tr, curve_ts


results = [run(find_csv(p)) for p in ("dset0", "dset1")]

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, metric, c_tr, c_ts) in zip(axes, results):
    m = np.arange(1, len(c_tr) + 1)
    ax.plot(m, c_tr, label="train")
    ax.plot(m, c_ts, label="test")
    ax.scatter(NUM_TREES, [c_ts[k - 1] for k in NUM_TREES], color="C1", zorder=3)
    ax.set_xscale("log")
    ax.set(title=name, xlabel="number of trees", ylabel=metric)
    ax.legend()
plt.tight_layout()
plt.savefig(HERE / "boosting_curves.png", dpi=150)
plt.show()