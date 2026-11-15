import os
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
SEED = 0

# ---------- data: last column "quality" is the target ----------
df = pd.read_csv(os.environ.get("WINE_CSV", "winequality-white.csv"), sep=None, engine="python")
X, y = df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy(dtype=float)

# 70 / 10 / 20: first take 20% test, then 0.1 / 0.8 = 12.5% of the rest as validation
X_rest, X_ts, y_rest, y_ts = train_test_split(X, y, test_size=0.20, random_state=SEED)
X_tr, X_val, y_tr, y_val = train_test_split(X_rest, y_rest, test_size=0.125, random_state=SEED)
print(f"train {len(X_tr)}, validation {len(X_val)}, test {len(X_ts)}")

# ---------- grids ----------
grid_rf = list(ParameterGrid({
    "n_estimators": [100, 300],
    "max_depth": [None, 5, 10, 20, 30],
    "min_samples_leaf": [1, 2, 4, 8, 16],
    "max_features": [0.2, 0.4, 0.6, 0.8, 1.0],
    "bootstrap": [True, False],
}))                                                   # 2 * 5 * 5 * 5 * 2 = 500
grid_mlp = list(ParameterGrid({
    "hidden_layer_sizes": [(16,), (32,), (64,), (128,), (64, 32), (128, 64)],
    "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
    "learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3],
    "batch_size": [32, 64, 128, 256],
}))                                                   # 6 * 6 * 4 * 4 = 576

if os.environ.get("QUICK"):                           # small grids for a smoke test
    grid_rf, grid_mlp = grid_rf[:6], grid_mlp[:6]
for name, g in [("RF", grid_rf), ("MLP", grid_mlp)]:
    print(f"{name} grid size {len(g)} (validation points {len(X_val)})")

def make_rf(p):
    return RandomForestRegressor(**p, random_state=SEED, n_jobs=1)

def make_mlp(p):
    return make_pipeline(StandardScaler(),
                         MLPRegressor(**p, max_iter=500, early_stopping=True, random_state=SEED))

def fit_and_score(make_model, params):
    model = make_model(params).fit(X_tr, y_tr)                  # train set only
    return mean_squared_error(y_val, model.predict(X_val))      # validation set only

def search(make_model, grid):
    scores = Parallel(n_jobs=-1)(delayed(fit_and_score)(make_model, p) for p in grid)
    best_mse, best_params = np.inf, None
    for params, val_mse in zip(grid, scores):
        if val_mse < best_mse:                                  # MSE: smaller is better
            best_mse, best_params = val_mse, params             # update both
    best_model = make_model(best_params).fit(X_tr, y_tr)        # same seed, same model
    return best_mse, best_params, best_model, np.array(scores)

results = {}
for name, make_model, grid in [("RandomForest", make_rf, grid_rf), ("MLP", make_mlp, grid_mlp)]:
    val_mse, params, model, scores = search(make_model, grid)
    test_mse = mean_squared_error(y_ts, model.predict(X_ts))   # test set used once
    results[name] = {"best val MSE": val_mse, "test MSE": test_mse,
                     "median val MSE over grid": float(np.median(scores))}
    print(f"\n{name} best params: {params}")

baseline = mean_squared_error(y_ts, np.full_like(y_ts, y_tr.mean()))
print("\n", pd.DataFrame(results).T.round(4))
print(f"\nmean baseline test MSE: {baseline:.4f}")