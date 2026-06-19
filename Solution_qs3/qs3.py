

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Tuple

from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from sklearn.ensemble import GradientBoostingClassifier

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except ImportError:
    USE_XGB = False

# Config ──────────────────────────────
DATA_PATH = "trade_data.csv"
HORIZONS  = [5, 10, 15, 20, 25, 30]
FEATURES  = ["Side", "Volume", "Trade Price", "M0", "Spread", "Minute", "ClientCode"]

# Task 5 quoting defaults (overwritten after calibration)
DEFAULT_K1, DEFAULT_K2, DEFAULT_K3, DEFAULT_K4 = 1.0, 0.002, 0.005, 0.002
SCALE = 1000.0
LAM, GAMMA, PHI = 0.8, 1.0, 1e-4


# Data loading ────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"]       = pd.to_datetime(df["Date"])
    t                = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce")
    df["Minute"]     = t.dt.hour * 60 + t.dt.minute
    df["ClientCode"] = df["Name"].astype("category").cat.codes
    for h in HORIZONS:
        df[f"PnL_{h}"] = df["Side"] * df["Volume"] * (df[f"M{h}"] - df["Trade Price"])
        df[f"Y_{h}"]   = (df[f"PnL_{h}"] < 0).astype(int)
    df["AggregatePnL"] = df[[f"PnL_{h}" for h in HORIZONS]].mean(axis=1)
    return df.sort_values(["Date", "time"]).reset_index(drop=True)


def time_split(df: pd.DataFrame):
    n = len(df)
    return df.iloc[:int(.6*n)], df.iloc[int(.6*n):int(.8*n)], df.iloc[int(.8*n):]


# Task 1 : Adversity Profile ──────────
def adversity_profile(client: str, tau: List[int], df: pd.DataFrame = None) -> List[float]:
    """Adversity percentage (0-100) per horizon for given client."""
    c = df[df["Name"] == client]
    return [100.0 * c[f"Y_{h}"].mean() for h in tau]


def run_task1(df: pd.DataFrame) -> pd.DataFrame:
    clients = sorted(df["Name"].unique())
    task1 = pd.DataFrame(
        [[c] + adversity_profile(c, HORIZONS, df) for c in clients],
        columns=["client"] + [f"tau_{h}" for h in HORIZONS]
    )
    task1.to_csv("task1_results.csv", index=False)
    print("[Task 1] task1_results.csv saved.")
    print(task1.to_string(index=False))
    return task1


# Task 2 : Client Profitability & Spread Recommendation ────────────────────
def expected_pnl(client: str, tau: List[int], df: pd.DataFrame = None) -> dict:
    """Expected PnL per trade at each horizon and aggregate."""
    c   = df[df["Name"] == client]
    pnl = [c[f"PnL_{h}"].mean() for h in tau]
    return {"per_horizon": pnl, "aggregate": float(np.mean(pnl))}


def classify_client(client: str, df: pd.DataFrame = None) -> str:
    """Returns 'profitable' or 'costly' based on aggregate expected PnL."""
    return "profitable" if expected_pnl(client, HORIZONS, df)["aggregate"] >= 0 else "costly"


def min_half_spread(client: str, df: pd.DataFrame = None) -> float:
    """Minimum half-spread delta* such that expected aggregate PnL >= 0."""
    c       = df[df["Name"] == client]
    agg_mid = np.mean([
        (c["Side"] * c["Volume"] * (c[f"M{h}"] - c["M0"])).mean()
        for h in HORIZONS
    ])
    avg_vol = c["Volume"].mean()
    return float(max(0.0, -agg_mid / avg_vol)) if avg_vol > 0 else 0.0


def run_task2(df: pd.DataFrame) -> pd.DataFrame:
    clients = sorted(df["Name"].unique())
    rows = []
    for c in clients:
        ep  = expected_pnl(c, HORIZONS, df)
        mhs = min_half_spread(c, df)
        rows.append([c] + ep["per_horizon"] + [ep["aggregate"], mhs])
    task2 = pd.DataFrame(rows,
        columns=["client"] + [f"tau_{h}" for h in HORIZONS] + ["agg_pnl", "delta_star"])
    task2.to_csv("task2_results.csv", index=False)
    print("[Task 2] task2_results.csv saved.")
    print(task2.to_string(index=False))
    for c in clients:
        cls = classify_client(c, df)
        agg = expected_pnl(c, HORIZONS, df)["aggregate"]
        print(f"  Client {c}: {cls.upper()}  (agg_pnl={agg:.4f})")
    return task2


# Task 3 : Adversity Prediction Model 
def build_model():
    if USE_XGB:
        return XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                             eval_metric="logloss", use_label_encoder=False, verbosity=0)
    return GradientBoostingClassifier(n_estimators=50, max_depth=4,
                                      learning_rate=0.1, subsample=0.8, random_state=42)


def train_models(train: pd.DataFrame) -> dict:
    models = {}
    for h in HORIZONS:
        m = build_model()
        m.fit(train[FEATURES], train[f"Y_{h}"])
        models[h] = m
        print(f"  [Task 3] horizon {h}s trained")
    return models


def predict_adversity(trade: pd.DataFrame, tau: int, models: dict) -> float:
    """Probability in [0,1] that the trade is adverse at horizon tau."""
    return float(models[tau].predict_proba(trade[FEATURES])[:, 1][0])


def compute_metrics(train, valid, test, models) -> pd.DataFrame:
    rows = []
    for split_df, name in [(train, "train"), (valid, "validation"), (test, "test")]:
        accs, precs, recs, lls = [], [], [], []
        for h in HORIZONS:
            y    = split_df[f"Y_{h}"]
            p    = models[h].predict_proba(split_df[FEATURES])[:, 1]
            pred = (p > .5).astype(int)
            accs.append(accuracy_score(y, pred))
            precs.append(precision_score(y, pred, zero_division=0))
            recs.append(recall_score(y, pred, zero_division=0))
            lls.append(log_loss(y, p))
        rows.append([name, np.mean(accs), np.mean(precs), np.mean(recs), np.mean(lls)])
    return pd.DataFrame(rows, columns=["split", "accuracy", "precision", "recall", "log_loss"])


def run_task3(df: pd.DataFrame):
    train, valid, test = time_split(df)
    print("[Task 3] Training models...")
    models  = train_models(train)
    metrics = compute_metrics(train, valid, test, models)
    metrics.to_csv("task3_results.csv", index=False)
    print("[Task 3] task3_results.csv saved.")
    print(metrics.to_string(index=False))
    return models, train, valid, test


# Task 4 : Optimal Externalization Threshold ───────────────────────────────
def plot_pnl_vs_theta(valid, models, path="pnl_vs_theta.png"):
    alpha = np.mean([models[h].predict_proba(valid[FEATURES])[:, 1] for h in HORIZONS], axis=0)
    rows  = []
    for theta in np.arange(0, 1.01, 0.01):
        keep = alpha <= theta
        rows.append([round(theta, 2), valid["AggregatePnL"].values[keep].sum(), 1 - keep.mean()])
    curve = pd.DataFrame(rows, columns=["theta", "pnl", "ext_rate"])
    best  = curve.loc[curve["pnl"].idxmax()]
    plt.figure(figsize=(8, 5))
    plt.plot(curve["theta"], curve["pnl"], color="steelblue", lw=2, label="Validation PnL")
    plt.axvline(best["theta"], color="red", ls="--", lw=1.5, label=f"θ*={best['theta']:.2f}")
    plt.xlabel("Threshold θ"); plt.ylabel("Validation PnL")
    plt.title("PnL vs Externalization Threshold"); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"[Task 4] {path} saved.")


def optimal_threshold(valid, test, models, client=None, tau=None) -> dict:
    def get_probs_pnl(split, c, h):
        mask = split["Name"] == c if c else slice(None)
        s    = split[mask] if c else split
        p    = models[h].predict_proba(s[FEATURES])[:, 1] if h else \
               np.mean([models[hh].predict_proba(s[FEATURES])[:, 1] for hh in HORIZONS], axis=0)
        pnl  = s[f"PnL_{h}"].values if h else s["AggregatePnL"].values
        return p, pnl
    pv, pnl_v = get_probs_pnl(valid, client, tau)
    best_theta, best_pnl = 0.5, -1e18
    for theta in np.arange(0, 1.01, 0.05):
        v = pnl_v[pv <= theta].sum()
        if v > best_pnl: best_pnl = v; best_theta = theta
    pt, pnl_t = get_probs_pnl(test, client, tau)
    return {"theta": best_theta, "validation_pnl": best_pnl,
            "test_pnl": float(pnl_t[pt <= best_theta].sum())}


def run_task4(df, models, valid, test) -> pd.DataFrame:
    plot_pnl_vs_theta(valid, models)
    clients = sorted(df["Name"].unique())
    rows = []
    for c in clients:
        for h in HORIZONS:
            res = optimal_threshold(valid, test, models, client=c, tau=h)
            rows.append([c, h, res["theta"], round(res["test_pnl"], 4)])
    task4 = pd.DataFrame(rows, columns=["client", "tau", "theta_star", "final_pnl"])
    task4.to_csv("task4_results.csv", index=False)
    print("[Task 4] task4_results.csv saved.")
    print(task4.to_string(index=False))
    return task4


# Task 5 : Dynamic Quoting Under Inventory Pressure ────────────────────────
def quote(inventory: float, sigma: float, alpha: float, eta: float,
          k1=DEFAULT_K1, k2=DEFAULT_K2, k3=DEFAULT_K3, k4=DEFAULT_K4) -> Tuple[float, float]:
    """
    Q(I, sigma, alpha, eta) -> (delta_bid, delta_ask)
    Constraints: 0.5*sigma <= delta <= 0.005
    """
    sigma = max(sigma, 1e-8)
    base    = k1 * sigma
    adv     = k3 * alpha
    inv_adj = k2 * np.tanh(inventory / SCALE)
    t_adj   = k4 * eta
    lo, hi  = 0.5 * sigma, 0.005
    delta_bid = float(np.clip(base + adv - inv_adj + t_adj, lo, hi))
    delta_ask = float(np.clip(base + adv + inv_adj + t_adj, lo, hi))
    return delta_bid, delta_ask


def validate_quote(delta_bid: float, delta_ask: float, sigma: float) -> bool:
    lo, hi = 0.5 * sigma, 0.005
    return delta_bid >= lo and delta_ask >= lo and delta_bid <= hi and delta_ask <= hi


def fill_probability(delta: float, sigma: float, lam=LAM, gamma=GAMMA) -> float:
    return lam * np.exp(-gamma * delta / max(sigma, 1e-8))


def inventory_penalty(end_inv: float, avg_sigma: float, phi=PHI) -> float:
    return phi * end_inv**2 * avg_sigma


def realized_vol(mid: np.ndarray, n: int = 20) -> np.ndarray:
    rets  = np.diff(mid) / np.maximum(mid[:-1], 1e-8)
    sigma = np.zeros(len(mid))
    for i in range(len(mid)):
        w = rets[max(0, i-n):i]
        sigma[i] = np.sqrt(np.mean(w**2)) if len(w) > 0 else 1e-4
    return sigma


def vectorized_backtest(df: pd.DataFrame, models: dict,
                        k1=DEFAULT_K1, k2=DEFAULT_K2,
                        k3=DEFAULT_K3, k4=DEFAULT_K4) -> dict:
    """Fast vectorized backtest using numpy; stochastic fill with fixed seed."""
    np.random.seed(42)
    df = df.reset_index(drop=True)
    mid      = df["M0"].values
    sigma_v  = realized_vol(mid)
    t        = pd.to_datetime(df["time"], format="%H:%M:%S", errors="coerce")
    mins     = t.dt.hour * 60 + t.dt.minute
    span     = max(mins.max() - mins.min(), 1)
    eta_v    = ((mins - mins.min()) / span).values
    alpha_v  = np.mean([models[h].predict_proba(df[FEATURES])[:, 1] for h in HORIZONS], axis=0)
    sides    = df["Side"].values
    vols     = df["Volume"].values
    pnl_cols = np.stack([df[f"PnL_{h}"].values for h in HORIZONS], axis=1)  # (N,6)

    daily_pnls = []
    dates      = df["Date"].values

    for date in np.unique(dates):
        idx = np.where(dates == date)[0]
        inv = 0.0; day_pnl = 0.0
        sigs_day = sigma_v[idx]

        for i in idx:
            sig = max(sigma_v[i], 1e-6)
            db, da = quote(inv, sig, alpha_v[i], eta_v[i], k1, k2, k3, k4)
            ds     = db if sides[i] == 1 else da
            if np.random.random() < fill_probability(ds, sig):
                day_pnl += float(np.mean(pnl_cols[i]))
                inv     += sides[i] * vols[i]

        pen = inventory_penalty(inv, float(np.mean(sigs_day)))
        daily_pnls.append(day_pnl - pen)

    total      = float(np.sum(daily_pnls))
    sigma_port = float(np.std(daily_pnls)) if len(daily_pnls) > 1 else 1.0
    score      = total / max(sigma_port, 1.0)
    return {"total_pnl": total, "score": score, "daily_pnls": daily_pnls}


def calibrate_params(valid: pd.DataFrame, models: dict):
    best_score, best = -1e18, (DEFAULT_K1, DEFAULT_K2, DEFAULT_K3, DEFAULT_K4)
    print("[Task 5] Calibrating on validation set...")
    for k1 in [0.5, 1.0, 1.5]:
        for k2 in [0.001, 0.002, 0.005]:
            for k3 in [0.002, 0.005, 0.01]:
                for k4 in [0.001, 0.002, 0.005]:
                    res = vectorized_backtest(valid, models, k1, k2, k3, k4)
                    if res["score"] > best_score:
                        best_score = res["score"]; best = (k1, k2, k3, k4)
    print(f"  Best: k1={best[0]}, k2={best[1]}, k3={best[2]}, k4={best[3]}  score={best_score:.4f}")
    return best


def run_task5(df, models, valid, test):
    k1, k2, k3, k4 = calibrate_params(valid, models)
    print("[Task 5] Backtesting on test set...")
    res = vectorized_backtest(test, models, k1, k2, k3, k4)
    print(f"  Total PnL: {res['total_pnl']:,.2f}  |  Sharpe: {res['score']:.4f}")
    pd.DataFrame([{"k1":k1,"k2":k2,"k3":k3,"k4":k4,
                   "test_total_pnl":res["total_pnl"],"sharpe_score":res["score"]}]
    ).to_csv("task5_results.csv", index=False)
    print("[Task 5] task5_results.csv saved.")

    plt.figure(figsize=(10, 4))
    plt.plot(res["daily_pnls"], lw=1.2, color="steelblue")
    plt.axhline(0, color="red", lw=1, ls="--")
    plt.xlabel("Trading Day"); plt.ylabel("Net PnL")
    plt.title("Task 5 — Daily Net PnL (test set)")
    plt.grid(True, alpha=0.4); plt.tight_layout()
    plt.savefig("task5_daily_pnl.png", dpi=150, bbox_inches="tight"); plt.close()
    print("[Task 5] task5_daily_pnl.png saved.")


# Main
def main():
    print("=" * 60)
    print(" Nomura Quant Challenge 5")
    print("=" * 60)
    df = load_data()
    print(f"Loaded {len(df)} rows | Clients: {sorted(df['Name'].unique())}")

    print("\nTASK 1 ──")
    run_task1(df)

    print("\nTASK 2 ──")
    run_task2(df)

    print("\nTASK 3 ──")
    models, train, valid, test = run_task3(df)

    print("\nTASK 4 ──")
    run_task4(df, models, valid, test)

    print("\nTASK 5 ──")
    run_task5(df, models, valid, test)

    print("\nDone. Output files:")
    for f in ["task1_results.csv","task2_results.csv","task3_results.csv",
              "task4_results.csv","pnl_vs_theta.png",
              "task5_results.csv","task5_daily_pnl.png"]:
        print(f"  {f}")


if __name__ == "__main__":
    main()