"""
Core / Non-Core split of deposits with quantile XGBoost (ALM / Liquidity)
=========================================================================
One independent model for each series ("Savings at sight" and "Savings other").

Target variable
---------------
For each date t and horizon h (business days), the retention ratio is defined as

    y_h(t) = min( B_{t+1}, ..., B_{t+h} ) / B_t

= share of today's balance that was still on the books at the WORST point of the
next h days. This is the natural definition for liquidity: it is not enough for
the balance to come back by 12 months; it must have stayed for the whole period.

Model
-----
For each series and each horizon h ∈ {1d, 1w, 1m, 3m, 6m, 12m}, a regularized XGBRegressor
with quantile loss (reg:quantileerror, pinball) estimates the quantiles
α ∈ {1%, 5%, 10%, 50%} of y_h simultaneously, conditional on variables observable
at t (volatility, drawdown, momentum, seasonality). Overfitting is controlled with
depth-2 trees, leaves of at least one full horizon, early stopping, monotone constraints
with an economic sign and a reduced feature set (see "Overfitting control" below).

Quantile XGBoost tends to be poorly calibrated in extreme tails (little data at
1%). A sequential conformal calibration is therefore applied (CQR adapted to time
series): the predicted quantile is shifted by the (1-α) quantile of out-of-sample
errors already realized, so the exception rate converges to α.

    Core_t     = B_t · min(1, q̃_α(y_12m | X_t))     (α = 1% → 99% confidence)
    Non-core_t = B_t − Core_t

The curve q̃_α(y_h) by horizon (forced monotone) allocates the non-core to
liquidity buckets: outflow in (h_{i-1}, h_i] = B_t · (q̃(h_{i-1}) − q̃(h_i)).

Validation: annual walk-forward with purging (no training label overlaps the test
period) and comparison against historical simulation (HS) calibrated with the same
procedure.

Usage:
    python core_noncore_model.py --input data/savings_deposits_indexed.csv --out results
"""
import argparse
import html
import os
import re
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------
SERIES = {"sight": "Savings at sight", "other": "Savings other"}
HORIZONS = [1, 5, 21, 63, 126, 252]                      # business days
H_LABEL = {1: "1d", 5: "1w", 21: "1m", 63: "3m", 126: "6m", 252: "12m"}
QUANTILES = [0.01, 0.05, 0.10, 0.50]
CORE_H = 252                                             # horizon that defines the core
CORE_ALPHAS = [0.01, 0.05]                               # 99% and 95% confidence
TEST_YEARS = list(range(2019, 2026))                     # walk-forward test years
MIN_CAL = 120                                            # minimum OOS errors needed to calibrate
SPIKE_THRESHOLD = 0.20                                   # |log-return| that flags an isolated spike
SEED = 42

# --- Overfitting control ---------------------------------------------------------------
# 1) Shallow trees and a single fixed configuration (no hyperparameter search on the same
#    period used for evaluation -> no selection bias).
# 2) min_child_weight scaled to the horizon: with quantile loss the hessian is 1 per
#    observation, so each leaf needs >= h observations, i.e. at least one full independent
#    window (12-month labels overlap on 251 of 252 days).
# 3) Early stopping on a validation block at the end of the training set, purged by h days.
# 4) Monotone constraints with an economic sign, and a reduced feature set.
# 5) More randomization (subsample / colsample) and a larger L2 penalty.
PARAM_GRID = [
    dict(max_depth=2, n_estimators=600, learning_rate=0.03),
]
MCW_MIN = 40                                             # min_child_weight = max(MCW_MIN, h)
VAL_FRAC = 0.20                                          # validation block for early stopping
EARLY_STOP = 40
BASE_PARAMS = dict(
    objective="reg:quantileerror",
    quantile_alpha=np.array(QUANTILES),
    subsample=0.7,
    colsample_bytree=0.6,
    reg_lambda=5.0,
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
)

# Features used (redundant ones and high-cardinality calendar variables are excluded:
# day of month, day of week, days to year end -> they only added noise to memorize).
FEATURES = ["ret_5", "ret_21", "ret_63", "vol_21", "vol_63", "vol_252", "vol_ewma",
            "dev_ma_63", "dev_ma_252", "dd_252", "up_252", "worst_1d_63", "worst_21d_252",
            "slope_63", "month", "month_end", "quarter_end"]
# Economic sign on retention (-1: the higher the value, the lower the expected retention).
MONOTONE = {
    "vol_21": -1, "vol_63": -1, "vol_252": -1, "vol_ewma": -1,   # more volatility, less core
    "up_252": -1,       # balance far above its 1-year low = "hot" money
    "dev_ma_252": -1,   # balance above its 1-year average = transitory excess
    "dd_252": -1,       # close to the 1-year high = more balance that can leave
    "worst_1d_63": 1, "worst_21d_252": 1,                        # milder recent worst falls
}


# --------------------------------------------------------------------------------------
# 1. Loading and cleaning
# --------------------------------------------------------------------------------------
def load_data(path):
    """Reads the table (exported HTML, CSV or Excel) with columns date, sight, other."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".html", ".htm"):
        s = open(path, encoding="utf-8", errors="ignore").read()
        rows = []
        for r in re.findall(r"<tr[^>]*>(.*?)</tr>", s, flags=re.S):
            c = [html.unescape(re.sub(r"<[^>]+>", "", x)).strip()
                 for x in re.findall(r"<td[^>]*>(.*?)</td>", r, flags=re.S)]
            if len(c) == 3:
                rows.append(c)
        df = pd.DataFrame(rows, columns=["date", "sight", "other"])
    elif ext == ".csv":
        df = pd.read_csv(path)
        df.columns = ["date", "sight", "other"]
    else:
        df = pd.read_excel(path)
        df.columns = ["date", "sight", "other"]
    # HTML export uses dd/mm/yyyy; CSV/Excel inputs are expected in ISO format (yyyy-mm-dd)
    df["date"] = pd.to_datetime(df["date"], dayfirst=ext in (".html", ".htm"))
    for k in SERIES:
        df[k] = pd.to_numeric(df[k].astype(str).str.replace(",", ""), errors="coerce")
    return df


def clean_data(df):
    """Fixes duplicated dates and isolated one-day spikes. Returns (clean_df, log)."""
    log = []
    df = df.copy()
    descending = df["date"].iloc[0] > df["date"].iloc[-1]

    for d, g in df[df["date"].duplicated(keep=False)].groupby("date"):
        if (g[list(SERIES)].round(2).nunique() == 1).all():
            df = df.drop(g.index[1:])
            log.append(dict(date=d.date(), series="both", action="Identical duplicate removed"))
            continue
        older = g.index[-1] if descending else g.index[0]
        prev_bd = d - pd.offsets.BDay(1)
        if prev_bd not in set(df["date"]):
            df.loc[older, "date"] = prev_bd
            log.append(dict(date=d.date(), series="both",
                            action=f"Duplicated date with different values: the earlier record is reassigned to {prev_bd.date()} (missing)"))
        else:
            df.loc[g.index[0], list(SERIES)] = g[list(SERIES)].mean().values
            df = df.drop(g.index[1:])
            log.append(dict(date=d.date(), series="both", action="Duplicated date with different values: averaged"))

    df = df.sort_values("date").reset_index(drop=True)

    # isolated spike: jump > threshold that almost fully reverses the next day
    for k in SERIES:
        s = df[k].values.copy()
        for i in range(1, len(s) - 1):
            r1, r2 = np.log(s[i] / s[i - 1]), np.log(s[i + 1] / s[i])
            if abs(r1) > SPIKE_THRESHOLD and np.sign(r1) != np.sign(r2) and abs(r1 + r2) < 0.25 * abs(r1):
                new = np.sqrt(s[i - 1] * s[i + 1])
                log.append(dict(date=df["date"][i].date(), series=SERIES[k],
                                action=f"Isolated spike ({r1:+.1%}, reverses {r2:+.1%} the next day) -> geometric mean of neighbours",
                                original_value=s[i], new_value=new))
                s[i] = new
        df[k] = s

    for k in SERIES:
        n = int((df[k].diff() == 0).sum())
        log.append(dict(date=None, series=SERIES[k],
                        action=f"{n} days with a balance identical to the previous day (possible carry-forward; not modified)"))
    return df, pd.DataFrame(log)


# --------------------------------------------------------------------------------------
# 2. Features and labels
# --------------------------------------------------------------------------------------
def build_features(s):
    """Variables observable at t (no future information)."""
    ls = np.log(s)
    r = ls.diff()
    X = pd.DataFrame(index=s.index)
    for n in (1, 5, 21, 63, 126, 252):
        X[f"ret_{n}"] = ls - ls.shift(n)
    for n in (21, 63, 252):
        X[f"vol_{n}"] = r.rolling(n).std()
    X["vol_ewma"] = np.sqrt((r ** 2).ewm(alpha=0.06).mean())
    X["vol_ratio"] = X["vol_21"] / X["vol_252"]
    for n in (21, 63, 252):
        X[f"dev_ma_{n}"] = ls - ls.rolling(n).mean()
    X["dd_252"] = ls - ls.rolling(252).max()                  # drawdown vs 1-year high
    X["up_252"] = ls - ls.rolling(252).min()                  # distance to 1-year low
    X["worst_1d_63"] = r.rolling(63).min()
    X["worst_21d_252"] = (ls - ls.shift(21)).rolling(252).min()
    X["skew_63"] = r.rolling(63).skew()
    t = np.arange(63)
    X["slope_63"] = ls.rolling(63).apply(lambda v: np.polyfit(t, v, 1)[0], raw=True)
    idx = s.index
    X["month"] = idx.month
    X["dom"] = idx.day
    X["dow"] = idx.dayofweek
    X["month_end"] = (idx + pd.offsets.BDay(1)).month != idx.month
    X["quarter_end"] = X["month_end"] & idx.month.isin([3, 6, 9, 12])
    X["days_to_ye"] = (pd.to_datetime(idx.year.astype(str) + "-12-31") - idx).days
    return X.astype(float)


def build_targets(s):
    """y_h(t) = min(B_{t+1..t+h}) / B_t."""
    Y = pd.DataFrame(index=s.index)
    fwd = s.shift(-1)
    for h in HORIZONS:
        Y[h] = fwd[::-1].rolling(h, min_periods=h).min()[::-1] / s
    return Y


def qcols(prefix):
    return [f"{prefix}{a}" for a in QUANTILES]


def _model(params, h, n_estimators=None):
    p = dict(params)
    if n_estimators is not None:
        p["n_estimators"] = n_estimators
    mono = tuple(MONOTONE.get(f, 0) for f in FEATURES)
    return xgb.XGBRegressor(**BASE_PARAMS, **p, min_child_weight=max(MCW_MIN, h),
                            monotone_constraints=mono)


def fit_predict(params, Xtr, ytr, Xte, h):
    """Trains with early stopping on a purged final block, then refits on the full training
    set with the optimal number of trees. Returns (model, sorted predictions, n_trees)."""
    Xtr, Xte = Xtr[FEATURES], Xte[FEATURES]
    n = len(Xtr)
    cut = int(n * (1 - VAL_FRAC))
    n_trees = params["n_estimators"]
    if cut - h >= 300 and n - cut >= 100:
        es = _model(params, h)
        es.set_params(early_stopping_rounds=EARLY_STOP)
        es.fit(Xtr.iloc[:cut - h], ytr.iloc[:cut - h],
               eval_set=[(Xtr.iloc[cut:], ytr.iloc[cut:])], verbose=False)
        n_trees = max(20, int((es.best_iteration + 1) * 1.1))
    m = _model(params, h, n_trees)
    m.fit(Xtr, ytr)
    P = np.sort(np.asarray(m.predict(Xte)).reshape(len(Xte), len(QUANTILES)), axis=1)  # no quantile crossing
    return m, P, n_trees


def pinball(y, q, a):
    d = y - q
    return np.mean(np.maximum(a * d, (a - 1) * d))


def kupiec(hits, n, a):
    x = int(hits)
    if n == 0:
        return np.nan
    p = x / n
    ll1 = 0.0 if x in (0, n) else x * np.log(p) + (n - x) * np.log(1 - p)
    ll0 = x * np.log(a) + (n - x) * np.log(1 - a)
    return 1 - stats.chi2.cdf(-2 * (ll0 - ll1), 1)


# --------------------------------------------------------------------------------------
# 3. Walk-forward + sequential conformal calibration
# --------------------------------------------------------------------------------------
def walk_forward(X, Y, params):
    dates = X.index
    pos = pd.Series(np.arange(len(X)), index=dates)
    feat_ok = X.notna().all(axis=1)
    out = []
    for yr in TEST_YEARS:
        first_test = pos[dates.year == yr].min()
        for h in HORIZONS:
            lab = feat_ok & Y[h].notna()
            tr = lab & (pos + h < first_test)                  # purge overlapping labels
            te = lab & (dates.year == yr)
            if te.sum() == 0 or tr.sum() < 200:
                continue
            _, P, nt = fit_predict(params, X[tr], Y.loc[tr, h], X[te], h)
            o = pd.DataFrame(P, index=dates[te], columns=qcols("q"))
            o["y"], o["h"], o["pos"], o["fold"], o["first_test"] = Y.loc[te, h].values, h, pos[te].values, yr, first_test
            o["n_trees"] = nt
            for a in QUANTILES:                                # benchmark: historical simulation
                o[f"hs{a}"] = Y.loc[tr, h].quantile(a)
            out.append(o)
    return pd.concat(out)


def conf_offset(err, a):
    """ONE-SIDED conformal shift (prudential criterion): it can only lower the quantile
    (more conservative), never raise it. If the calibration period was calm, the stress
    episodes the model learned in training are not 'forgotten'."""
    return max(0.0, float(np.quantile(err, 1 - a)))


def conformal(P):
    """Sequential adjustment: in each fold, the quantile is shifted by the (1-α) quantile of the
    errors (q - y) from earlier folds whose labels were already realized before the fold starts."""
    P = P.copy()
    for pre in ("q", "hs"):
        for a in QUANTILES:
            P[f"{pre}c{a}"] = np.nan
    for (yr, h), g in P.groupby(["fold", "h"]):
        prior = P[(P["h"] == h) & (P["fold"] < yr) & (P["pos"] + h < g["first_test"].iloc[0])]
        for pre in ("q", "hs"):
            for a in QUANTILES:
                c = 0.0
                if len(prior) >= MIN_CAL:
                    c = conf_offset(prior[f"{pre}{a}"] - prior["y"], a)
                P.loc[(P["fold"] == yr) & (P["h"] == h), f"{pre}c{a}"] = g[f"{pre}{a}"].values - c
    return P


def final_offsets(P):
    """Conformal shift for the production model: all available OOS errors."""
    off = {}
    for h in HORIZONS:
        g = P[P["h"] == h]
        for a in QUANTILES:
            off[(h, a)] = conf_offset(g[f"q{a}"] - g["y"], a)
    return off


def evaluate(P):
    rows = []
    for h in HORIZONS:
        g = P[(P["h"] == h) & P[f"qc{QUANTILES[0]}"].notna()]
        n = len(g)
        for a in QUANTILES:
            r = dict(horizon=H_LABEL[h], alpha=a, n_obs=n, period=f"{g['fold'].min()}-{g['fold'].max()}")
            for lbl, col in (("xgb_raw", f"q{a}"), ("xgb_calibrated", f"qc{a}"),
                             ("hs_raw", f"hs{a}"), ("hs_calibrated", f"hsc{a}")):
                r[f"exc_{lbl}"] = (g["y"] < g[col]).mean()
                r[f"pinball_{lbl}"] = pinball(g["y"].values, g[col].values, a)
            r["kupiec_p_xgb_cal"] = kupiec((g["y"] < g[f"qc{a}"]).sum(), n, a)
            r["skill_xgb_vs_hs (calibrated)"] = 1 - r["pinball_xgb_calibrated"] / r["pinball_hs_calibrated"]
            rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# 4. Per-series pipeline
# --------------------------------------------------------------------------------------
def run_series(df, key, outdir):
    name = SERIES[key]
    s = df.set_index("date")[key]
    X, Y = build_features(s), build_targets(s)
    dates = s.index

    # --- hyperparameter selection: normalized OOS pinball (raw), averaged over horizons/quantiles
    grid_res, best, best_score, best_P = [], None, np.inf, None
    for params in PARAM_GRID:
        P = walk_forward(X, Y, params)
        score = np.mean([pinball(g["y"].values, g[f"q{a}"].values, a) / a
                         for _, g in P.groupby("h") for a in QUANTILES])
        grid_res.append(dict(**params, score_pinball_norm=score))
        print(f"  [{name}] {params} -> {score:.5f}")
        if score < best_score:
            best, best_score, best_P = params, score, P
    P = conformal(best_P)
    metrics = evaluate(P)
    off = final_offsets(P)

    # --- production models (one per horizon) on the full labelled history
    feat_ok = X.notna().all(axis=1)
    Xall = X[feat_ok]
    curves = {}
    imps = []
    for h in HORIZONS:
        tr = feat_ok & Y[h].notna()
        m, Pq, nt = fit_predict(best, X[tr], Y.loc[tr, h], Xall, h)
        print(f"  [{name}] production model {H_LABEL[h]}: {nt} trees, min_child_weight={max(MCW_MIN, h)}")
        for j, a in enumerate(QUANTILES):
            curves[(h, a)] = pd.Series(Pq[:, j] - off[(h, a)], index=Xall.index)
        m.save_model(os.path.join(outdir, f"xgb_{key}_{H_LABEL[h]}.json"))
        g = pd.Series(m.get_booster().get_score(importance_type="gain"))
        imps.append((g / g.sum()).rename(H_LABEL[h]))
    imp = pd.concat(imps, axis=1).fillna(0)
    imp["average"] = imp.mean(axis=1)
    imp = imp.sort_values("average", ascending=False).reset_index().rename(columns={"index": "feature"})

    # monotone curve bounded in [0, 1]
    def curve_frame(a, src):
        C = pd.DataFrame({h: src[(h, a)] for h in HORIZONS}).clip(0, 1)
        return C.cummin(axis=1)

    daily = pd.DataFrame({"balance": s})
    for a in CORE_ALPHAS:
        C = curve_frame(a, curves)
        daily[f"core_pct_{a}"] = C[CORE_H]
        daily[f"core_{a}"] = s * C[CORE_H]
        daily[f"noncore_{a}"] = s - daily[f"core_{a}"]
        # strictly out-of-sample series (calibrated walk-forward)
        oos = {}
        for h in HORIZONS:
            g = P[P["h"] == h][f"qc{a}"]
            oos[(h, a)] = g[~g.index.duplicated()]
        Co = pd.DataFrame({h: oos[(h, a)] for h in HORIZONS}).dropna().clip(0, 1).cummin(axis=1)
        daily[f"core_pct_oos_{a}"] = Co[CORE_H]
        daily[f"core_oos_{a}"] = s * Co[CORE_H]
    daily["actual_retention_12m"] = Y[CORE_H]

    # --- liquidity buckets at the latest date
    last, B = dates[-1], s.iloc[-1]
    buckets = []
    for a in CORE_ALPHAS:
        q = curve_frame(a, curves).loc[last]
        prev, prev_lbl = 1.0, "0"
        for h in HORIZONS:
            buckets.append(dict(series=name, confidence=f"{1-a:.0%}", bucket=f"{prev_lbl} - {H_LABEL[h]}",
                                outflow_pct=prev - q[h], outflow_amount=B * (prev - q[h]),
                                cum_retention_pct=q[h]))
            prev, prev_lbl = q[h], H_LABEL[h]
        buckets.append(dict(series=name, confidence=f"{1-a:.0%}", bucket="> 12m (CORE)",
                            outflow_pct=q[CORE_H], outflow_amount=B * q[CORE_H], cum_retention_pct=np.nan))
    buckets = pd.DataFrame(buckets)

    summary = dict(series=name, date=last.date(), balance=B, hyperparameters=str(best))
    for a in CORE_ALPHAS:
        c = f"{1-a:.0%}"
        summary[f"core_{c}"] = daily[f"core_{a}"].iloc[-1]
        summary[f"core_pct_{c}"] = daily[f"core_pct_{a}"].iloc[-1]
        summary[f"noncore_{c}"] = daily[f"noncore_{a}"].iloc[-1]
        summary[f"core_pct_HS_{c} (unconditional benchmark)"] = min(1, Y[CORE_H].quantile(a))
    summary["offset_conformal_12m_1%"] = off[(CORE_H, 0.01)]
    summary["offset_conformal_12m_5%"] = off[(CORE_H, 0.05)]

    plot_series(daily, buckets, metrics, name, key, outdir)
    return dict(summary=summary, daily=daily, buckets=buckets, metrics=metrics,
                grid=pd.DataFrame(grid_res), importance=imp)


# --------------------------------------------------------------------------------------
# 5. Charts
# --------------------------------------------------------------------------------------
def plot_series(daily, buckets, metrics, name, key, outdir):
    a = CORE_ALPHAS[0]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), gridspec_kw=dict(height_ratios=[1.4, 1]))
    d = daily.dropna(subset=[f"core_{a}"]) / 1e6

    ax = axes[0, 0]
    ax.fill_between(d.index, 0, d[f"core_{a}"], color="#2a6f97", alpha=.85, label="Core 99%")
    ax.fill_between(d.index, d[f"core_{a}"], d["balance"], color="#f4a259", alpha=.85, label="Non-core")
    ax.plot(d.index, d[f"core_oos_{a}"], color="black", lw=.8, ls="--", label="Core 99% out-of-sample")
    ax.set_title(f"{name}: core / non-core balance (millions)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=.3)

    ax = axes[0, 1]
    dd = daily.dropna(subset=[f"core_pct_{a}"])
    ax.plot(dd.index, dd["actual_retention_12m"], color="grey", lw=.8, label="Actual minimum 12m retention (ex post)")
    ax.plot(dd.index, dd[f"core_pct_{a}"], color="#2a6f97", lw=1.3, label="Core % 99%")
    ax.plot(dd.index, dd[f"core_pct_{CORE_ALPHAS[1]}"], color="#61a5c2", lw=1.3, label="Core % 95%")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_ylim(0, 1.4)
    ax.set_title("Estimated core % vs observed minimum 12-month retention")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)

    ax = axes[1, 0]
    for conf, c in (("99%", "#2a6f97"), ("95%", "#61a5c2")):
        b = buckets[(buckets["confidence"] == conf) & (buckets["bucket"] != "> 12m (CORE)")]
        ax.plot(["today"] + [x.split(" - ")[1] for x in b["bucket"]],
                [1] + list(b["cum_retention_pct"]), marker="o", color=c, label=f"Confidence {conf}")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_title(f"Retention curve at {daily.index[-1].date()}")
    ax.legend()
    ax.grid(alpha=.3)

    ax = axes[1, 1]
    x = np.arange(len(HORIZONS))
    for i, (al, c) in enumerate(((0.01, "#2a6f97"), (0.05, "#61a5c2"))):
        mm = metrics[metrics["alpha"] == al]
        ax.bar(x + (i - .5) * .38, mm["exc_xgb_calibrated"], .38, color=c, label=f"XGB calibrated α={al:.0%}")
        ax.scatter(x + (i - .5) * .38, mm["exc_xgb_raw"], color="black", marker="_", s=180,
                   label="XGB uncalibrated" if i == 0 else None)
        ax.axhline(al, color=c, ls=":", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([H_LABEL[h] for h in HORIZONS])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_title("Out-of-sample backtest: exception rate (dotted lines = target)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"chart_{key}.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# 6. Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, clean_log = clean_data(load_data(args.input))
    print(f"Records: {len(df)}  ({df['date'].min().date()} to {df['date'].max().date()})")

    res = {}
    for k in SERIES:
        print(f"Modelling {SERIES[k]} ...")
        res[k] = run_series(df, k, args.out)

    summary = pd.DataFrame([res[k]["summary"] for k in SERIES])
    xlsx = os.path.join(args.out, "core_noncore_xgboost.xlsx")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        summary.T.reset_index().rename(columns={"index": "item", 0: SERIES["sight"], 1: SERIES["other"]}) \
            .to_excel(w, sheet_name="Summary", index=False)
        pd.concat([res[k]["buckets"] for k in SERIES]).to_excel(w, sheet_name="Liquidity_buckets", index=False)
        for k in SERIES:
            res[k]["daily"].reset_index().to_excel(w, sheet_name=f"Daily_{k}", index=False)
        pd.concat([res[k]["metrics"].assign(series=SERIES[k]) for k in SERIES]).to_excel(
            w, sheet_name="Backtest", index=False)
        pd.concat([res[k]["grid"].assign(series=SERIES[k]) for k in SERIES]).to_excel(
            w, sheet_name="Hyperparameters", index=False)
        pd.concat([res[k]["importance"].assign(series=SERIES[k]) for k in SERIES]).to_excel(
            w, sheet_name="Importance", index=False)
        clean_log.to_excel(w, sheet_name="Cleaning", index=False)
    format_excel(xlsx, args.out)
    print(summary.T)
    print(f"Results in {args.out}")


def format_excel(xlsx, outdir):
    """Number formats, column widths and a sheet with charts."""
    from openpyxl import load_workbook
    from openpyxl.drawing.image import Image
    from openpyxl.styles import Font, PatternFill
    wb = load_workbook(xlsx)
    head = PatternFill("solid", fgColor="1F4E78")
    for ws in wb.worksheets:
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = head
        ws.freeze_panes = "B2"
        for col in ws.columns:
            name = str(col[0].value or "")
            w = max(10, min(45, len(name) + 2))
            ws.column_dimensions[col[0].column_letter].width = w
            pct = any(t in name for t in ("pct", "exc_", "retention", "skill", "kupiec", "alpha", "offset"))
            for c in col[1:]:
                if isinstance(c.value, float):
                    c.number_format = "0.00%" if pct else "#,##0.00" if abs(c.value) >= 100 else "0.0000"
    ws = wb["Summary"]
    for row in ws.iter_rows(min_row=2):
        lbl = str(row[0].value)
        for c in row[1:]:
            if isinstance(c.value, (int, float)):
                c.number_format = "0.00%" if ("pct" in lbl or "offset" in lbl) else "#,##0"
    ws.column_dimensions["A"].width = 44
    ws.column_dimensions["B"].width = 30
    ws.column_dimensions["C"].width = 30
    g = wb.create_sheet("Charts", 1)
    for i, k in enumerate(SERIES):
        img = Image(os.path.join(outdir, f"chart_{k}.png"))
        img.width, img.height = img.width * 0.55, img.height * 0.55
        g.add_image(img, f"A{1 + i * 36}")
    wb.save(xlsx)


if __name__ == "__main__":
    main()
