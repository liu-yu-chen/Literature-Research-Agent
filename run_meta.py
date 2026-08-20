# -*- coding: utf-8 -*-
"""
Meta-analysis of proportions for the ABM bibliometric corpus.

Study unit : publication year (1996-2025), k = 30 studies
Outcomes   : COVID-19 prevalence; prevalence of top main topics,
             top methodologies, top research fields  (17 outcomes total)
Model      : random-effects inverse-variance pooling on the logit scale
             (DerSimonian-Laird tau^2), I^2, Cochran's Q, prediction
             interval, subgroup analysis, meta-regression on year,
             funnel plots + Egger-type test for EVERY outcome.

All statistics are implemented manually (numpy/scipy) for transparency.
All figures are written to the figures/ directory.
"""
import json
import re
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- config
DATA_FILE = "llama_topics_v2.json"
OUT_DIR = "meta_analysis"
FIG_DIR = "figures"
YEAR_MIN, YEAR_MAX = 1996, 2025   # studies; 2026 excluded (incomplete year)
Z = stats.norm.ppf(0.975)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "axes.titlesize": 12,
})

# ---------------------------------------------------------------- load
def extract_json(text):
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return {}
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {}

def norm_method(m):
    m = str(m).strip()
    return m[0].upper() + m[1:].lower() if m else m

with open(DATA_FILE, encoding="utf-8") as f:
    raw = json.load(f)

records = []
for item in raw:
    y = item.get("year")
    try:
        y = int(str(y)[:4])
    except Exception:
        y = None
    if y is None or not (YEAR_MIN <= y <= YEAR_MAX):
        continue
    a = extract_json(item.get("topic_analysis", ""))
    records.append({
        "year": y,
        "is_covid19": bool(a.get("is_covid19", False)),
        "main_topic": (a.get("main_topic") or "").strip(),
        "research_field": (a.get("research_field") or "").strip(),
        "methodology": [norm_method(m) for m in (a.get("methodology") or []) if str(m).strip()],
    })

print(f"Records in window {YEAR_MIN}-{YEAR_MAX}: {len(records)}")

years = sorted({r["year"] for r in records})
n_by_year = pd.Series({y: sum(1 for r in records if r["year"] == y) for y in years})
print("Years with data:", len(years), "| total papers:", int(n_by_year.sum()))

def events_by_year(pred):
    return pd.Series({y: sum(1 for r in records if r["year"] == y and pred(r))
                      for y in years}, dtype=float)

# ---------------------------------------------------------------- outcomes
TOPICS = ["Social Network Analysis", "Urban Planning",
          "Transportation Planning and Optimization", "Population Dynamics",
          "Epidemiology"]
FIELDS = ["Computer Science", "Ecology", "Epidemiology", "Economics", "Sociology"]
METHODS = ["Agent-based modeling", "Simulation", "Regression analysis",
           "Network analysis", "Individual-based modeling", "Spatial analysis"]

OUTCOMES = [
    ("covid", "COVID-19 prevalence",
     events_by_year(lambda r: r["is_covid19"]), "COVID-19"),
]
for t in TOPICS:
    OUTCOMES.append((f"topic:{t}", f"Main topic: {t}",
                     events_by_year(lambda r, t=t: r["main_topic"] == t), t))
for m in METHODS:
    OUTCOMES.append((f"method:{m}", f"Methodology: {m}",
                     events_by_year(lambda r, m=m: m in r["methodology"]), m))
for f_ in FIELDS:
    OUTCOMES.append((f"field:{f_}", f"Research field: {f_}",
                     events_by_year(lambda r, f_=f_: r["research_field"] == f_), f_))

# ---------------------------------------------------------------- statistics
def study_logit(x, n):
    """logit of the observed proportion with 0.5 continuity correction."""
    xc = np.minimum(np.maximum(x, 0.5), n - 0.5)
    p = xc / n
    y = np.log(p / (1.0 - p))
    v = 1.0 / xc + 1.0 / (n - xc)   # within-study variance on logit scale
    return y, v

def _expit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))

def re_pool(x, n):
    """Random-effects (DerSimonian-Laird) pooling on the logit scale."""
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    y, v = study_logit(x, n)
    w = 1.0 / v
    y_fe = np.sum(w * y) / np.sum(w)
    Q = np.sum(w * (y - y_fe) ** 2)
    k = len(y)
    df = k - 1
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - df) / c)
    w_star = 1.0 / (v + tau2)
    y_re = np.sum(w_star * y) / np.sum(w_star)
    se_re = np.sqrt(1.0 / np.sum(w_star))
    ci = (y_re - Z * se_re, y_re + Z * se_re)
    t_crit = stats.t.ppf(0.975, max(df - 1, 1))
    pi = (y_re - t_crit * np.sqrt(se_re ** 2 + tau2),
          y_re + t_crit * np.sqrt(se_re ** 2 + tau2))
    I2 = max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0
    return {
        "k": int(k), "n_total": int(np.sum(n)), "events": int(np.sum(x)),
        "y": y, "v": v, "w_star": w_star,
        "Q": Q, "df": int(df), "I2": I2, "tau2": tau2,
        "pooled_logit": y_re, "se": se_re,
        "pooled_pct": 100 * float(_expit(y_re)),
        "ci_pct": tuple(float(100 * _expit(v_)) for v_ in ci),
        "pi_pct": tuple(float(100 * _expit(v_)) for v_ in pi),
        "range_pct": (float(np.min(x / n) * 100), float(np.max(x / n) * 100)),
    }

def meta_regression(x, n, cov):
    """Weighted least-squares meta-regression of logit(p) on covariate.

    tau^2 estimated by iterative DerSimonian-Laird on the model residuals;
    R^2 = 1 - tau2_model / tau2_empty.
    """
    y, v = study_logit(x, n)
    w0 = 1.0 / v
    y_fe = np.sum(w0 * y) / np.sum(w0)
    Q0 = np.sum(w0 * (y - y_fe) ** 2)
    c0 = np.sum(w0) - np.sum(w0 ** 2) / np.sum(w0)
    tau2_empty = max(0.0, (Q0 - (len(y) - 1)) / c0)

    X = np.column_stack([np.ones(len(cov)), np.asarray(cov, dtype=float)])
    p = X.shape[1]
    tau2 = tau2_empty
    for _ in range(200):
        w = 1.0 / (v + tau2)
        W = np.diag(w)
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
        resid = y - X @ beta
        Q_resid = np.sum(w * resid ** 2)
        c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
        tau2_new = max(0.0, (Q_resid - (len(y) - p)) / c)
        if abs(tau2_new - tau2) < 1e-10:
            tau2 = tau2_new
            break
        tau2 = tau2_new

    w = 1.0 / (v + tau2)
    W = np.diag(w)
    beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
    XWX_inv = np.linalg.inv(X.T @ W @ X)
    se_beta = np.sqrt(np.diag(XWX_inv))
    z_slope = beta[1] / se_beta[1]
    p_slope = 2 * (1 - stats.norm.cdf(abs(z_slope)))
    r2 = max(0.0, (tau2_empty - tau2) / tau2_empty) if tau2_empty > 0 else 0.0
    return {
        "intercept": beta[0], "slope": beta[1],
        "slope_se": se_beta[1], "slope_z": z_slope, "slope_p": p_slope,
        "tau2": tau2, "tau2_empty": tau2_empty, "r2": r2,
    }

def egger_test(x, n):
    """Egger-type test on the logit scale: regress y/se on 1/se."""
    y, v = study_logit(x, n)
    se = np.sqrt(v)
    precision = 1.0 / se
    z = y / se
    slope, intercept, r, p, se_slope = stats.linregress(precision, z)
    return {"intercept": intercept, "p": p, "slope": slope}

def outcome_series(key):
    for k, label, series, short in OUTCOMES:
        if k == key:
            return series
    raise KeyError(key)

# ---------------------------------------------------------------- run
rows_summary, rows_years, rows_sub = [], [], []
subgroups = {"1996-2009": (1996, 2009), "2010-2019": (2010, 2019), "2020-2025": (2020, 2025)}

for key, label, x_series, short in OUTCOMES:
    n_series = n_by_year.reindex(x_series.index)
    x = x_series.values.astype(float)
    n = n_series.values.astype(float)
    keep = n > 0
    x, n = x[keep], n[keep]
    yv = np.array(x_series.index)[keep]

    res = re_pool(x, n)
    reg = meta_regression(x, n, yv)
    eg = egger_test(x, n)

    rows_summary.append({
        "outcome": key, "label": label, "short": short,
        "k": res["k"], "total_n": res["n_total"], "events": res["events"],
        "pooled_pct": round(res["pooled_pct"], 3),
        "ci_low_pct": round(res["ci_pct"][0], 3),
        "ci_high_pct": round(res["ci_pct"][1], 3),
        "pi_low_pct": round(res["pi_pct"][0], 3),
        "pi_high_pct": round(res["pi_pct"][1], 3),
        "range_low_pct": round(res["range_pct"][0], 3),
        "range_high_pct": round(res["range_pct"][1], 3),
        "Q": round(res["Q"], 3), "df": res["df"],
        "I2_pct": round(res["I2"], 2), "tau2": round(res["tau2"], 5),
        "reg_slope_per_year": round(reg["slope"], 5),
        "reg_slope_se": round(reg["slope_se"], 5),
        "reg_z": round(reg["slope_z"], 3),
        "reg_p": f"{reg['slope_p']:.3g}",
        "reg_r2": round(reg["r2"], 3),
        "egger_intercept": round(eg["intercept"], 4),
        "egger_p": f"{eg['p']:.3g}",
    })

    for yy, xi, ni, yi, vi, wi in zip(yv, x, n, res["y"], res["v"], res["w_star"]):
        rows_years.append({
            "outcome": key, "label": label, "short": short, "year": int(yy),
            "n": int(ni), "events": int(xi),
            "pct": round(100 * xi / ni, 3),
            "logit": round(yi, 5), "var_logit": round(vi, 6),
            "se_logit": round(np.sqrt(vi), 5),
            "w_re": round(wi, 5),
        })

    for sname, (s0, s1) in subgroups.items():
        mask = (yv >= s0) & (yv <= s1)
        if mask.sum() >= 2:
            sr = re_pool(x[mask], n[mask])
            rows_sub.append({
                "outcome": key, "label": label, "short": short, "subgroup": sname,
                "k": sr["k"], "n": sr["n_total"], "events": sr["events"],
                "pooled_pct": round(sr["pooled_pct"], 3),
                "ci_low_pct": round(sr["ci_pct"][0], 3),
                "ci_high_pct": round(sr["ci_pct"][1], 3),
                "I2_pct": round(sr["I2"], 2),
            })

df_summary = pd.DataFrame(rows_summary)
df_years = pd.DataFrame(rows_years)
df_sub = pd.DataFrame(rows_sub)
df_summary.to_csv(os.path.join(OUT_DIR, "meta_analysis/meta_pooled_summary.csv"), index=False, encoding="utf-8-sig")
df_years.to_csv(os.path.join(OUT_DIR, "meta_analysis/meta_year_data.csv"), index=False, encoding="utf-8-sig")
df_sub.to_csv(os.path.join(OUT_DIR, "meta_analysis/meta_subgroups.csv"), index=False, encoding="utf-8-sig")

pd.set_option("display.width", 220)
print("\n================= POOLED SUMMARY (all 17 outcomes) =================")
print(df_summary[["short", "pooled_pct", "ci_low_pct", "ci_high_pct", "I2_pct",
                  "reg_slope_per_year", "reg_p", "reg_r2", "egger_p"]].to_string(index=False))

# ---------------------------------------------------------------- figures
def _short_name(s, limit=34):
    s = str(s)
    return s if len(s) <= limit else s[:limit - 1] + "…"

def draw_forest(ax, key, label, x_series, pooled, xmax_pct=None):
    n_series = n_by_year.reindex(x_series.index)
    x = x_series.values.astype(float)
    n = n_series.values.astype(float)
    keep = n > 0
    x, n = x[keep], n[keep]
    yv = np.array(x_series.index)[keep]
    y, v = study_logit(x, n)
    se = np.sqrt(v)
    pct = 100 * x / n
    w = pooled["w_star"]

    order = np.argsort(-yv)
    yv_o, pct_o, se_o, w_o = yv[order], pct[order], se[order], w[order]
    w_norm = w_o / w_o.max()

    for i, (yy, pp, ss, ww) in enumerate(zip(yv_o, pct_o, se_o, w_norm)):
        pp_c = np.clip(pp, 0.05, 99.95)
        lo = float(100 * _expit(np.log(pp_c / (100 - pp_c)) - Z * ss))
        hi = float(100 * _expit(np.log(pp_c / (100 - pp_c)) + Z * ss))
        ax.plot([lo, hi], [i, i], color="#555555", lw=1.1, zorder=2)
        ax.scatter([pp], [i], s=3 + 60 * ww, color="#2a6f97", zorder=3,
                   edgecolors="white", linewidths=0.4)

    pct_p = pooled["pooled_pct"]
    ci_lo, ci_hi = pooled["ci_pct"]
    ax.plot([ci_lo, pct_p], [len(yv_o), len(yv_o) - 0.35], color="#d62728", lw=2.4)
    ax.plot([pct_p, ci_hi], [len(yv_o) - 0.35, len(yv_o)], color="#d62728", lw=2.4)
    ax.plot([ci_lo, pct_p], [len(yv_o), len(yv_o) + 0.35], color="#d62728", lw=2.4)
    ax.plot([pct_p, ci_hi], [len(yv_o) + 0.35, len(yv_o)], color="#d62728", lw=2.4)
    ax.scatter([pct_p], [len(yv_o)], s=30, color="#d62728", zorder=4)
    pi_lo, pi_hi = pooled["pi_pct"]
    ax.plot([pi_lo, pi_hi], [len(yv_o) + 1.1, len(yv_o) + 1.1],
            color="#d62728", lw=1.2, ls="--")
    ax.axvline(pct_p, color="#d62728", lw=0.6, ls=":", alpha=0.6)

    yticks = list(range(len(yv_o))) + [len(yv_o)]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(int(yy)) for yy in yv_o] + ["RE"], fontsize=6.5)
    ax.set_ylim(-1, len(yv_o) + 2.4)
    if xmax_pct is None:
        xmax_pct = max(float(100 * _expit(pooled["ci_pct"][1])),
                       np.max(pct) * 1.1, 5)
    ax.set_xlim(0, min(xmax_pct, 100))
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title((f"{_short_name(label)}\nRE pooled = {pct_p:.2f}%  "
                  f"(95% CI {ci_lo:.2f}-{ci_hi:.2f})  |  "
                  f"I² = {pooled['I2']:.1f}%  τ² = {pooled['tau2']:.4f}"),
                 fontsize=8.5, loc="left")
    ax.set_xlabel("Proportion of papers (%)", fontsize=7)

def draw_funnel(ax, x_series, pooled_logit, title):
    n_series = n_by_year.reindex(x_series.index)
    x = x_series.values.astype(float)
    n = n_series.values.astype(float)
    keep = n > 0
    y, v = study_logit(x[keep], n[keep])
    se = np.sqrt(v)
    eg = egger_test(x[keep], n[keep])
    ax.scatter(y, se, s=18, color="#2a6f97", alpha=0.8, edgecolors="white", zorder=3)
    se_grid = np.linspace(se.min(), se.max() * 1.3, 80)
    ax.plot(pooled_logit + Z * se_grid, se_grid, "--", color="#d62728", lw=0.9)
    ax.plot(pooled_logit - Z * se_grid, se_grid, "--", color="#d62728", lw=0.9)
    ax.axvline(pooled_logit, color="#d62728", lw=1.0, ls=":")
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_title(f"{_short_name(title)}\nEgger p = {eg['p']:.3f}", fontsize=9)
    ax.set_xlabel("Logit(proportion)", fontsize=7)
    ax.set_ylabel("SE (logit)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2)

def draw_regression(ax, x_series, title):
    n_series = n_by_year.reindex(x_series.index)
    x = x_series.values.astype(float)
    n = n_series.values.astype(float)
    keep = n > 0
    x, n = x[keep], n[keep]
    yv = np.array(x_series.index)[keep]
    y, v = study_logit(x, n)
    reg = meta_regression(x, n, yv)
    ax.scatter(yv, y, s=18, color="#2a6f97", alpha=0.8, edgecolors="white", zorder=3)
    xx = np.linspace(yv.min(), yv.max(), 80)
    ax.plot(xx, reg["intercept"] + reg["slope"] * xx, color="#d62728", lw=1.8,
            label=f"slope {reg['slope']:.3f} log-odds/yr (p={reg['slope_p']:.2g}, R²={reg['r2']:.2f})")
    ax.legend(fontsize=6.5)
    ax.set_title(_short_name(title), fontsize=9)
    ax.set_xlabel("Publication year", fontsize=7)
    ax.set_ylabel("Logit(proportion)", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2)

def group_figures(items, fname_forest, fname_funnel, fname_reg,
                  cols, figsize, xmax=None, ncols_funnel=None):
    items = list(items)
    # forest
    fig, axes = plt.subplots(1, cols, figsize=figsize)
    for ax, (key, label, series, short) in zip(axes, items):
        n_series = n_by_year.reindex(series.index)
        x = series.values.astype(float)
        n = n_series.values.astype(float)
        keep = n > 0
        res = re_pool(x[keep], n[keep])
        draw_forest(ax, key, label, series, res, xmax_pct=xmax)
    fig.suptitle("Meta-analysis of prevalence (year-studies, 1996-2025)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(FIG_DIR, fname_forest))
    plt.close(fig)
    print("saved", fname_forest)

    # funnel
    fc = ncols_funnel or cols
    rows_f = int(np.ceil(len(items) / fc))
    fig, axes = plt.subplots(rows_f, fc, figsize=(5.2 * fc, 4.6 * rows_f))
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, label, series, short) in zip(axes, items):
        n_series = n_by_year.reindex(series.index)
        x = series.values.astype(float)
        n = n_series.values.astype(float)
        keep = n > 0
        res = re_pool(x[keep], n[keep])
        draw_funnel(ax, series, res["pooled_logit"], short)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle("Funnel plots (logit scale, pseudo-95% CI boundaries)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(FIG_DIR, fname_funnel))
    plt.close(fig)
    print("saved", fname_funnel)

    # regression
    fig, axes = plt.subplots(rows_f, fc, figsize=(5.2 * fc, 4.6 * rows_f))
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, label, series, short) in zip(axes, items):
        draw_regression(ax, series, short)
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.suptitle("Meta-regression of logit prevalence on publication year",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(FIG_DIR, fname_reg))
    plt.close(fig)
    print("saved", fname_reg)

topic_items = [o for o in OUTCOMES if o[0].startswith("topic:")]
meth_items = [o for o in OUTCOMES if o[0].startswith("method:")]
field_items = [o for o in OUTCOMES if o[0].startswith("field:")]
covid_item = [o for o in OUTCOMES if o[0] == "covid"]

group_figures(topic_items, "meta_topics_forest.png", "meta_topics_funnel.png",
              "meta_topics_regression.png", cols=5, figsize=(22, 9), xmax=26)
group_figures(meth_items, "meta_methods_forest.png", "meta_methods_funnel.png",
              "meta_methods_regression.png", cols=6, figsize=(26, 9), xmax=95)
group_figures(field_items, "meta_fields_forest.png", "meta_fields_funnel.png",
              "meta_fields_regression.png", cols=5, figsize=(22, 9), xmax=40)

# ---- COVID single-panel versions (kept for the special-case section)
key, label, series, short = covid_item[0]
n_series = n_by_year.reindex(series.index)
x = series.values.astype(float)
n = n_series.values.astype(float)
keep = n > 0
covid_res = re_pool(x[keep], n[keep])

fig, ax = plt.subplots(figsize=(7.2, 9))
draw_forest(ax, key, "COVID-19 prevalence (per-year studies, 1996-2025)",
            series, covid_res, xmax_pct=16)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "meta_covid_forest.png"))
plt.close(fig)
print("saved meta_covid_forest.png")

fig, ax = plt.subplots(figsize=(7.5, 6.5))
draw_funnel(ax, series, covid_res["pooled_logit"], "COVID-19 prevalence")
ax.set_title(f"Funnel plot — COVID-19 prevalence (Egger p = "
             f"{egger_test(x[keep], n[keep])['p']:.3f})", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "meta_covid_funnel.png"))
plt.close(fig)
print("saved meta_covid_funnel.png")

fig, ax = plt.subplots(figsize=(8, 5.5))
draw_regression(ax, series, "COVID-19 prevalence")
ax.set_title("Meta-regression of COVID-19 prevalence on publication year", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "meta_covid_regression.png"))
plt.close(fig)
print("saved meta_covid_regression.png")

# ---- summary forest of all 17 pooled estimates
summ = df_summary.sort_values("pooled_pct")
fig, ax = plt.subplots(figsize=(10, 9))
ypos = np.arange(len(summ))
for i, (_, r) in enumerate(summ.iterrows()):
    lo, hi = r["ci_low_pct"], r["ci_high_pct"]
    sig = r["reg_p"] and float(r["reg_p"]) < 0.05
    col = "#d62728" if sig else "#2a6f97"
    ax.plot([lo, hi], [i, i], color=col, lw=1.6, alpha=0.85)
    ax.scatter([r["pooled_pct"]], [i], s=42, color=col, zorder=3,
               edgecolors="white", linewidths=0.5)
    ax.text(hi + max(summ["ci_high_pct"]) * 0.02, i,
            f"{r['pooled_pct']:.1f}%  [I²={r['I2_pct']:.0f}%]  "
            f"slope {r['reg_slope_per_year']:+.3f}{'*' if sig else ''}",
            va="center", fontsize=7.5)
ax.set_yticks(ypos)
ax.set_yticklabels([_short_name(s, 40) for s in summ["short"]], fontsize=8)
ax.set_xlabel("Pooled prevalence (%) — random effects, logit scale", fontsize=9)
ax.set_title("Summary forest: pooled prevalence of all 17 outcomes (1996-2025)\n"
             "red = significant yearly trend (p<0.05), * = p<0.05",
             fontsize=11)
ax.grid(axis="x", alpha=0.25)
ax.set_xlim(0, max(summ["ci_high_pct"]) * 1.35)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "meta_summary_forest.png"))
plt.close(fig)
print("saved meta_summary_forest.png")

# ---- subgroup heatmap (pooled % by outcome x era)
piv = df_sub.pivot_table(index="short", columns="subgroup", values="pooled_pct")
piv = piv[list(subgroups.keys())]
order = df_summary.sort_values("pooled_pct")["short"].tolist()
piv = piv.loc[[s for s in order if s in piv.index]]

fig, ax = plt.subplots(figsize=(9, 11))
im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto")
ax.set_xticks(range(len(piv.columns)))
ax.set_xticklabels(piv.columns, fontsize=10)
ax.set_yticks(range(len(piv.index)))
ax.set_yticklabels([_short_name(s, 36) for s in piv.index], fontsize=8)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.values[i, j]
        ax.text(j, i, f"{v:.2f}" if v > 0 else "0", ha="center", va="center",
                fontsize=7, color="#333")
ax.set_title("Subgroup pooled prevalence (%) by era (random effects)",
             fontsize=12)
fig.colorbar(im, ax=ax, shrink=0.7, label="Pooled prevalence (%)")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "meta_subgroup_heatmap.png"))
plt.close(fig)
print("saved meta_subgroup_heatmap.png")

print("\nAll meta-analysis outputs written to:", OUT_DIR, "and", FIG_DIR)
