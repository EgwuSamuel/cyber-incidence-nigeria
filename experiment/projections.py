import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostRegressor
from . import config as C
from .models import tasks, make_model, fit, cb_fit, proba, MODELS
from .plots import save, MODEL_COLOR, NEUTRAL, INK2, SURFACE

SECTORS = C.SECTOR_KEEP + ["Other sectors"]


def oof_proba(name, X, y, classes):
    acc, cnt = np.zeros((len(X), len(classes))), np.zeros(len(X))
    for tr, te in RepeatedStratifiedKFold(n_splits=C.CV_SPLITS, n_repeats=C.CV_REPEATS,
                                          random_state=C.SEED).split(X, y):
        m = fit(make_model(name, weighted=False), name, X.iloc[tr], y.iloc[tr])
        acc[te] += proba(m, X.iloc[te], classes); cnt[te] += 1
    return acc / cnt[:, None]


def make_reg(name):
    if name == "Random Forest":
        return make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                             RandomForestRegressor(n_estimators=C.RF_PARAMS["n_estimators"], min_samples_leaf=3,
                                                   random_state=C.SEED, n_jobs=1))
    p = {k: v for k, v in C.CB_PARAMS.items() if k != "auto_class_weights"}
    if C.USE_GPU:
        p.update(C.GPU_PARAMS)
    return CatBoostRegressor(**p)


def fit_reg(name, X, y):
    m = make_reg(name)
    if name == "CatBoost":
        m = cb_fit(m, X, y)
    else:
        m.fit(X, y)
    return m


def person_rates(df):
    T = tasks(df)
    XA, yA = T["A"][0].reset_index(drop=True), T["A"][1].reset_index(drop=True)
    XB, yB = T["B1"]
    vic_idx = XB.index; XB, yB = XB.reset_index(drop=True), yB.reset_index(drop=True)
    types = sorted(yB.unique())
    rate = df.loc[vic_idx, "freq_self"].map(C.RATE_PER_YEAR).values / 12
    out = {}
    for name in MODELS:
        p_vic = oof_proba(name, XA, yA, ["Not victim", "Victim"])[:, 1]
        P = pd.DataFrame(proba(fit(make_model(name, weighted=False), name, XB, yB), XA, types),
                         columns=types, index=df.index)
        pos = df.index.get_indexer(vic_idx)
        P.iloc[pos] = oof_proba(name, XB, yB, types)
        r_hat = fit_reg(name, XB, pd.Series(rate)).predict(XA)
        acc, cnt = np.zeros(len(XB)), np.zeros(len(XB))
        for tr, te in RepeatedKFold(n_splits=C.CV_SPLITS, n_repeats=C.CV_REPEATS, random_state=C.SEED).split(XB):
            acc[te] += fit_reg(name, XB.iloc[tr], pd.Series(rate[tr])).predict(XB.iloc[te]); cnt[te] += 1
        r_hat[pos] = acc / cnt
        out[name] = dict(p_victim=p_vic, P_type=P, rate=np.clip(r_hat, 0, None),
                         lam=p_vic * np.clip(r_hat, 0, None))
    return out, types


def observed_baseline(df):
    end = pd.Timestamp(C.SURVEY_END)
    start = end - pd.DateOffset(months=12)
    when = pd.to_datetime(df.incident_month, format="%Y-%m", errors="coerce")
    recent = (df.victim == 1) & (when > start) & (when <= end) & df.attack_group.isin(C.ATTACK_GROUPS)
    rows = []
    for s in SECTORS:
        n = (df.sector_group == s).sum()
        for g in C.ATTACK_GROUPS:
            k = (recent & (df.sector_group == s) & (df.attack_group == g)).sum()
            for h in C.HORIZONS:
                rows.append({"sector": s, "attack_group": g, "horizon_months": h, "n_people": n,
                             "incidents_last_12m": int(k), "observed_per_100": 100 * k / n * h / 12})
    return pd.DataFrame(rows)


def run(df: pd.DataFrame):
    T = C.TABLES; rng = np.random.default_rng(C.SEED)
    est, types = person_rates(df)
    sector = df.sector_group.values
    rows, any_rows, sens = [], [], []
    for name, e in est.items():
        for s in SECTORS:
            idx = np.where(sector == s)[0]
            boots = [rng.choice(idx, len(idx)) for _ in range(C.N_BOOT)]
            for h in C.HORIZONS:
                for t in types:
                    val = e["P_type"][t].values * e["lam"] * h * 100
                    b = [val[i].mean() for i in boots]
                    rows.append({"model": name, "sector": s, "horizon_months": h, "attack_group": t,
                                 "n_people": len(idx), "expected_per_100": val[idx].mean(),
                                 "ci_low": np.percentile(b, 2.5), "ci_high": np.percentile(b, 97.5)})
                pa = 1 - np.exp(-e["lam"] * h)
                b = [pa[i].mean() for i in boots]
                any_rows.append({"model": name, "sector": s, "horizon_months": h,
                                 "prob_any_incident_pct": 100 * pa[idx].mean(),
                                 "ci_low": 100 * np.percentile(b, 2.5), "ci_high": 100 * np.percentile(b, 97.5)})
                for lab, f in C.RATE_SENSITIVITY.items():
                    sens.append({"model": name, "sector": s, "horizon_months": h, "rate_assumption": lab,
                                 "expected_all_types_per_100": 100 * (e["lam"][idx] * f * h).mean(),
                                 "prob_any_incident_pct": 100 * (1 - np.exp(-e["lam"][idx] * f * h)).mean()})
    proj = pd.DataFrame(rows); proj.round(3).to_csv(T / "P01_sector_projection_by_type.csv", index=False)
    anyp = pd.DataFrame(any_rows); anyp.round(2).to_csv(T / "P02_sector_prob_any_incident.csv", index=False)
    obs = observed_baseline(df); obs.round(3).to_csv(T / "P03_observed_last12m_baseline.csv", index=False)
    sens = pd.DataFrame(sens); sens.round(3).to_csv(T / "P04_rate_assumption_sensitivity.csv", index=False)
    share = (proj[proj.horizon_months == 3].assign(
        share_pct=lambda d: 100 * d.expected_per_100 / d.groupby(["model", "sector"]).expected_per_100.transform("sum")))
    share[["model", "sector", "attack_group", "share_pct"]].round(1).to_csv(T / "P05_projected_type_share.csv", index=False)
    piv = proj[proj.horizon_months == 3].pivot_table(index=["sector", "attack_group"], columns="model", values="expected_per_100")
    agree = np.corrcoef(piv["Random Forest"], piv["CatBoost"])[0, 1]
    pd.DataFrame([{"pearson_r_RF_vs_CatBoost": round(agree, 3),
                   "same_top_type_per_sector": int(sum(
                       piv.xs(s)["Random Forest"].idxmax() == piv.xs(s)["CatBoost"].idxmax() for s in SECTORS))}]
                 ).to_csv(T / "P06_model_agreement.csv", index=False)

    figures(proj, obs, anyp)

    md = ["## Objective 4b: projected incidents by sector (next 3 and 6 months)\n",
          f"- Assumed incidents/year: {C.RATE_PER_YEAR} ('Never' -> 'Rarely' for victims). Projections are "
          "scenario estimates for people like the respondents, not a time-series forecast for Nigeria as a whole."]
    for s in SECTORS:
        parts = []
        for name in MODELS:
            p = proj[(proj.model == name) & (proj.sector == s) & (proj.horizon_months == 3)].sort_values("expected_per_100", ascending=False)
            a = anyp[(anyp.model == name) & (anyp.sector == s) & (anyp.horizon_months == 3)].iloc[0]
            a6 = anyp[(anyp.model == name) & (anyp.sector == s) & (anyp.horizon_months == 6)].iloc[0]
            parts.append(f"{name}: P(any incident) {a.prob_any_incident_pct:.1f}% in 3 months / {a6.prob_any_incident_pct:.1f}% in 6; "
                         f"top type {p.iloc[0].attack_group} ({p.iloc[0].expected_per_100:.1f} per 100 people per 3 months, "
                         f"95% CI {p.iloc[0].ci_low:.1f}-{p.iloc[0].ci_high:.1f})")
        o = obs[(obs.sector == s) & (obs.horizon_months == 3)]
        md.append(f"- **{s}** (n = {o.n_people.iloc[0]}): " + " | ".join(parts)
                  + f" | observed in last 12 months: {o.observed_per_100.sum():.1f} per 100 people per 3 months.")
    md.append(f"- RF vs CatBoost agreement on the 3-month sector x type projections: Pearson r = {agree:.2f}.")
    b = sens[(sens.horizon_months == 3)].pivot_table(index=["model", "sector"], columns="rate_assumption",
                                                      values="prob_any_incident_pct")
    md.append("- Rate-assumption sensitivity (P(any incident) in 3 months, low / base / high): " + "; ".join(
        f"{m} {s}: {r['low (x0.5)']:.1f} / {r['base']:.1f} / {r['high (x2)']:.1f}%" for (m, s), r in b.iterrows()))
    return "\n".join(md) + "\n"


def figures(proj, obs, anyp):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), sharex=True)
    for ax, s in zip(axes, SECTORS):
        y = np.arange(len(C.ATTACK_GROUPS))[::-1]
        for k, name in enumerate(MODELS):
            p = proj[(proj.model == name) & (proj.sector == s) & (proj.horizon_months == 3)].set_index("attack_group").loc[C.ATTACK_GROUPS]
            ax.barh(y + (0.2 if k == 0 else -0.2), p.expected_per_100, height=0.38, color=MODEL_COLOR[name], label=name,
                    xerr=[p.expected_per_100 - p.ci_low, p.ci_high - p.expected_per_100], edgecolor=SURFACE, linewidth=1,
                    error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=2))
        o = obs[(obs.sector == s) & (obs.horizon_months == 3)].set_index("attack_group").loc[C.ATTACK_GROUPS]
        ax.scatter(o.observed_per_100, y, marker="D", s=30, color=NEUTRAL, edgecolor="#52514e", zorder=3,
                   label="Observed (last 12 months, per quarter)")
        ax.set_yticks(y); ax.set_yticklabels([g.split(" /")[0] for g in C.ATTACK_GROUPS]); ax.grid(axis="y", visible=False)
        ax.set_title(f"{s} (n = {o.n_people.iloc[0]})", fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    fig.supxlabel("Expected incidents per 100 people over the next 3 months (95% bootstrap CI)", y=0.08, fontsize=9, color=INK2)
    fig.suptitle("Fig. 11  Projected incidents by sector and attack group: Random Forest vs CatBoost",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    save(fig, "fig11_sector_projection_by_type", tight=False)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(SECTORS)); w = 0.2
    for k, (name, h) in enumerate([(m, h) for h in C.HORIZONS for m in MODELS]):
        a = anyp[(anyp.model == name) & (anyp.horizon_months == h)].set_index("sector").loc[SECTORS]
        col = MODEL_COLOR[name] if h == 3 else MODEL_COLOR[name] + "88"
        ax.bar(x + (k - 1.5) * w, a.prob_any_incident_pct, width=w * 0.92, color=col, label=f"{name}, {h} months",
               yerr=[a.prob_any_incident_pct - a.ci_low, a.ci_high - a.prob_any_incident_pct], edgecolor=SURFACE,
               linewidth=1, error_kw=dict(ecolor=INK2, elinewidth=0.8, capsize=2))
    ax.set_xticks(x); ax.set_xticklabels(SECTORS); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Probability of at least one incident (%)")
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Fig. 12  Projected chance of at least one incident, by sector"); save(fig, "fig12_sector_prob_any_incident")
