import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier, CatBoostRegressor
from . import config as C
from .ahp_topsis import topsis, entropy_weights
from .models import cb_fit
from .plots import save, SERIES, INK2, NEUTRAL, SURFACE

SECTORS = ["Public administration", "Health", "Information and telecom", "Finance", "Education", "Other sectors"]
GROUPS = ["Server and data theft", "Ransomware and destructive", "Denial of service", "Account and message compromise"]
GROUP_COLOR = dict(zip(GROUPS, SERIES))
FEATURES = ["sector", "region", "actor", "motive"]

SECTOR_MAP = {"public administration": "Public administration", "health care and social assistance": "Health",
              "information": "Information and telecom", "finance and insurance": "Finance",
              "educational services": "Education", "undetermined": None}
GROUP_MAP = {
    "exploitation of application server": GROUPS[0], "exploitation of network infrastructure": GROUPS[0],
    "exploitation of sensors": GROUPS[0], "exploitation of data in transit": GROUPS[0],
    "data attack": GROUPS[1], "physical attack": GROUPS[1], "internal denial of service": GROUPS[1],
    "external denial of service": GROUPS[2],
    "exploitation of end hosts": GROUPS[3], "message manipulation": GROUPS[3],
}
ACTOR_MAP = {"criminal": "Criminal", "hacktivist": "Hacktivist", "nation-state": "Nation-state",
             "hobbyist": "Hobbyist", "terrorist": "Other", "undetermined": "Undetermined"}
MOTIVE_MAP = {"financial": "Financial", "protest": "Protest", "political-espionage": "Espionage",
              "industrial-espionage": "Espionage", "sabotage": "Sabotage", "undetermined": "Undetermined"}
ASIA_PACIFIC = {"Australia", "Japan", "Republic of Korea", "New Zealand", "Taiwan", "Singapore",
                "China", "India", "Indonesia", "Philippines", "Malaysia", "Thailand", "Viet Nam", "Pakistan"}


def first(s):
    return str(s).split(",")[0].strip().lower()


def region(r):
    if r.country == "United States of America":
        return "United States"
    if r.au == 1:
        return "Africa"
    if r.country in ASIA_PACIFIC or r.asean == 1:
        return "Asia-Pacific"
    if r.osce == 1 or r.eu == 1:
        return "Europe and Central Asia"
    if r.oas == 1:
        return "Other Americas"
    return "Other or undetermined"


def load():
    files = sorted(C.CEDB_DIR.glob("*.csv"))
    if not files:
        return None
    d = pd.read_csv(files[-1], low_memory=False)
    d["date"] = pd.to_datetime(d.event_date, errors="coerce")
    d["ym"] = d.date.dt.to_period("M")
    d["sector"] = d.industry.str.strip().str.lower().map(lambda s: SECTOR_MAP.get(s, "Other sectors"))
    d["group"] = d.event_subtype.map(first).map(GROUP_MAP)
    d["actor"] = d.actor_type.map(first).map(lambda s: ACTOR_MAP.get(s, "Other"))
    d["motive_c"] = d.motive.map(first).map(lambda s: MOTIVE_MAP.get(s, "Other"))
    d["region"] = d.apply(region, axis=1)
    d["gdelt"] = (d.original_method == 0).astype(int)
    return d


def clf_model(name):
    if name == "Random Forest":
        return make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                             RandomForestClassifier(**C.RF_PARAMS))
    return CatBoostClassifier(**C.CB_PARAMS)


def fit_clf(name, X, y):
    m = clf_model(name)
    if name == "CatBoost":
        return cb_fit(m, X, y)
    return m.fit(X, y)


def predict(m, X):
    return np.asarray(m.predict(X)).ravel()


def macro_f1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def perm_score(name, X, y, Xte, yte, seed):
    yp = pd.Series(np.random.default_rng(seed).permutation(y.values), index=y.index)
    return macro_f1(yte, predict(fit_clf(name, X, yp), Xte))


def classification(d, rng, md):
    T = C.TABLES
    m = d[d.group.notna() & d.sector.notna()].copy()
    m["motive"] = m.motive_c
    tr, te = m[m.year <= C.CEDB_TRAIN_END], m[m.year > C.CEDB_TRAIN_END]
    assert tr.date.max() < te.date.min(), "training events must precede test events"
    Xtr, ytr, Xte, yte = tr[FEATURES].astype(str), tr.group, te[FEATURES].astype(str), te.group
    rows, preds = [], {}
    for strat in ["most_frequent", "stratified"]:
        p = DummyClassifier(strategy=strat, random_state=C.SEED).fit(Xtr, ytr).predict(Xte)
        rows.append({"model": f"Baseline: {strat.replace('_', ' ')}", "macro_f1": macro_f1(yte, p),
                     "balanced_accuracy": balanced_accuracy_score(yte, p), "accuracy": accuracy_score(yte, p)})
    for name in ["Random Forest", "CatBoost"]:
        p = predict(fit_clf(name, Xtr, ytr), Xte)
        preds[name] = p
        boots = []
        idx = np.arange(len(yte))
        for _ in range(C.N_BOOT):
            i = rng.choice(idx, len(idx))
            boots.append(macro_f1(yte.values[i], p[i]))
        null = Parallel(n_jobs=C.N_JOBS)(delayed(perm_score)(name, Xtr, ytr, Xte, yte, C.SEED + k)
                                         for k in range(C.N_PERM_CEDB))
        obs = macro_f1(yte, p)
        rows.append({"model": name, "macro_f1": obs, "macro_f1_lo": np.percentile(boots, 2.5),
                     "macro_f1_hi": np.percentile(boots, 97.5),
                     "balanced_accuracy": balanced_accuracy_score(yte, p), "accuracy": accuracy_score(yte, p),
                     "p_perm": (1 + np.sum(np.array(null) >= obs)) / (1 + C.N_PERM_CEDB)})
    res = pd.DataFrame(rows)
    rf_ok, cb_ok = preds["Random Forest"] == yte.values, preds["CatBoost"] == yte.values
    b, c = int(np.sum(rf_ok & ~cb_ok)), int(np.sum(~rf_ok & cb_ok))
    p_mcnemar = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    diffs = []
    idx = np.arange(len(yte))
    for _ in range(C.N_BOOT):
        i = rng.choice(idx, len(idx))
        diffs.append(macro_f1(yte.values[i], preds["CatBoost"][i]) - macro_f1(yte.values[i], preds["Random Forest"][i]))
    comp = pd.DataFrame([{"rf_correct_cb_wrong": b, "cb_correct_rf_wrong": c, "p_mcnemar": p_mcnemar,
                          "diff_macro_f1_cb_minus_rf": np.mean(diffs), "diff_lo": np.percentile(diffs, 2.5),
                          "diff_hi": np.percentile(diffs, 97.5)}])
    res.round(4).to_csv(T / "C04_classification_temporal.csv", index=False)
    comp.round(4).to_csv(T / "C05_rf_vs_catboost.csv", index=False)
    s = res.set_index("model")
    md.append(f"- Classification of attack group from sector, region, actor type and motive. Trained on {len(tr):,} "
              f"events up to {C.CEDB_TRAIN_END}, tested on {len(te):,} events from {C.CEDB_TRAIN_END + 1} to "
              f"{te.date.max():%B %Y}. Macro-F1: RF {s.loc['Random Forest','macro_f1']:.3f} "
              f"(95% CI {s.loc['Random Forest','macro_f1_lo']:.3f}-{s.loc['Random Forest','macro_f1_hi']:.3f}), "
              f"CatBoost {s.loc['CatBoost','macro_f1']:.3f} ({s.loc['CatBoost','macro_f1_lo']:.3f}-"
              f"{s.loc['CatBoost','macro_f1_hi']:.3f}); baselines majority "
              f"{s.loc['Baseline: most frequent','macro_f1']:.3f}, random {s.loc['Baseline: stratified','macro_f1']:.3f}. "
              f"Balanced accuracy RF {s.loc['Random Forest','balanced_accuracy']:.3f}, CatBoost "
              f"{s.loc['CatBoost','balanced_accuracy']:.3f}. Permutation p (training labels shuffled "
              f"{C.N_PERM_CEDB} times): RF {s.loc['Random Forest','p_perm']:.3f}, CatBoost {s.loc['CatBoost','p_perm']:.3f}.")
    md.append(f"- RF vs CatBoost on the same test events: McNemar p = {p_mcnemar:.3f}; CatBoost minus RF macro-F1 "
              f"{comp.diff_macro_f1_cb_minus_rf[0]:+.3f} (95% CI {comp.diff_lo[0]:+.3f} to {comp.diff_hi[0]:+.3f}).")
    return res, comp


def monthly_grid(d):
    m = d[d.group.notna() & d.sector.notna() & (d.ym <= pd.Period(C.CEDB_LAST_MONTH, "M"))]
    months = pd.period_range(m.ym.min(), C.CEDB_LAST_MONTH, freq="M")
    idx = pd.MultiIndex.from_product([SECTORS, GROUPS, months], names=["sector", "group", "ym"])
    g = m.groupby(["sector", "group", "ym"]).size().reindex(idx, fill_value=0).rename("y").reset_index()
    return g


def make_rows(g, h):
    out = []
    for (s, k), q in g.groupby(["sector", "group"], sort=False):
        y = q.y.to_numpy(float)
        ym = q.ym.to_numpy()
        for t in range(12, len(y)):
            fut = y[t + 1:t + 1 + h]
            out.append({"sector": s, "group": k, "ym": ym[t], "lag0": y[t], "lag1": y[t - 1], "lag2": y[t - 2],
                        "m3": y[t - 2:t + 1].mean(), "m6": y[t - 5:t + 1].mean(), "m12": y[t - 11:t + 1].mean(),
                        "seas": y[t - 11:t - 11 + h].sum(), "moy": ym[t].month,
                        "gdelt": int(ym[t] >= pd.Period("2025-01", "M")),
                        "target": fut.sum() if len(fut) == h else np.nan})
    return pd.DataFrame(out)


NUM = ["lag0", "lag1", "lag2", "m3", "m6", "m12", "seas", "moy", "gdelt"]


def reg_model(name):
    if name == "Random Forest":
        return make_pipeline(make_column_transformer((OneHotEncoder(handle_unknown="ignore"), ["sector", "group"]),
                                                     remainder="passthrough"),
                             RandomForestRegressor(n_estimators=C.RF_PARAMS["n_estimators"], min_samples_leaf=3,
                                                   random_state=C.SEED, n_jobs=1))
    p = {k: v for k, v in C.CB_PARAMS.items() if k != "auto_class_weights"}
    return CatBoostRegressor(**p)


def fit_reg(name, X, y):
    m = reg_model(name)
    if name == "CatBoost":
        return m.fit(X, y, cat_features=[0, 1])
    return m.fit(X, y)


def forecast(d, md):
    T = C.TABLES
    g = monthly_grid(d)
    last = pd.Period(C.CEDB_LAST_MONTH, "M")
    back, fc = [], []
    for h in C.HORIZONS:
        R = make_rows(g, h)
        X = R[["sector", "group"] + NUM]
        origins = pd.period_range(last - C.CEDB_BACKTEST_MONTHS - h, last - h, freq="M")
        for o in origins:
            trn = R[(R.ym + h <= o) & R.target.notna()]
            tst = R[R.ym == o]
            for name in ["Random Forest", "CatBoost"]:
                mdl = fit_reg(name, X.loc[trn.index], trn.target)
                pr = np.clip(mdl.predict(X.loc[tst.index]), 0, None)
                back += [{"h": h, "origin": str(o), "model": name, "sector": a, "group": b, "y": yy, "pred": pp}
                         for a, b, yy, pp in zip(tst.sector, tst.group, tst.target, pr)]
            back += [{"h": h, "origin": str(o), "model": "Seasonal naive", "sector": a, "group": b, "y": yy, "pred": pp}
                     for a, b, yy, pp in zip(tst.sector, tst.group, tst.target, tst.seas)]
            back += [{"h": h, "origin": str(o), "model": "12-month mean", "sector": a, "group": b, "y": yy, "pred": pp}
                     for a, b, yy, pp in zip(tst.sector, tst.group, tst.target, tst.m12 * h)]
        trn = R[R.target.notna()]
        now = R[R.ym == last]
        bt = pd.DataFrame([r for r in back if r["h"] == h])
        for name in ["Random Forest", "CatBoost"]:
            mdl = fit_reg(name, X.loc[trn.index], trn.target)
            pr = np.clip(mdl.predict(X.loc[now.index]), 0, None)
            res = bt[bt.model == name].assign(e=lambda q: q.y - q.pred)
            lo_q, hi_q = np.percentile(res.e, [5, 95])
            fc += [{"h": h, "model": name, "sector": a, "group": b, "forecast": p,
                    "lo90": max(0, p + lo_q), "hi90": p + hi_q} for a, b, p in zip(now.sector, now.group, pr)]
    back = pd.DataFrame(back)
    back["ae"] = (back.y - back.pred).abs()
    score = back.groupby(["h", "model"]).agg(mae=("ae", "mean"), rmse=("ae", lambda e: np.sqrt(np.mean(e ** 2))),
                                             n=("ae", "size")).reset_index()
    tests = []
    for h in C.HORIZONS:
        b = back[back.h == h].pivot_table(index=["origin", "sector", "group"], columns="model", values="ae")
        for a, c in [("Random Forest", "Seasonal naive"), ("CatBoost", "Seasonal naive"),
                     ("Random Forest", "12-month mean"), ("CatBoost", "12-month mean"), ("CatBoost", "Random Forest")]:
            w = stats.wilcoxon(b[a], b[c])
            tests.append({"h": h, "comparison": f"{a} vs {c}", "mae_a": b[a].mean(), "mae_b": b[c].mean(), "p_wilcoxon": w.pvalue})
    score.round(3).to_csv(T / "C06_forecast_backtest.csv", index=False)
    pd.DataFrame(tests).round(4).to_csv(T / "C06b_forecast_tests.csv", index=False)
    fc = pd.DataFrame(fc)
    fc.round(2).to_csv(T / "C07_forecast_sector_group.csv", index=False)
    best = {h: score[(score.h == h) & score.model.isin(["Random Forest", "CatBoost"])].sort_values("mae").model.iloc[0]
            for h in C.HORIZONS}
    md.append(f"- Forecasts: monthly counts for {len(SECTORS)} sectors x {len(GROUPS)} attack groups "
              f"({g.ym.min()} to {last}). Rolling backtest with {C.CEDB_BACKTEST_MONTHS + 1} forecast origins. MAE per "
              "series: " + "; ".join(f"{r.h} months {r.model} {r.mae:.2f}" for r in score.itertuples()) + ".")
    tt = pd.DataFrame(tests)
    md.append("- Paired Wilcoxon tests on absolute errors: " + "; ".join(
        f"{r.h}m {r.comparison}: p = {r.p_wilcoxon:.3f}" for r in tt.itertuples()))
    for h in C.HORIZONS:
        f = fc[(fc.h == h) & (fc.model == best[h])]
        tot = f.groupby("group").forecast.sum().reindex(GROUPS)
        sec = f.groupby("sector").forecast.sum().reindex(SECTORS)
        md.append(f"- Next {h} months after {last} ({best[h]}, best backtest MAE): total {f.forecast.sum():.0f} events; by "
                  "group " + ", ".join(f"{k} {v:.0f}" for k, v in tot.items()) + "; by sector " +
                  ", ".join(f"{k} {v:.0f}" for k, v in sec.items()) + ".")
    return g, fc, best, score


def group_ranking(d, md):
    T = C.TABLES
    rows = []
    for scope, sub in [("World", d), ("Africa", d[d.au == 1])]:
        sub = sub[sub.group.notna()]
        rec, old = sub[sub.year > C.CEDB_TRAIN_END], sub[sub.year <= C.CEDB_TRAIN_END]
        for k in GROUPS:
            s = sub[sub.group == k]
            sec = s.sector.value_counts(normalize=True)
            breadth = float(-(sec * np.log(sec)).sum() / np.log(len(SECTORS))) if len(sec) > 1 else 0.0
            rows.append({"scope": scope, "group": k, "n": len(s),
                         "share_recent": (rec.group == k).mean(), "share_earlier": (old.group == k).mean(),
                         "breadth": breadth, "disruptive": s.event_type.isin(["Disruptive", "Mixed"]).mean(),
                         "criminal": (s.actor == "Criminal").mean()})
    t = pd.DataFrame(rows)
    t["growth"] = t.share_recent - t.share_earlier
    out = []
    for scope, q in t.groupby("scope", sort=False):
        X = q[["share_recent", "growth", "breadth", "disruptive", "criminal"]].to_numpy(float)
        X[:, 1] = X[:, 1] - X[:, 1].min() + 1e-3
        w_ent = entropy_weights(X + 1e-9)
        for wname, w in [("Entropy", w_ent), ("Equal", np.full(5, 0.2))]:
            c = topsis(X, w)[0]
            phat = np.clip(q.share_recent + q.growth, 0, None).to_numpy()
            for k, ci, fti in zip(q.group, c, phat * c):
                out.append({"scope": scope, "weights": wname, "group": k, "closeness": ci, "fti": fti})
    o = pd.DataFrame(out).merge(t, on=["scope", "group"])
    o.round(4).to_csv(T / "C08_group_topsis.csv", index=False)
    for scope in ["World", "Africa"]:
        q = o[(o.scope == scope) & (o.weights == "Entropy")].sort_values("fti", ascending=False)
        e = o[(o.scope == scope) & (o.weights == "Equal")].sort_values("fti", ascending=False)
        md.append(f"- Group ranking ({scope}, n = {int(q.n.sum()):,}): FTI with entropy weights " + "; ".join(
            f"{r.group} {r.fti:.3f} (share {100*r.share_recent:.1f}%, change {100*r.growth:+.1f} pts)" for r in q.itertuples())
                  + f". Top group with equal weights: {e.group.iloc[0]}.")
    return o


def figures(d, g, fc, best):
    yr = d[d.group.notna()].groupby(["year", "group"]).size().unstack().reindex(columns=GROUPS).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for k in GROUPS:
        ax.plot(yr.index, yr[k], marker="o", ms=3, lw=2, color=GROUP_COLOR[k], label=k)
    ax.set_ylabel("Events per year"); ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Cyber events by attack group, 2014 to 2026 (2026 to September)")
    save(fig, "figC1_events_by_group_year")
    h = max(C.HORIZONS)
    f = fc[(fc.h == h) & (fc.model == best[h])].pivot(index="sector", columns="group", values="forecast").reindex(SECTORS)[GROUPS]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    left = np.zeros(len(f))
    for k in GROUPS:
        ax.barh(f.index, f[k], left=left, color=GROUP_COLOR[k], label=k, height=0.6, edgecolor=SURFACE, linewidth=1)
        left += f[k].values
    ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlabel(f"Projected events, next {h} months")
    ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title(f"Projected cyber events by sector and attack group ({best[h]})")
    save(fig, "figC2_projected_events_by_sector")


def run(_df=None):
    d = load()
    if d is None:
        return None
    rng = np.random.default_rng(C.SEED)
    T = C.TABLES
    assert len(d) >= 10000, f"expected at least 10,000 events, got {len(d)}"
    md = ["## Study 2: Cyber Events Database (University of Maryland)\n"]
    ok = d[d.group.notna() & d.sector.notna()]
    md.append(f"- Events: {len(d):,} ({d.date.min():%b %Y} to {d.date.max():%b %Y}), {d.country.nunique()} target countries. "
              f"Used for modelling after removing undetermined sector or subtype: {len(ok):,}. "
              f"Africa (AU members): {int(d.au.sum())}; ECOWAS: {int(d.ecowas.sum())}; Nigeria: {int((d.country == 'Nigeria').sum())}.")
    ct = pd.crosstab(ok.sector, ok.group).reindex(index=SECTORS, columns=GROUPS)
    ct.assign(n=ct.sum(1)).to_csv(T / "C02_sector_by_group.csv")
    d[d.group.notna()].groupby(["year", "group"]).size().unstack().reindex(columns=GROUPS).fillna(0).astype(int)\
        .to_csv(T / "C01_events_by_year_group.csv")
    reg = pd.crosstab(d[d.group.notna()].region, d[d.group.notna()].group, normalize="index").reindex(columns=GROUPS)
    (100 * reg).round(1).to_csv(T / "C03_region_by_group_pct.csv")
    share = (100 * ok.group.value_counts(normalize=True)).reindex(GROUPS)
    md.append("- Attack groups (all years): " + "; ".join(f"{k} {v:.1f}%" for k, v in share.items()))
    md.append("- Africa: " + "; ".join(f"{k} {v:.1f}%" for k, v in (100 * reg.loc['Africa']).items()) + ".")
    ng = d[(d.country == "Nigeria")]
    md.append(f"- Nigeria ({len(ng)} events): sectors " + ", ".join(f"{k} {v}" for k, v in ng.sector.value_counts().items())
              + "; groups " + ", ".join(f"{k} {v}" for k, v in ng.group.value_counts().items()) + ".")
    pd.DataFrame({"sector": ng.sector, "group": ng.group, "year": ng.year, "actor": ng.actor})\
        .value_counts().rename("n").reset_index().to_csv(T / "C03b_nigeria_events.csv", index=False)
    classification(d, rng, md)
    g, fc, best, score = forecast(d, md)
    group_ranking(d, md)
    figures(d, g, fc, best)
    return "\n".join(md) + "\n"
