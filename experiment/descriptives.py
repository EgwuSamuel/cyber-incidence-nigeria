import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from . import config as C
from .plots import save, hbar, SERIES, GROUP_COLOR, INK2, SURFACE

RATINGS = {"rate_easy_interface": "Easy reporting interface", "rate_anonymous": "Anonymous reporting",
           "rate_ai_classification": "AI-based incident classification", "rate_realtime_alerts": "Real-time alerts",
           "rate_tracking": "Incident tracking", "rate_email_sms": "Email/SMS notifications",
           "rate_web_app": "Website/mobile app experience"}


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(k, n):
    return 100 * k / n if n else np.nan


def chi2_stat(tab):
    tab = np.asarray(tab, float)
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    m = exp > 0
    return (((tab - exp) ** 2)[m] / exp[m]).sum()


def assoc_test(a, b, rng):
    ok = pd.notna(a) & pd.notna(b)
    ai, la = pd.factorize(np.asarray(a)[ok]); bi, lb = pd.factorize(np.asarray(b)[ok])
    na, nb = len(la), len(lb)
    xtab = lambda x, y: np.bincount(x * nb + y, minlength=na * nb).reshape(na, nb)
    kdim = lambda t: min((t.sum(1) > 0).sum(), (t.sum(0) > 0).sum()) - 1
    tab = xtab(ai, bi)
    obs = chi2_stat(tab)
    n, k = tab.sum(), kdim(tab)
    null = np.array([chi2_stat(xtab(ai, rng.permutation(bi))) for _ in range(C.N_PERM_CHI2)])
    p = (1 + (null >= obs - 1e-9).sum()) / (1 + C.N_PERM_CHI2)
    v = np.sqrt(obs / (n * k)) if k > 0 else np.nan
    boots = []
    for _ in range(C.N_BOOT):
        i = rng.integers(0, len(ai), len(ai))
        t = xtab(ai[i], bi[i]); kk = kdim(t)
        boots.append(np.sqrt(chi2_stat(t) / (t.sum() * kk)) if kk > 0 else 0.0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(chi2=obs, p_perm=p, cramers_v=v, v_lo=lo, v_hi=hi, n=int(n))


def counts(s, n=None, name="category"):
    vc = s.value_counts()
    n = n or int(s.notna().sum())
    return pd.DataFrame({name: vc.index, "n": vc.values, "pct": (100 * vc.values / n).round(1)})


def multi_counts(s, name="option"):
    s = s.dropna()
    exploded = s.str.split(";").explode().str.strip()
    out = counts(exploded, n=len(s), name=name)
    out.attrs["respondents"] = len(s)
    return out


def rate_table(df, by, outcome):
    rows = []
    for g, d in df.groupby(by):
        k, n = int(d[outcome].sum()), int(d[outcome].notna().sum())
        lo, hi = wilson(k, n)
        rows.append({by: g, "n": n, "yes": k, "pct": round(pct(k, n), 1),
                     "ci_low": round(100 * lo, 1), "ci_high": round(100 * hi, 1)})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def run(df: pd.DataFrame):
    rng = np.random.default_rng(C.SEED)
    T = C.TABLES; T.mkdir(parents=True, exist_ok=True)
    vic = df[df.victim == 1]
    N, NV = len(df), len(vic)
    md = ["## Objectives 1-2: state of incidents and people's experience by sector\n"]

    prof = pd.concat([counts(df[c], name="category").assign(variable=c)
                      for c in ["gender", "zone", "occupation", "organization", "sector"]])
    prof[["variable", "category", "n", "pct"]].to_csv(T / "D01_sample_profile.csv", index=False)

    k = int(df.victim.sum()); lo, hi = wilson(k, N)
    md += [f"- Respondents: **{N}**; experienced a cyber-attack or online fraud: **{k} ({pct(k, N):.1f}%, "
           f"95% CI {100*lo:.1f}-{100*hi:.1f}%)**. Questions after Q9 were answered by victims only (n = {NV}).",
           f"- Gender: {(df.gender=='Male').sum()} male ({pct((df.gender=='Male').sum(), N):.1f}%), "
           f"{(df.gender=='Female').sum()} female. Zone: {(df.zone=='South West').sum()} of {N} "
           f"({pct((df.zone=='South West').sum(), N):.1f}%) live in the South West (sample is not nationally representative)."]
    tests = []
    for by in ["sector_group", "gender", "occupation_group"]:
        rt = rate_table(df, by, "victim"); rt.to_csv(T / f"D02_victimisation_by_{by}.csv", index=False)
        tests.append({"test": f"victim x {by}", **assoc_test(df[by], df.victim, rng)})
    rt = rate_table(df, "sector_group", "victim")
    md.append("- Victimisation by sector: " + "; ".join(
        f"{r.sector_group} {r.yes}/{r.n} ({r.pct}%)" for r in rt.itertuples()))

    at = counts(vic.attack_type, name="attack_type"); at.to_csv(T / "D03_attack_types.csv", index=False)
    ag = counts(vic.attack_group, name="attack_group"); ag.to_csv(T / "D04_attack_groups.csv", index=False)
    md.append("- Most common incident types (n = %d): " % NV + "; ".join(
        f"{r.attack_type} {r.n} ({r.pct}%)" for r in at.head(5).itertuples()))
    md.append("- Grouped: " + "; ".join(f"{r.attack_group} {r.n} ({r.pct}%)" for r in ag.itertuples()))
    yr = vic.incident_year.value_counts().sort_index()
    yr.rename_axis("year").reset_index(name="n").to_csv(T / "D05_incidents_by_year.csv", index=False)
    md.append("- Incident year: " + ", ".join(f"{int(y)}: {n}" for y, n in yr.items()))

    pl = multi_counts(vic.platforms, "platform"); pl.to_csv(T / "D06_platforms_affected.csv", index=False)
    md.append("- Platforms affected (multi-select, % of victims): " + "; ".join(
        f"{r.platform} {r.n} ({r.pct}%)" for r in pl.head(5).itertuples()))
    loss = counts(vic.loss_band, name="loss_band_ngn"); loss.to_csv(T / "D07_financial_loss.csv", index=False)
    s, d = int(vic.sensitive.sum()), int(vic.disrupted.sum())
    md.append(f"- Impact: sensitive data compromised in {s}/{NV} ({pct(s, NV):.1f}%); normal operations disrupted in "
              f"{d}/{NV} ({pct(d, NV):.1f}%); loss band: " + "; ".join(f"N{r.loss_band_ngn} {r.n}" for r in loss.itertuples()))

    r = int(vic.ever_reported.sum()); lo, hi = wilson(r, NV)
    md.append(f"- Ever reported a cyber incident: **{r}/{NV} ({pct(r, NV):.1f}%, 95% CI {100*lo:.1f}-{100*hi:.1f}%)**; "
              f"not reported: {NV - r} ({pct(NV - r, NV):.1f}%).")
    ch = counts(vic.report_channel, name="channel"); ch.to_csv(T / "D08_where_reported.csv", index=False)
    md.append("- Where reported (Q20, cleaned): " + "; ".join(f"{x.channel} {x.n}" for x in ch.itertuples()))
    rs = multi_counts(vic.loc[vic.ever_reported == 0, "reasons_not_reported"], "reason")
    rs.to_csv(T / "D09_reasons_not_reporting.csv", index=False)
    md.append(f"- Reasons for not reporting (multi-select, among the {rs.attrs['respondents']} who never reported): " + "; ".join(
        f"{x.reason} {x.n} ({x.pct}%)" for x in rs.head(6).itertuples()))
    aw = counts(vic.awareness, name="awareness"); aw.to_csv(T / "D10_awareness.csv", index=False)
    md.append("- Awareness of reporting channels: " + "; ".join(f"{x.awareness} {x.n} ({x.pct}%)" for x in aw.itertuples())
              + ". (The 77.84% 'did not know where to report' figure is from Ahmad et al. 2020, not this survey.)")
    sat = counts(vic.satisfaction, name="satisfaction"); sat.to_csv(T / "D11_satisfaction.csv", index=False)
    rep_sector = rate_table(vic, "sector_group", "ever_reported")
    rep_sector.to_csv(T / "D12_reporting_by_sector.csv", index=False)
    tests.append({"test": "ever_reported x sector_group", **assoc_test(vic.sector_group, vic.ever_reported, rng)})

    typed = vic[vic.attack_group.isin(C.ATTACK_GROUPS)]
    ct = pd.crosstab(typed.sector_group, typed.attack_group)[C.ATTACK_GROUPS]
    ct_pct = (100 * ct.div(ct.sum(1), axis=0)).round(1)
    ct.assign(n=ct.sum(1)).to_csv(T / "D13_sector_x_attack_group_counts.csv")
    ct_pct.to_csv(T / "D14_sector_x_attack_group_rowpct.csv")
    t = assoc_test(typed.sector_group, typed.attack_group, rng)
    tests.append({"test": "attack_group x sector_group", **t})
    md.append(f"- Attack group by sector (n = {len(typed)}): " + "; ".join(
        f"{s}: " + ", ".join(f"{g.split(' /')[0]} {ct_pct.loc[s, g]:.0f}%" for g in C.ATTACK_GROUPS)
        for s in ct.index) + f". Association: permutation chi-square p = {t['p_perm']:.3f}, Cramer's V = "
        f"{t['cramers_v']:.2f} (95% CI {t['v_lo']:.2f}-{t['v_hi']:.2f}).")

    rows = []
    for c, lab in RATINGS.items():
        x = vic[c].dropna().values
        b = [rng.choice(x, len(x)).mean() for _ in range(C.N_BOOT)]
        rows.append({"feature": lab, "n": len(x), "mean": round(x.mean(), 2), "median": np.median(x),
                     "ci_low": round(np.percentile(b, 2.5), 2), "ci_high": round(np.percentile(b, 97.5), 2),
                     "pct_4_or_5": round(pct((x >= 4).sum(), len(x)), 1)})
    rat = pd.DataFrame(rows).sort_values("mean", ascending=False); rat.to_csv(T / "D15_feature_ratings.csv", index=False)
    md.append("- Feature ratings (1-5, mean): " + "; ".join(f"{x.feature} {x.mean}" for x in rat.itertuples()))
    for col, name, fn in [("encouragers", "D16_encouragers", "encourager"), ("language", "D17_language", "language"),
                          ("pref_mode", "D18_preferred_mode", "mode"), ("pref_channel", "D19_preferred_channel", "channel")]:
        m = multi_counts(vic[col], fn); m.to_csv(T / f"{name}.csv", index=False)
        md.append(f"- {fn.capitalize()} (multi-select): " + "; ".join(f"{getattr(x, fn)} {x.n} ({x.pct}%)" for x in m.head(5).itertuples()))

    tt = pd.DataFrame(tests).round(4); tt.to_csv(T / "D20_association_tests.csv", index=False)
    md.append("- Association tests (permutation chi-square, %d shuffles): " % C.N_PERM_CHI2 + "; ".join(
        f"{x.test}: p = {x.p_perm:.3f}, V = {x.cramers_v:.2f}" for x in tt.itertuples()))

    figures(vic, at, ct_pct, ct, rs, rat, pl)
    return "\n".join(md) + "\n"


def figures(vic, at, ct_pct, ct, rs, rat, pl):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    hbar(ax, at.attack_type, at.n, xlabel=f"Respondents (n = {len(vic)} victims)")
    ax.set_title("Fig. 1  Cyber incident types reported by respondents"); save(fig, "fig01_attack_types")

    fig, ax = plt.subplots(figsize=(7, 2.8))
    left = np.zeros(len(ct_pct))
    for g in C.ATTACK_GROUPS:
        ax.barh(ct_pct.index, ct_pct[g], left=left, color=GROUP_COLOR[g], label=g,
                height=0.6, edgecolor=SURFACE, linewidth=2)
        for i, (l, v) in enumerate(zip(left, ct_pct[g])):
            if v >= 8:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color="#0b0b0b" if g == C.ATTACK_GROUPS[3] else "white")
        left += ct_pct[g].values
    ax.set_yticks(range(len(ct)))
    ax.set_yticklabels([f"{s} (n={n})" for s, n in zip(ct.index, ct.sum(1))])
    ax.set_xlim(0, 100); ax.set_xlabel("% of incidents in sector"); ax.grid(axis="y", visible=False)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.32))
    ax.set_title("Fig. 2  Attack group by sector"); save(fig, "fig02_attack_group_by_sector")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    hbar(ax, rs.reason.head(8), rs.n.head(8), color=SERIES[1], xlabel=f"Respondents ({rs.attrs['respondents']} answered)")
    ax.set_title("Fig. 3  Reasons for not reporting"); save(fig, "fig03_reasons_not_reporting")

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    y = np.arange(len(rat))[::-1]
    ax.errorbar(rat["mean"], y, xerr=[rat["mean"] - rat.ci_low, rat.ci_high - rat["mean"]], fmt="o",
                color=SERIES[0], ecolor=INK2, elinewidth=1, capsize=3, markersize=6)
    ax.set_yticks(y); ax.set_yticklabels(rat.feature); ax.set_xlim(1, 5)
    ax.set_xlabel("Mean importance rating (1-5) with 95% bootstrap CI")
    ax.grid(axis="y", visible=False)
    ax.set_title("Fig. 4  What respondents want from a reporting system"); save(fig, "fig04_feature_ratings")

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    hbar(ax, pl.platform, pl.n, color=SERIES[2], xlabel=f"Respondents (multi-select, n = {pl.attrs['respondents']})")
    ax.set_title("Fig. 5  Platforms and systems affected"); save(fig, "fig05_platforms_affected")
