from fractions import Fraction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from . import config as C
from .plots import save, SERIES, GROUP_COLOR, INK2, NEUTRAL, SURFACE

CODES = list(C.CRITERIA)
COLS = [C.CRITERIA[c][0] for c in CODES]
LABELS = [C.CRITERIA[c][1] for c in CODES]
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}


def pairwise_matrices(path=C.AHP_CSV):
    j = pd.read_csv(path, dtype=str)
    mats = {}
    for eid, g in j.groupby("judge_id"):
        A = np.eye(len(CODES))
        for r in g.itertuples():
            a, b, v = CODES.index(r.criterion_a), CODES.index(r.criterion_b), float(Fraction(r.value))
            A[a, b], A[b, a] = v, 1 / v
        mats[eid] = A
    return mats


def ahp_weights(A):
    vals, vecs = np.linalg.eig(A)
    i = np.argmax(vals.real)
    w = np.abs(vecs[:, i].real); w /= w.sum()
    n = len(A); lam = vals[i].real
    ci = (lam - n) / (n - 1)
    return w, dict(lambda_max=lam, CI=ci, CR=ci / RI[n])


def aggregate(mats):
    return np.exp(np.mean([np.log(A) for A in mats.values()], axis=0))


def combine(w_ahp, w_ent):
    w = C.AHP_WEIGHT_SHARE * w_ahp + (1 - C.AHP_WEIGHT_SHARE) * w_ent
    return w / w.sum()


def entropy_weights(X):
    P = X / np.where(X.sum(0) == 0, 1, X.sum(0))
    with np.errstate(divide="ignore", invalid="ignore"):
        e = -np.nansum(np.where(P > 0, P * np.log(P), 0), axis=0) / np.log(len(X))
    d = 1 - e
    return d / d.sum()


def topsis(X, w):
    norm = np.sqrt((X ** 2).sum(0)); norm[norm == 0] = 1
    V = w * X / norm
    best, worst = V.max(0), V.min(0)
    s_plus = np.sqrt(((V - best) ** 2).sum(1))
    s_minus = np.sqrt(((V - worst) ** 2).sum(1))
    denom = s_plus + s_minus
    c = np.divide(s_minus, denom, out=np.zeros_like(denom), where=denom > 0)
    return c, s_plus, s_minus


def fti_table(groups, years, c):
    recent = np.isin(years, C.RECENT_YEARS)
    rows = []
    for g in C.ATTACK_GROUPS:
        m = groups == g
        p_r = (m & recent).sum() / max(recent.sum(), 1)
        p_e = (m & ~recent).sum() / max((~recent).sum(), 1)
        p_hat = max(0.0, p_r + (p_r - p_e))
        cbar = c[m].mean() if m.any() else 0.0
        rows.append(dict(attack_group=g, n=int(m.sum()), n_recent=int((m & recent).sum()),
                         n_earlier=int((m & ~recent).sum()), mean_C=cbar, p_recent=p_r,
                         p_earlier=p_e, p_hat=p_hat, FTI=p_hat * cbar))
    t = pd.DataFrame(rows)
    t["rank_severity"] = t.mean_C.rank(ascending=False, method="min").astype(int)
    t["rank_FTI"] = t.FTI.rank(ascending=False, method="min").astype(int)
    return t


def run(df: pd.DataFrame):
    T = C.TABLES; rng = np.random.default_rng(C.SEED)
    inc = df[(df.victim == 1) & df.attack_group.isin(C.ATTACK_GROUPS)].reset_index(drop=True)
    X = inc[COLS].astype(float).values
    assert not np.isnan(X).any(), "missing criterion values for an incident"
    groups, years = inc.attack_group.values, inc.incident_year.astype(int).values

    mats = pairwise_matrices()
    per_judge = [{"judge_id": k, **ahp_weights(A)[1]} for k, A in mats.items()]
    pd.DataFrame(per_judge).round(4).to_csv(T / "A02_ahp_consistency.csv", index=False)
    A = aggregate(mats)
    w_ahp, cons = ahp_weights(A)
    assert abs(w_ahp.sum() - 1) < 1e-9
    w_ent = entropy_weights(X)
    w = combine(w_ahp, w_ent)
    w_eq = np.full(len(COLS), 1 / len(COLS))
    W = {"Combined": w, "AHP only": w_ahp, "Entropy only": w_ent, "Equal": w_eq}
    pd.DataFrame({"code": CODES, "criterion": LABELS, "variable": COLS,
                  **{f"w_{k.replace(' ', '_')}": v.round(4) for k, v in W.items()}}).to_csv(T / "A01_criterion_weights.csv", index=False)
    pd.DataFrame(A, index=CODES, columns=CODES).round(3).to_csv(T / "A03_ahp_pairwise_matrix.csv")

    c, sp, sm = topsis(X, w)
    assert ((c >= 0) & (c <= 1)).all()
    out = inc[["respondent_id", "attack_type", "attack_group", "sector_group", "incident_year"] + COLS].copy()
    out["S_plus"], out["S_minus"], out["C_i"] = sp.round(4), sm.round(4), c.round(4)
    out.sort_values("C_i", ascending=False).to_csv(T / "A04_incident_topsis_scores.csv", index=False)
    ft = fti_table(groups, years, c)

    boot_fti, top = [], []
    for _ in range(C.N_BOOT):
        i = rng.integers(0, len(X), len(X))
        cb, _, _ = topsis(X[i], combine(w_ahp, entropy_weights(X[i])))
        t = fti_table(groups[i], years[i], cb)
        boot_fti.append(t.FTI.values); top.append(t.attack_group[t.FTI.idxmax()])
    boot_fti = np.array(boot_fti)
    ft["FTI_ci_low"], ft["FTI_ci_high"] = np.percentile(boot_fti, 2.5, 0), np.percentile(boot_fti, 97.5, 0)
    ft["P_rank1"] = [np.mean(np.array(top) == g) for g in ft.attack_group]
    ft.round(4).to_csv(T / "A05_group_severity_and_FTI.csv", index=False)

    base_rank = ft.set_index("attack_group").FTI

    def rank_under(wv):
        t = fti_table(groups, years, topsis(X, wv)[0]).set_index("attack_group")
        return t.FTI, t.FTI.idxmax(), round(spearmanr(t.FTI, base_rank)[0], 3)

    sch = []
    for k, wv in W.items():
        f, top_k, rho = rank_under(wv)
        sch.append({"scheme": k, **{g: round(f[g], 4) for g in C.ATTACK_GROUPS}, "top_threat": top_k,
                    "spearman_vs_combined": rho})
    for a in [0.25, 0.75]:
        f, top_k, rho = rank_under(a * w_ahp + (1 - a) * w_ent)
        sch.append({"scheme": f"Mix {int(100*a)}% AHP / {int(100*(1-a))}% entropy",
                    **{g: round(f[g], 4) for g in C.ATTACK_GROUPS}, "top_threat": top_k, "spearman_vs_combined": rho})
    sch = pd.DataFrame(sch); sch.to_csv(T / "A06_weighting_scheme_comparison.csv", index=False)
    sens = []
    for j, code in enumerate(CODES):
        for fct in (0.8, 1.2):
            wv = w.copy(); wv[j] *= fct; wv /= wv.sum()
            _, top_k, rho = rank_under(wv)
            sens.append({"criterion": code, "change": f"{int((fct - 1) * 100):+d}%", "top_threat": top_k,
                         "spearman_vs_base": rho})
    sens = pd.DataFrame(sens); sens.to_csv(T / "A07_weight_sensitivity.csv", index=False)

    figures(w_ahp, w_ent, w, ft)

    top_row = ft.sort_values("FTI", ascending=False).iloc[0]
    md = ["## Objective 3: hybrid AHP-TOPSIS and Future Threat Index\n",
          f"- Alternatives: {len(inc)} real incidents (4 attack groups; 1 lost-device case excluded). "
          f"Recent period = {C.RECENT_YEARS[0]}-{C.RECENT_YEARS[1]} ({np.isin(years, C.RECENT_YEARS).sum()} incidents), "
          f"earlier = up to {C.RECENT_YEARS[0]-1} ({(~np.isin(years, C.RECENT_YEARS)).sum()}).",
          "- Subjective AHP weights (research team's pairwise judgements, data/ahp_judgements.csv): " + "; ".join(
              f"{c} {l} {v:.3f}" for c, l, v in zip(CODES, LABELS, w_ahp))
          + f". lambda_max = {cons['lambda_max']:.3f}, CI = {cons['CI']:.4f}, **CR = {cons['CR']:.4f}** "
          + ("(acceptable, < 0.10)." if cons["CR"] < 0.10 else "(**too inconsistent, >= 0.10: revise judgements**)."),
          f"- Objective entropy weights (from the {len(inc)} incidents): " + "; ".join(f"{c} {v:.3f}" for c, v in zip(CODES, w_ent)),
          f"- **Combined weights** ({int(100*C.AHP_WEIGHT_SHARE)}% AHP + {int(100*(1-C.AHP_WEIGHT_SHARE))}% entropy), used for TOPSIS: "
          + "; ".join(f"{c} {v:.3f}" for c, v in zip(CODES, w)),
          "- Group severity (mean TOPSIS C_i) and FTI: " + "; ".join(
              f"{r.attack_group}: n = {r.n}, mean C = {r.mean_C:.3f}, share {100*r.p_earlier:.0f}% -> {100*r.p_recent:.0f}%, "
              f"P_hat = {r.p_hat:.3f}, **FTI = {r.FTI:.3f}** (95% CI {r.FTI_ci_low:.3f}-{r.FTI_ci_high:.3f}), "
              f"ranked #1 in {100*r.P_rank1:.0f}% of bootstraps" for r in ft.sort_values('FTI', ascending=False).itertuples()),
          f"- **Projected next-wave threat: {top_row.attack_group}** (highest FTI; top in {100*top_row.P_rank1:.0f}% of "
          f"{C.N_BOOT} bootstrap resamples, with entropy weights re-estimated on each).",
          "- Robustness, top threat under other weightings: " + "; ".join(
              f"{r.scheme} -> {r.top_threat} (Spearman {r.spearman_vs_combined})" for r in sch.itertuples())
          + f". +/-20% on each combined weight: top threat unchanged in {(sens.top_threat == top_row.attack_group).sum()}"
          f"/{len(sens)} cases; minimum Spearman {sens.spearman_vs_base.min():.2f}.",
          "- Note: the lower FTI CI bound is 0 for every group because P_hat is floored at 0 when a group's share falls "
          "in a resample. Read the CIs as wide, and use the rank-1 probability as the main uncertainty measure."]
    return "\n".join(md) + "\n"


def figures(w_ahp, w_ent, w, ft):
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    y = np.arange(len(CODES))[::-1] * 1.2
    for k, (vals, lab, col) in enumerate([(w_ahp, "AHP (research team)", SERIES[0]),
                                          (w_ent, "Entropy (survey data)", SERIES[1]),
                                          (w, "Combined (used)", SERIES[2])]):
        ax.barh(y + 0.32 - 0.32 * k, vals, height=0.3, color=col, label=lab, edgecolor=SURFACE, linewidth=1)
    for yi, v in zip(y, w):
        ax.text(v, yi - 0.32, f" {v:.2f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([f"{c} {l}" for c, l in zip(CODES, LABELS)])
    ax.set_xlabel("Weight"); ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=3, fontsize=7.5)
    fig.suptitle("Fig. 6  Criterion weights: subjective, objective and combined", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    save(fig, "fig06_criterion_weights")

    t = ft.sort_values("FTI")
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    y = np.arange(len(t))
    ax.barh(y, t.FTI, color=[GROUP_COLOR[g] for g in t.attack_group], height=0.55, edgecolor=SURFACE, linewidth=1)
    ax.errorbar(t.FTI, y, xerr=[t.FTI - t.FTI_ci_low, t.FTI_ci_high - t.FTI], fmt="none", ecolor=INK2,
                elinewidth=1, capsize=3)
    for yi, r in zip(y, t.itertuples()):
        ax.text(r.FTI_ci_high, yi, f"  FTI {r.FTI:.2f} | #1 in {100*r.P_rank1:.0f}% of bootstraps",
                va="center", fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(t.attack_group); ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(t.FTI_ci_high) * 1.9); ax.set_xlabel("Future Threat Index (95% bootstrap CI)")
    fig.suptitle("Fig. 7  Projected next-wave threat (AHP-TOPSIS Future Threat Index)", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    save(fig, "fig07_future_threat_index")
