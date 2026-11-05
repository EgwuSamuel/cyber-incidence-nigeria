import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTEN
from joblib import Parallel, delayed
from . import config as C
from .plots import save, MODEL_COLOR, NEUTRAL, INK2, SURFACE, BLUE_SEQ

warnings.filterwarnings("ignore", category=UserWarning)
MODELS = ["Random Forest", "CatBoost"]
AUGS = ["none", "SMOTEN"]
TASK_LABEL = {"A": "A: victim or not (profile)", "B1": "B1: attack group (profile)",
              "B2": "B2: attack group (profile + incident)"}
TASK_TITLE = {"A": "A: victim or not\n(profile features)", "B1": "B1: attack group\n(profile features)",
              "B2": "B2: attack group\n(profile + incident features)"}


def tasks(df):
    vic = df[(df.victim == 1) & df.attack_group.isin(C.ATTACK_GROUPS)]

    def as_str(d, cols):
        return d[cols].apply(lambda s: s.astype("Int64").astype(str) if pd.api.types.is_numeric_dtype(s) else s.astype(str))
    return {
        "A": (as_str(df, C.PROFILE_FEATURES), df.victim.map({1: "Victim", 0: "Not victim"})),
        "B1": (as_str(vic, C.PROFILE_FEATURES), vic.attack_group),
        "B2": (as_str(vic, C.PROFILE_FEATURES + C.INCIDENT_FEATURES), vic.attack_group),
    }


def make_model(name, weighted=True, gpu=None):
    if name == "Random Forest":
        p = dict(C.RF_PARAMS); p["class_weight"] = p["class_weight"] if weighted else None
        return make_pipeline(OneHotEncoder(handle_unknown="ignore"), RandomForestClassifier(**p))
    p = dict(C.CB_PARAMS)
    if not weighted:
        p.pop("auto_class_weights")
    if C.USE_GPU if gpu is None else gpu:
        p.update(C.GPU_PARAMS)
    return CatBoostClassifier(**p)


def cb_fit(model, X, y):
    cats = list(range(X.shape[1]))
    try:
        return model.fit(X, y, cat_features=cats)
    except Exception:
        params = model.get_params()
        if params.get("task_type") != "GPU":
            raise
        if not getattr(cb_fit, "warned", False):
            print("[models] CatBoost GPU fit failed, falling back to CPU")
            cb_fit.warned = True
        C.USE_GPU = False
        params = {k: v for k, v in params.items() if k not in C.GPU_PARAMS}
        return type(model)(**params).fit(X, y, cat_features=cats)


def fit(model, name, X, y):
    if name == "CatBoost":
        return cb_fit(model, X, y)
    model.fit(X, y)
    return model


def proba(model, X, classes):
    p = model.predict_proba(X)
    order = [list(model.classes_).index(c) for c in classes]
    return p[:, order]


def augment(X, y, seed):
    k = int(y.value_counts().min()) - 1
    if k < 1:
        return X, y
    sm = SMOTEN(k_neighbors=min(5, k), random_state=seed)
    Xr, yr = sm.fit_resample(X, y)
    return pd.DataFrame(Xr, columns=X.columns), pd.Series(yr)


def scores(y_true, proba, classes):
    pred = np.asarray(classes)[proba.argmax(1)]
    return dict(accuracy=accuracy_score(y_true, pred), balanced_accuracy=balanced_accuracy_score(y_true, pred),
                macro_f1=f1_score(y_true, pred, average="macro", zero_division=0),
                log_loss=log_loss(y_true, np.clip(proba, 1e-6, 1), labels=classes))


def corrected_ttest(d, n_train, n_test):
    d = np.asarray(d); J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return np.nan, np.nan
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    return t, 2 * stats.t.sf(abs(t), J - 1)


def perm_importance(model, name, X, y, classes, rng, reps=5):
    base = f1_score(y, np.asarray(classes)[proba(model, X, classes).argmax(1)], average="macro", zero_division=0)
    out = {}
    for col in X.columns:
        drops = []
        for _ in range(reps):
            Xp = X.copy(); Xp[col] = rng.permutation(Xp[col].values)
            p = np.asarray(classes)[proba(model, Xp, classes).argmax(1)]
            drops.append(base - f1_score(y, p, average="macro", zero_division=0))
        out[col] = np.mean(drops)
    return out


def cv_macro_f1(name, X, y, folds):
    s = []
    for tr, te in folds:
        m = fit(make_model(name, gpu=False), name, X.iloc[tr], y.iloc[tr])
        pred = np.asarray(m.classes_)[np.asarray(m.predict_proba(X.iloc[te])).argmax(1)]
        s.append(f1_score(y.iloc[te], pred, average="macro", zero_division=0))
    return np.mean(s)


def run(df: pd.DataFrame):
    T = C.TABLES; rng = np.random.default_rng(C.SEED)
    rows, cms, imps, perm_rows, ttests = [], {}, [], [], []
    for task, (X, y) in tasks(df).items():
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
        classes = sorted(y.unique())
        rskf = RepeatedStratifiedKFold(n_splits=C.CV_SPLITS, n_repeats=C.CV_REPEATS, random_state=C.SEED)
        folds = list(rskf.split(X, y))
        fold_sig = hash(tuple(tuple(te) for _, te in folds))
        per_fold = {}
        for f, (tr, te) in enumerate(folds):
            Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
            for strat in ["most_frequent", "stratified"]:
                d = DummyClassifier(strategy=strat if strat == "stratified" else "prior",
                                    random_state=C.SEED + f).fit(Xtr, ytr)
                p = d.predict_proba(Xte)
                if strat == "stratified":
                    pred = d.predict(Xte)
                    p = np.eye(len(classes))[[classes.index(c) for c in pred]] * 0.98 + 0.02 / len(classes)
                rows.append({"task": task, "fold": f, "model": f"Baseline: {strat.replace('_', ' ')}",
                             "augmentation": "none", **scores(yte, p, classes)})
            for aug in AUGS:
                Xa, ya = augment(Xtr, ytr, C.SEED + f) if aug == "SMOTEN" else (Xtr, ytr)
                for name in MODELS:
                    m = fit(make_model(name), name, Xa, ya)
                    p = proba(m, Xte, classes)
                    s = scores(yte, p, classes)
                    rows.append({"task": task, "fold": f, "model": name, "augmentation": aug, **s})
                    per_fold.setdefault((name, aug), []).append(s["macro_f1"])
                    if aug == "none":
                        pred = np.asarray(classes)[p.argmax(1)]
                        cms[(task, name)] = cms.get((task, name), 0) + confusion_matrix(yte, pred, labels=classes)
                        if f < C.CV_SPLITS * 5:
                            imp = perm_importance(m, name, Xte, yte, classes, rng, reps=3)
                            imps += [{"task": task, "model": name, "feature": k, "drop_macro_f1": v} for k, v in imp.items()]
        assert fold_sig == hash(tuple(tuple(te) for _, te in folds))
        n_te = len(X) / C.CV_SPLITS; n_tr = len(X) - n_te
        base = [r["macro_f1"] for r in rows if r["task"] == task and r["model"] == "Baseline: most frequent"]
        for aug in AUGS:
            t, p = corrected_ttest(np.array(per_fold[("CatBoost", aug)]) - per_fold[("Random Forest", aug)], n_tr, n_te)
            ttests.append({"task": task, "comparison": "CatBoost - Random Forest", "augmentation": aug,
                           "mean_diff_macro_f1": np.mean(per_fold[("CatBoost", aug)]) - np.mean(per_fold[("Random Forest", aug)]),
                           "t": t, "p": p})
            for name in MODELS:
                t, p = corrected_ttest(np.array(per_fold[(name, aug)]) - base, n_tr, n_te)
                ttests.append({"task": task, "comparison": f"{name} - majority baseline", "augmentation": aug,
                               "mean_diff_macro_f1": np.mean(per_fold[(name, aug)]) - np.mean(base), "t": t, "p": p})
        pfolds = list(StratifiedKFold(C.CV_SPLITS, shuffle=True, random_state=C.SEED).split(X, y))
        perms = [pd.Series(rng.permutation(y.values)) for _ in range(C.N_PERM_MODEL)]
        for name in MODELS:
            obs = cv_macro_f1(name, X, y, pfolds)
            null = Parallel(n_jobs=C.N_JOBS)(delayed(cv_macro_f1)(
                name, X, yp, list(StratifiedKFold(C.CV_SPLITS, shuffle=True, random_state=C.SEED).split(X, yp)))
                for yp in perms)
            perm_rows.append({"task": task, "model": name, "macro_f1_observed": obs,
                              "null_mean": np.mean(null), "null_95pct": np.percentile(null, 95),
                              "p_perm": (1 + np.sum(np.array(null) >= obs)) / (1 + C.N_PERM_MODEL)})
        print(f"[models] task {task} done ({len(X)} rows, {len(classes)} classes)")

    res = pd.DataFrame(rows)
    res.round(4).to_csv(T / "M01_cv_scores_per_fold.csv", index=False)
    summ = (res.groupby(["task", "model", "augmentation"])
            [["accuracy", "balanced_accuracy", "macro_f1", "log_loss"]].agg(["mean", "std"]).round(3))
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index(); summ.to_csv(T / "M02_cv_summary.csv", index=False)
    tt = pd.DataFrame(ttests).round(4); tt.to_csv(T / "M03_corrected_ttests.csv", index=False)
    pr = pd.DataFrame(perm_rows).round(4); pr.to_csv(T / "M04_permutation_tests.csv", index=False)
    im = (pd.DataFrame(imps).groupby(["task", "model", "feature"]).drop_macro_f1.mean().round(4)
          .reset_index().sort_values(["task", "model", "drop_macro_f1"], ascending=[True, True, False]))
    im.to_csv(T / "M05_permutation_importance.csv", index=False)
    for (task, name), cm in cms.items():
        classes = sorted(tasks(df)[task][1].unique())
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(T / f"M06_confusion_{task}_{name.replace(' ', '')}.csv")

    figures(summ, cms, im, df)

    md = ["## Objective 4a: Random Forest vs CatBoost (repeated stratified %d-fold x %d, identical folds)\n"
          % (C.CV_SPLITS, C.CV_REPEATS)]
    for task in TASK_LABEL:
        s = summ[summ.task == task].set_index(["model", "augmentation"])
        g = lambda m, a, k: s.loc[(m, a), f"{k}_mean"]
        md.append(f"- **{TASK_LABEL[task]}**: macro-F1 RF {g('Random Forest','none','macro_f1'):.3f} "
                  f"(SMOTEN {g('Random Forest','SMOTEN','macro_f1'):.3f}), CatBoost {g('CatBoost','none','macro_f1'):.3f} "
                  f"(SMOTEN {g('CatBoost','SMOTEN','macro_f1'):.3f}); baselines: majority "
                  f"{g('Baseline: most frequent','none','macro_f1'):.3f}, stratified random {g('Baseline: stratified','none','macro_f1'):.3f}. "
                  f"Balanced accuracy RF {g('Random Forest','none','balanced_accuracy'):.3f}, CatBoost {g('CatBoost','none','balanced_accuracy'):.3f}.")
        for r in pr[pr.task == task].itertuples():
            md.append(f"  - {r.model}: label-permutation test p = {r.p_perm:.3f} (observed macro-F1 {r.macro_f1_observed:.3f} vs "
                      f"null 95th percentile {r.null_95pct:.3f}) -> " + ("better than chance." if r.p_perm < 0.05 else "**not better than chance**."))
        for r in tt[(tt.task == task) & (tt.comparison == "CatBoost - Random Forest")].itertuples():
            md.append(f"  - CatBoost minus RF ({r.augmentation}): {r.mean_diff_macro_f1:+.3f} macro-F1, corrected t-test p = {r.p:.3f}.")
        top = im[(im.task == task)].groupby("feature").drop_macro_f1.mean().sort_values(ascending=False).head(3)
        md.append("  - Most useful features (mean permutation importance, both models): " + ", ".join(
            f"{k} ({v:.3f})" for k, v in top.items()))
    return "\n".join(md) + "\n"


def figures(summ, cms, im, df):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharex=True)
    for ax, task in zip(axes, TASK_LABEL):
        s = summ[summ.task == task].set_index(["model", "augmentation"])
        labels, vals, errs, cols = [], [], [], []
        for name in MODELS:
            for aug in AUGS:
                labels.append(f"{'RF' if name == 'Random Forest' else 'CatBoost'}{' + SMOTEN' if aug == 'SMOTEN' else ''}")
                vals.append(s.loc[(name, aug), "macro_f1_mean"]); errs.append(s.loc[(name, aug), "macro_f1_std"])
                cols.append(MODEL_COLOR[name] if aug == "none" else MODEL_COLOR[name] + "99")
        y = np.arange(len(labels))[::-1]
        ax.barh(y, vals, xerr=errs, color=cols, height=0.6, edgecolor=SURFACE, linewidth=1,
                error_kw=dict(ecolor=INK2, elinewidth=1, capsize=2))
        for b, ls, lab in [("Baseline: most frequent", "--", "majority"), ("Baseline: stratified", ":", "random")]:
            ax.axvline(s.loc[(b, "none"), "macro_f1_mean"], color=NEUTRAL, ls=ls, lw=1.2, label=f"{lab} baseline")
        ax.set_yticks(y); ax.set_yticklabels(labels); ax.grid(axis="y", visible=False)
        ax.set_title(TASK_TITLE[task], fontsize=9); ax.set_xlim(0, 1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.supxlabel("Macro-F1 (mean +/- SD over repeated folds)", y=0.09, fontsize=9, color=INK2)
    fig.suptitle("Fig. 8  Random Forest vs CatBoost on the real survey data", x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "fig08_model_comparison", tight=False)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7.5))
    for row, task in enumerate(["B1", "B2"]):
        for col, name in enumerate(MODELS):
            cm = cms[(task, name)].astype(float); cmn = cm / cm.sum(1, keepdims=True)
            classes = sorted(tasks(df)[task][1].unique())
            ax = axes[row, col]; ax.imshow(cmn, cmap=BLUE_SEQ, vmin=0, vmax=1); ax.grid(False)
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if cmn[i, j] > 0.55 else "#0b0b0b")
            short = [c.split(" /")[0] for c in classes]
            ax.set_xticks(range(len(classes))); ax.set_xticklabels(short, rotation=30, ha="right")
            ax.set_yticks(range(len(classes))); ax.set_yticklabels(short)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"{name}, task {task}", fontsize=9)
    fig.suptitle("Fig. 9  Out-of-fold confusion matrices (row-normalised, pooled over repeats)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    save(fig, "fig09_confusion_matrices")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    for ax, task in zip(axes, TASK_LABEL):
        t = im[im.task == task].pivot(index="feature", columns="model", values="drop_macro_f1")
        t = t.loc[t.mean(1).sort_values().index]
        y = np.arange(len(t))
        for k, name in enumerate(MODELS):
            ax.barh(y + (0.2 if k == 0 else -0.2), t[name], height=0.38, color=MODEL_COLOR[name], label=name,
                    edgecolor=SURFACE, linewidth=1)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(t.index); ax.grid(axis="y", visible=False)
        ax.set_title(TASK_TITLE[task], fontsize=9)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.supxlabel("Drop in macro-F1 when the feature is shuffled (held-out folds)", y=0.07, fontsize=9, color=INK2)
    fig.suptitle("Fig. 10  Permutation feature importance", x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig10_feature_importance", tight=False)
