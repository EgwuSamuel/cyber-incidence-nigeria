import argparse
import datetime as dt
import time
import pandas as pd
from experiment import config as C


def sanity(df):
    assert len(df) == 66 or args.allow_new_n, f"expected 66 respondents, got {len(df)} (use --allow-new-n after new data)"
    assert df.victim.sum() == 38 or args.allow_new_n, f"expected 38 victims, got {df.victim.sum()}"
    assert df.loc[df.victim == 0, ["attack_type", "loss_band", "ever_reported"]].isna().all().all(), \
        "non-victims must not have incident answers"
    from experiment.prepare import check_no_identifiers
    check_no_identifiers(df)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="train CatBoost on GPU (Kaggle T4 x2)")
    ap.add_argument("--fast", action="store_true", help="fewer resamples/permutations, for testing only")
    ap.add_argument("--allow-new-n", action="store_true", help="skip the n = 66 check once new responses are added")
    ap.add_argument("--only", nargs="*", choices=["descriptives", "ahp", "models", "projections"])
    args = ap.parse_args()
    if args.gpu:
        C.USE_GPU = True
    if args.fast:
        C.N_BOOT, C.N_PERM_CHI2, C.CV_REPEATS, C.N_PERM_MODEL = 200, 500, 3, 10

    t0 = time.time()
    if C.RAW_CSV.exists():
        from experiment import prepare
        prepare.main()
    df = pd.read_csv(C.CLEAN_CSV)
    sanity(df)
    C.TABLES.mkdir(parents=True, exist_ok=True); C.FIGURES.mkdir(parents=True, exist_ok=True)

    from experiment import descriptives, ahp_topsis, models, projections
    steps = {"descriptives": descriptives.run, "ahp": ahp_topsis.run, "models": models.run,
             "projections": projections.run}
    sec_dir = C.RESULTS / "sections"; sec_dir.mkdir(parents=True, exist_ok=True)
    for k, fn in steps.items():
        if args.only and k not in args.only:
            continue
        s = time.time()
        (sec_dir / f"{list(steps).index(k) + 1}_{k}.md").write_text(fn(df), encoding="utf-8")
        print(f"[run_all] {k} finished in {time.time() - s:.0f}s")

    sections = [p.read_text(encoding="utf-8") for p in sorted(sec_dir.glob("*.md"))]
    if len(sections) == len(steps):
        head = [f"# Experiment results\n",
                f"Generated {dt.date.today()} by `python run_all.py{' --fast' if args.fast else ''}` "
                f"(steps re-run individually with --only are merged in) "
                f"(seed {C.SEED}, CatBoost on {'GPU' if C.USE_GPU else 'CPU'}, {C.N_BOOT} bootstraps, {C.CV_SPLITS}-fold x {C.CV_REPEATS} repeated CV). "
                f"Every number below comes from `results/tables/`; figures are in `results/figures/`.\n",
                "**Data:** the original 66 survey responses (38 victims). No synthetic respondents are reported. "
                "Synthetic rows (SMOTEN) are generated only inside model training folds (Objective 4a) and never "
                "appear in any count, table, AHP-TOPSIS score or test fold.\n"]
        if args.fast:
            head.insert(1, "> **FAST MODE: smoke-test numbers only. Do not cite.**\n")
        (C.RESULTS / "RESULTS.md").write_text("\n".join(head + sections), encoding="utf-8")
        print(f"[run_all] wrote results/RESULTS.md")
    print(f"[run_all] done in {time.time() - t0:.0f}s")
