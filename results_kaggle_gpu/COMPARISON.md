# Kaggle GPU run vs local CPU run

Run on Kaggle with GPU T4 x2 using `python run_all.py --gpu`. CatBoost trained on GPU; Random Forest always runs on CPU.

| Part | CPU vs GPU |
|---|---|
| Objectives 1-2 (descriptives, tests) | Identical (3 tables list tied rows in a different order) |
| Objective 3 (AHP-TOPSIS, FTI) | Identical |
| Random Forest results | Identical |
| Permutation tests | Identical (always run on CPU) |
| CatBoost results | Small differences, as expected from GPU training |

CatBoost macro-F1, CPU vs GPU:

| Task | CPU | GPU |
|---|---|---|
| A: victim or not | 0.603 | 0.554 |
| B1: attack group (profile) | 0.273 | 0.283 |
| B2: attack group (profile + incident) | 0.367 | 0.380 |

Projected chance of at least one incident in 3 months (CatBoost): Education 24.0% vs 24.3%, IT 23.5% vs 26.1%, other sectors 31.5% vs 33.7%. RF vs CatBoost agreement r = 0.97 (CPU) and 0.98 (GPU).

No conclusion changes: account takeover is the top threat in every sector, CatBoost and Random Forest are not significantly different, B2 beats chance, and profile-only models do not. The CPU results in `results/` remain the reference because they are exactly reproducible.
