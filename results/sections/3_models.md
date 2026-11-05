## Objective 4a: Random Forest vs CatBoost (repeated stratified 3-fold x 20, identical folds)

- **A: victim or not (profile)**: macro-F1 RF 0.546 (SMOTEN 0.546), CatBoost 0.603 (SMOTEN 0.604); baselines: majority 0.365, stratified random 0.490. Balanced accuracy RF 0.553, CatBoost 0.608.
  - Random Forest: label-permutation test p = 0.545 (observed macro-F1 0.470 vs null 95th percentile 0.625) -> **not better than chance**.
  - CatBoost: label-permutation test p = 0.059 (observed macro-F1 0.605 vs null 95th percentile 0.604) -> **not better than chance**.
  - CatBoost minus RF (none): +0.057 macro-F1, corrected t-test p = 0.473.
  - CatBoost minus RF (SMOTEN): +0.058 macro-F1, corrected t-test p = 0.448.
  - Most useful features (mean permutation importance, both models): gender (0.116), organization_group (0.036), sector_group (0.032)
- **B1: attack group (profile)**: macro-F1 RF 0.228 (SMOTEN 0.215), CatBoost 0.273 (SMOTEN 0.254); baselines: majority 0.144, stratified random 0.232. Balanced accuracy RF 0.251, CatBoost 0.297.
  - Random Forest: label-permutation test p = 0.247 (observed macro-F1 0.297 vs null 95th percentile 0.367) -> **not better than chance**.
  - CatBoost: label-permutation test p = 0.257 (observed macro-F1 0.279 vs null 95th percentile 0.360) -> **not better than chance**.
  - CatBoost minus RF (none): +0.045 macro-F1, corrected t-test p = 0.498.
  - CatBoost minus RF (SMOTEN): +0.040 macro-F1, corrected t-test p = 0.514.
  - Most useful features (mean permutation importance, both models): organization_group (0.052), zone_group (0.016), gender (-0.004)
- **B2: attack group (profile + incident)**: macro-F1 RF 0.425 (SMOTEN 0.422), CatBoost 0.367 (SMOTEN 0.370); baselines: majority 0.144, stratified random 0.232. Balanced accuracy RF 0.471, CatBoost 0.392.
  - Random Forest: label-permutation test p = 0.010 (observed macro-F1 0.509 vs null 95th percentile 0.333) -> better than chance.
  - CatBoost: label-permutation test p = 0.020 (observed macro-F1 0.393 vs null 95th percentile 0.362) -> better than chance.
  - CatBoost minus RF (none): -0.057 macro-F1, corrected t-test p = 0.416.
  - CatBoost minus RF (SMOTEN): -0.052 macro-F1, corrected t-test p = 0.434.
  - Most useful features (mean permutation importance, both models): plat_computer_web (0.168), sensitive (0.040), occupation_group (0.006)
