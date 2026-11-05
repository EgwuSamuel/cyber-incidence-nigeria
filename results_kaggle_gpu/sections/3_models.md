## Objective 4a: Random Forest vs CatBoost (repeated stratified 3-fold x 20, identical folds)

- **A: victim or not (profile)**: macro-F1 RF 0.546 (SMOTEN 0.546), CatBoost 0.554 (SMOTEN 0.562); baselines: majority 0.365, stratified random 0.490. Balanced accuracy RF 0.553, CatBoost 0.561.
  - Random Forest: label-permutation test p = 0.545 (observed macro-F1 0.470 vs null 95th percentile 0.625) -> **not better than chance**.
  - CatBoost: label-permutation test p = 0.059 (observed macro-F1 0.605 vs null 95th percentile 0.604) -> **not better than chance**.
  - CatBoost minus RF (none): +0.008 macro-F1, corrected t-test p = 0.899.
  - CatBoost minus RF (SMOTEN): +0.016 macro-F1, corrected t-test p = 0.818.
  - Most useful features (mean permutation importance, both models): gender (0.093), organization_group (0.017), occupation_group (0.005)
- **B1: attack group (profile)**: macro-F1 RF 0.228 (SMOTEN 0.215), CatBoost 0.283 (SMOTEN 0.245); baselines: majority 0.144, stratified random 0.232. Balanced accuracy RF 0.251, CatBoost 0.305.
  - Random Forest: label-permutation test p = 0.247 (observed macro-F1 0.297 vs null 95th percentile 0.367) -> **not better than chance**.
  - CatBoost: label-permutation test p = 0.257 (observed macro-F1 0.279 vs null 95th percentile 0.360) -> **not better than chance**.
  - CatBoost minus RF (none): +0.055 macro-F1, corrected t-test p = 0.442.
  - CatBoost minus RF (SMOTEN): +0.030 macro-F1, corrected t-test p = 0.654.
  - Most useful features (mean permutation importance, both models): organization_group (0.060), zone_group (0.018), gender (-0.002)
- **B2: attack group (profile + incident)**: macro-F1 RF 0.425 (SMOTEN 0.422), CatBoost 0.380 (SMOTEN 0.389); baselines: majority 0.144, stratified random 0.232. Balanced accuracy RF 0.471, CatBoost 0.409.
  - Random Forest: label-permutation test p = 0.010 (observed macro-F1 0.509 vs null 95th percentile 0.333) -> better than chance.
  - CatBoost: label-permutation test p = 0.020 (observed macro-F1 0.393 vs null 95th percentile 0.362) -> better than chance.
  - CatBoost minus RF (none): -0.044 macro-F1, corrected t-test p = 0.504.
  - CatBoost minus RF (SMOTEN): -0.033 macro-F1, corrected t-test p = 0.641.
  - Most useful features (mean permutation importance, both models): plat_computer_web (0.168), sensitive (0.036), zone_group (0.008)
