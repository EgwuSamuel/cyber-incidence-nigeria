# Experiment results

Generated 2026-11-05 by `python run_all.py` (steps re-run individually with --only are merged in) (seed 42, CatBoost on CPU, 2000 bootstraps, 3-fold x 20 repeated CV). Every number below comes from `results/tables/`; figures are in `results/figures/`.

**Data:** the original 66 survey responses (38 victims). No synthetic respondents are reported. Synthetic rows (SMOTEN) are generated only inside model training folds (Objective 4a) and never appear in any count, table, AHP-TOPSIS score or test fold.

## Objectives 1-2: state of incidents and people's experience by sector

- Respondents: **66**; experienced a cyber-attack or online fraud: **38 (57.6%, 95% CI 45.6-68.8%)**. Questions after Q9 were answered by victims only (n = 38).
- Gender: 47 male (71.2%), 19 female. Zone: 47 of 66 (71.2%) live in the South West (sample is not nationally representative).
- Victimisation by sector: Education 17/28 (60.7%); Information Technology 11/23 (47.8%); Other sectors 10/15 (66.7%)
- Most common incident types (n = 38): Account Hijacking 6 (15.8%); Social Media Account Compromise 6 (15.8%); Phishing 5 (13.2%); Financial/online banking fraud 4 (10.5%); Identity Theft 3 (7.9%)
- Grouped: Account takeover 15 (39.5%); Financial fraud / scams 9 (23.7%); Malware / ransomware / DDoS 7 (18.4%); Phishing / social engineering 6 (15.8%); Other 1 (2.6%)
- Incident year: 2020: 1, 2022: 7, 2023: 7, 2024: 3, 2025: 6, 2026: 14
- Platforms affected (multi-select, % of victims): Social Media Platform 17 (44.7%); Mobile Application 12 (31.6%); Financial/Payment System 5 (13.2%); Website/Web Application 5 (13.2%); Computer/Laptop 5 (13.2%)
- Impact: sensitive data compromised in 18/38 (47.4%); normal operations disrupted in 27/38 (71.1%); loss band: N0 - 500,000 34; N500,000 - 2,000,000 2; N10,000,000 - 50,000,000 1; N2,000,000 - 10,000,000 1
- Ever reported a cyber incident: **16/38 (42.1%, 95% CI 27.9-57.8%)**; not reported: 22 (57.9%).
- Where reported (Q20, cleaned): Not reported 18; Bank 6; Police cybercrime unit 6; Platform / service provider 3; Other 2; NCC-CSIRT 2; ngCERT 1
- Reasons for not reporting (multi-select, among the 22 who never reported): Lack of awareness of reporting channels 8 (36.4%); Incident was not serious enough 4 (18.2%); Other (please specify) 3 (13.6%); Reporting was not mandatory 2 (9.1%); Belief that no action would be taken 2 (9.1%); Lack of evidence or information about the incident 2 (9.1%)
- Awareness of reporting channels: Fairly Aware 18 (47.4%); Aware 12 (31.6%); Not Aware 8 (21.1%). (The 77.84% 'did not know where to report' figure is from Ahmad et al. 2020, not this survey.)
- Attack group by sector (n = 37): Education: Account takeover 35%, Financial fraud 29%, Phishing 18%, Malware 18%; Information Technology: Account takeover 46%, Financial fraud 9%, Phishing 9%, Malware 36%; Other sectors: Account takeover 44%, Financial fraud 33%, Phishing 22%, Malware 0%. Association: permutation chi-square p = 0.480, Cramer's V = 0.28 (95% CI 0.23-0.54).
- Feature ratings (1-5, mean): Website/mobile app experience 3.55; Email/SMS notifications 3.53; Easy reporting interface 3.53; Incident tracking 3.5; Real-time alerts 3.5; AI-based incident classification 2.95; Anonymous reporting 2.82
- Encourager (multi-select): Feedback on the outcome 24 (63.2%); Assurance of confidentiality 21 (55.3%); A simple reporting process 16 (42.1%)
- Language (multi-select): English 35 (92.1%); Yoruba 2 (5.3%); Hausa 2 (5.3%); Pidgin 1 (2.6%)
- Mode (multi-select): Evidence-based(Screenshot, Log files, Document) 21 (55.3%); Text-based 18 (47.4%); Voice-based 13 (34.2%); Video-based 3 (7.9%)
- Channel (multi-select): Email 23 (60.5%); WhatsApp 16 (42.1%); Phone Call 15 (39.5%); In-person 8 (21.1%); SMS 4 (10.5%)
- Association tests (permutation chi-square, 10000 shuffles): victim x sector_group: p = 0.455, V = 0.15; victim x gender: p = 0.012, V = 0.33; victim x occupation_group: p = 0.661, V = 0.20; ever_reported x sector_group: p = 0.276, V = 0.28; attack_group x sector_group: p = 0.480, V = 0.28

## Objective 3: hybrid AHP-TOPSIS and Future Threat Index

- Alternatives: 37 real incidents (4 attack groups; 1 lost-device case excluded). Recent period = 2025-2026 (20 incidents), earlier = up to 2024 (17).
- Subjective AHP weights (research team's pairwise judgements, data/ahp_judgements.csv): C1 Financial loss 0.313; C2 Platforms affected 0.099; C3 Sensitive data compromised 0.313; C4 Operations disrupted 0.176; C5 Not reported 0.099. lambda_max = 5.013, CI = 0.0033, **CR = 0.0030** (acceptable, < 0.10).
- Objective entropy weights (from the 37 incidents): C1 0.047; C2 0.094; C3 0.377; C4 0.185; C5 0.296
- **Combined weights** (50% AHP + 50% entropy), used for TOPSIS: C1 0.180; C2 0.096; C3 0.345; C4 0.180; C5 0.197
- Group severity (mean TOPSIS C_i) and FTI: Account takeover: n = 15, mean C = 0.467, share 41% -> 40%, P_hat = 0.388, **FTI = 0.181** (95% CI 0.000-0.443), ranked #1 in 61% of bootstraps; Financial fraud / scams: n = 9, mean C = 0.297, share 24% -> 25%, P_hat = 0.265, **FTI = 0.079** (95% CI 0.000-0.220), ranked #1 in 14% of bootstraps; Malware / ransomware / DDoS: n = 7, mean C = 0.295, share 18% -> 20%, P_hat = 0.224, **FTI = 0.066** (95% CI 0.000-0.217), ranked #1 in 11% of bootstraps; Phishing / social engineering: n = 6, mean C = 0.472, share 18% -> 15%, P_hat = 0.124, **FTI = 0.058** (95% CI 0.000-0.262), ranked #1 in 14% of bootstraps
- **Projected next-wave threat: Account takeover** (highest FTI; top in 61% of 2000 bootstrap resamples, with entropy weights re-estimated on each).
- Robustness, top threat under other weightings: Combined -> Account takeover (Spearman 1.0); AHP only -> Account takeover (Spearman 1.0); Entropy only -> Account takeover (Spearman 1.0); Equal -> Account takeover (Spearman 0.8); Mix 25% AHP / 75% entropy -> Account takeover (Spearman 1.0); Mix 75% AHP / 25% entropy -> Account takeover (Spearman 1.0). +/-20% on each combined weight: top threat unchanged in 10/10 cases; minimum Spearman 1.00.
- Note: the lower FTI CI bound is 0 for every group because P_hat is floored at 0 when a group's share falls in a resample. Read the CIs as wide, and use the rank-1 probability as the main uncertainty measure.

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

## Objective 4b: projected incidents by sector (next 3 and 6 months)

- Assumed incidents/year: {'Rarely': 1, 'Occasionally': 3, 'Frequently': 12} ('Never' -> 'Rarely' for victims). Projections are scenario estimates for people like the respondents, not a time-series forecast for Nigeria as a whole.
- **Education** (n = 28): Random Forest: P(any incident) 24.0% in 3 months / 42.0% in 6; top type Account takeover (10.9 per 100 people per 3 months, 95% CI 9.2-12.7) | CatBoost: P(any incident) 24.0% in 3 months / 40.7% in 6; top type Account takeover (12.1 per 100 people per 3 months, 95% CI 8.4-16.6) | observed in last 12 months: 7.1 per 100 people per 3 months.
- **Information Technology** (n = 23): Random Forest: P(any incident) 27.6% in 3 months / 46.9% in 6; top type Account takeover (15.5 per 100 people per 3 months, 95% CI 11.5-20.3) | CatBoost: P(any incident) 23.5% in 3 months / 37.8% in 6; top type Account takeover (18.3 per 100 people per 3 months, 95% CI 8.7-33.8) | observed in last 12 months: 5.4 per 100 people per 3 months.
- **Other sectors** (n = 15): Random Forest: P(any incident) 35.1% in 3 months / 57.1% in 6; top type Account takeover (19.5 per 100 people per 3 months, 95% CI 13.3-27.1) | CatBoost: P(any incident) 31.5% in 3 months / 50.3% in 6; top type Account takeover (19.1 per 100 people per 3 months, 95% CI 10.8-28.5) | observed in last 12 months: 5.0 per 100 people per 3 months.
- RF vs CatBoost agreement on the 3-month sector x type projections: Pearson r = 0.97.
- Rate-assumption sensitivity (P(any incident) in 3 months, low / base / high): CatBoost Education: 13.1 / 24.0 / 40.7%; CatBoost Information Technology: 13.5 / 23.5 / 37.8%; CatBoost Other sectors: 17.9 / 31.5 / 50.3%; Random Forest Education: 12.9 / 24.0 / 42.0%; Random Forest Information Technology: 15.1 / 27.6 / 46.9%; Random Forest Other sectors: 19.6 / 35.1 / 57.1%
