# Cyber incidence in Nigeria

Analysis code for the study *Intelligent Analysis of Cyberattack Incidence Reporting in Nigeria*. It uses an online survey of 66 respondents, 38 of whom had experienced a cyber-attack or online fraud.

## What it covers

1. State of cyber incidents among respondents: prevalence, attack types, platforms, losses, reporting and barriers.
2. Experience by sector: sector comparisons with permutation chi-square tests.
3. Hybrid AHP-TOPSIS: combined AHP and entropy weights, a TOPSIS score for each incident, and a Future Threat Index for each attack group.
4. Random Forest vs CatBoost: model comparison on repeated cross-validation, then 3- and 6-month projections by sector.
5. Study 2 at scale: the University of Maryland Cyber Events Database (17,753 publicly reported cyber events, 2014 to 2026). Random Forest and CatBoost classify attack groups on a time-based split (train up to 2023, test 2024 to 2026), forecast monthly events by sector and attack group with a rolling backtest, and the hybrid ranking is repeated at group level for the world and for Africa.

All results are in `results/RESULTS.md`. Tables are in `results/tables` and figures in `results/figures`.

## Running

```bash
pip install -r requirements.txt
python run_all.py
```

Options:

- `--fast` runs a quick check with fewer resamples. Don't cite those numbers.
- `--only ahp` reruns a single step. The choices are descriptives, ahp, models and projections.
- `--gpu` trains CatBoost on GPU.

A full CPU run takes about 35 minutes on a 4-core laptop.

## Kaggle

Open `notebooks/kaggle_run.ipynb` in Kaggle, switch Internet on and run all cells. The notebook has a CPU cell and a GPU (T4 x2) cell.

## Data

`data/survey_deidentified.csv` contains no names, emails, phone numbers, states or free-text answers. `experiment/prepare.py` rebuilds it from the raw Google Forms export, which is kept outside the repository in `data/raw/`.

No synthetic records appear in any reported count. SMOTEN oversampling is applied only inside training folds, as one model variant.

## Study 2 data (Cyber Events Database)

The raw database is not included in this repository. Register for free at https://cissm.umd.edu/cissm-cyber-events-database, download the CSV and place it in `data/raw/cedb/`. The `cedb` step is skipped when the file is missing. Please cite Harry, C. and Gallagher, N. (2018), "Classifying cyber events", *Journal of Information Warfare*, 17(3), 17-31.

## Settings

All settings are in `experiment/config.py`:

- seed 42
- 2000 bootstraps
- 3-fold x 20 repeated cross-validation
- the 50/50 AHP and entropy weight mix
- the frequency-to-rate assumption: Rarely = 1, Occasionally = 3 and Frequently = 12 incidents per year

The AHP pairwise judgements are in `data/ahp_judgements.csv`.
