import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "Worksheet.csv"
CLEAN_CSV = ROOT / "data" / "survey_deidentified.csv"
AHP_CSV = ROOT / "data" / "ahp_judgements.csv"
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

SEED = 42
N_BOOT = 2000
N_PERM_CHI2 = 10000
CV_SPLITS, CV_REPEATS = 3, 20
N_PERM_MODEL = 100

SURVEY_END = "2026-08-31"
RECENT_YEARS = (2025, 2026)

RATE_PER_YEAR = {"Rarely": 1, "Occasionally": 3, "Frequently": 12}
RATE_SENSITIVITY = {"low (x0.5)": 0.5, "base": 1.0, "high (x2)": 2.0}
HORIZONS = [3, 6]

SECTOR_KEEP = ["Education", "Information Technology"]
ATTACK_GROUPS = ["Account takeover", "Financial fraud / scams",
                 "Phishing / social engineering", "Malware / ransomware / DDoS"]

AHP_WEIGHT_SHARE = 0.5

CRITERIA = {
    "C1": ("loss_ordinal", "Financial loss"),
    "C2": ("n_platforms", "Platforms affected"),
    "C3": ("sensitive", "Sensitive data compromised"),
    "C4": ("disrupted", "Operations disrupted"),
    "C5": ("not_reported", "Not reported"),
}

PROFILE_FEATURES = ["sector_group", "gender", "occupation_group", "organization_group", "zone_group"]
INCIDENT_FEATURES = ["plat_social", "plat_mobile", "plat_financial", "plat_computer_web",
                     "loss_ordinal", "sensitive", "disrupted"]

RF_PARAMS = dict(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                 random_state=SEED, n_jobs=1)
CB_PARAMS = dict(iterations=500, depth=4, learning_rate=0.05, auto_class_weights="Balanced",
                 one_hot_max_size=10, random_seed=SEED, verbose=False, thread_count=1,
                 allow_writing_files=False)
N_JOBS = -1

CEDB_DIR = ROOT / "data" / "raw" / "cedb"
CEDB_TRAIN_END = 2023
CEDB_LAST_MONTH = "2026-08"
CEDB_BACKTEST_MONTHS = 23
N_PERM_CEDB = 50
USE_GPU = os.environ.get("CYBER_GPU") == "1"
GPU_PARAMS = dict(task_type="GPU", devices="0:1")
