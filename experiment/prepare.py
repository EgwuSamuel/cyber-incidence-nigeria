import re
import numpy as np
import pandas as pd
from . import config as C

RAW = {
    0: "timestamp", 1: "name", 2: "email", 3: "phone", 4: "gender", 5: "state", 6: "occupation",
    7: "occupation_other", 8: "organization", 9: "organization_other", 10: "sector",
    11: "victim_raw", 12: "others_victim", 13: "freq_self", 14: "freq_org", 15: "attack_type",
    16: "attack_type_other", 17: "incident_date", 18: "platforms", 19: "platform_other",
    20: "loss_band", 21: "sensitive_raw", 22: "disrupted_raw", 23: "awareness",
    24: "report_channel_raw", 25: "report_channel_other", 26: "reasons_not_reported",
    27: "reasons_other", 28: "language", 29: "pref_mode", 30: "pref_channel", 31: "encouragers",
    32: "ever_reported_raw", 33: "satisfaction", 34: "rate_easy_interface",
    35: "rate_anonymous", 36: "rate_ai_classification", 37: "rate_realtime_alerts",
    38: "rate_tracking", 39: "rate_email_sms", 40: "rate_web_app", 41: "comments",
}
IDENTIFYING = ["timestamp", "name", "email", "phone", "state", "occupation_other",
               "organization_other", "attack_type_other", "platform_other",
               "report_channel_raw", "report_channel_other", "reasons_other", "comments"]

ZONES = {
    "North Central": ["Benue", "Kogi", "Kwara", "Nasarawa", "Niger", "Plateau",
                      "Federal Capital Territory (FCT) Abuja"],
    "North East": ["Adamawa", "Bauchi", "Borno", "Gombe", "Taraba", "Yobe"],
    "North West": ["Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Sokoto", "Zamfara"],
    "South East": ["Abia", "Anambra", "Ebonyi", "Enugu", "Imo"],
    "South South": ["Akwa Ibom", "Bayelsa", "Cross River", "Delta", "Edo", "Rivers"],
    "South West": ["Ekiti", "Lagos", "Ogun", "Ondo", "Osun", "Oyo"],
}
STATE_TO_ZONE = {s: z for z, ss in ZONES.items() for s in ss}

LOSS = {
    "0 - 500,000": (1, 250_000), "500,000 - 2,000,000": (2, 1_250_000),
    "2,000,000 - 10,000,000": (3, 6_000_000), "10,000,000 - 50,000,000": (4, 30_000_000),
}

TYPE_GROUP = {
    "Account Hijacking": "Account takeover",
    "Social Media Account Compromise": "Account takeover",
    "Password / Credential Theft": "Account takeover",
    "Unauthorized Access": "Account takeover",
    "Financial Fraud or Online Banking Fraud": "Financial fraud / scams",
    "Identity Theft": "Financial fraud / scams",
    "Online Shopping Scam": "Financial fraud / scams",
    "Phishing": "Phishing / social engineering",
    "Social Engineering Attack": "Phishing / social engineering",
    "Malware Infection": "Malware / ransomware / DDoS",
    "Ransomware Attack": "Malware / ransomware / DDoS",
    "Distributed Denial-of-Service (DDoS) or Denial-of-Service (DoS) Attack": "Malware / ransomware / DDoS",
    "Lost or Stolen Device": "Other",
}
SHORT_TYPE = {
    "Financial Fraud or Online Banking Fraud": "Financial/online banking fraud",
    "Distributed Denial-of-Service (DDoS) or Denial-of-Service (DoS) Attack": "DDoS/DoS",
    "Password / Credential Theft": "Password/credential theft",
}


def group_from_text(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["malware", "ransom", "ddos", "denial", "defac"]):
        return "Malware / ransomware / DDoS"
    if any(k in t for k in ["phish", "social engineering", "credential harvest"]):
        return "Phishing / social engineering"
    if any(k in t for k in ["bank", "financial", "shopping", "scam", "identity"]):
        return "Financial fraud / scams"
    if any(k in t for k in ["account", "hijack", "password", "unauthor", "hack", "taken down"]):
        return "Account takeover"
    return "Other"


def channel(raw: str, other: str) -> str:
    t = f"{raw} {other}".strip().lower()
    if not t:
        return np.nan
    if re.fullmatch(r"(none|nil|non|nih|n/a|no one|not reported|did not report|i didn't report)\.?", raw.strip().lower()):
        return "Not reported"
    if "police" in t:
        return "Police cybercrime unit"
    if "ncc" in t:
        return "NCC-CSIRT"
    if "ngcert" in t:
        return "ngCERT"
    if "bank" in t:
        return "Bank"
    if any(k in t for k in ["facebook", "google", "mtn"]):
        return "Platform / service provider"
    if re.fullmatch(r"(none|nil|non|nih|n/a|no one|not reported|did not report|i didn't report)\.?", raw.strip().lower()):
        return "Not reported"
    return "Other"


def clean_multi(s: str) -> str:
    parts = [p.strip() for p in str(s).split(";") if p.strip()]
    return ";".join(parts) if parts else np.nan


def occupation_group(o):
    return o if o in ["Civil Servant", "Teacher/Lecturer", "IT Professional", "Student"] else "Other"


def organization_group(o):
    m = {"Educational Institution": "Educational institution", "Government/Public Sector": "Government/public",
         "Private Company": "Private company", "Self-Employed/Own Business": "Self-employed"}
    return m.get(o, "Other")


def main():
    raw = pd.read_csv(C.RAW_CSV, header=0, dtype=str, keep_default_na=False)
    raw.columns = [RAW[i] for i in range(raw.shape[1])]
    raw = raw.apply(lambda s: s.str.strip())
    df = pd.DataFrame(index=raw.index)

    df["respondent_id"] = [f"R{i + 1:03d}" for i in range(len(raw))]
    df["submission_month"] = pd.to_datetime(raw["timestamp"].str[:10], format="%Y/%m/%d").dt.strftime("%Y-%m")
    df["gender"] = raw["gender"]
    df["zone"] = raw["state"].map(STATE_TO_ZONE)
    df["zone_group"] = np.where(df["zone"] == "South West", "South West", "Other zones")
    df["occupation"] = raw["occupation"].replace({"Other (please specify)": "Other"})
    df["occupation_group"] = raw["occupation"].map(occupation_group)
    df["organization"] = raw["organization"].replace({"Other (please specify)": "Other"})
    df["organization_group"] = raw["organization"].map(organization_group)
    df["sector"] = raw["sector"]
    df["sector_group"] = raw["sector"].where(raw["sector"].isin(C.SECTOR_KEEP), "Other sectors")

    df["victim"] = (raw["victim_raw"] == "Yes").astype(int)
    v = df["victim"] == 1
    blank = lambda s: s.where(v & (s != ""), np.nan)

    df["others_victim"] = blank(raw["others_victim"])
    df["freq_self"] = blank(raw["freq_self"]).replace({"Never": "Rarely"})
    df["freq_org"] = blank(raw["freq_org"])
    df["attack_type"] = blank(raw["attack_type"])
    others = df["attack_type"] == "Others (please specify)"
    df["attack_group"] = df["attack_type"].map(TYPE_GROUP)
    df.loc[others, "attack_group"] = raw.loc[others, "attack_type_other"].map(group_from_text)
    df["attack_type"] = df["attack_type"].replace(SHORT_TYPE).replace({"Others (please specify)": "Other (specified)"})

    date = pd.to_datetime(blank(raw["incident_date"]), errors="coerce")
    df["incident_year"] = date.dt.year.astype("Int64")
    df["incident_month"] = date.dt.strftime("%Y-%m")

    df["platforms"] = blank(raw["platforms"]).map(clean_multi, na_action="ignore")
    plats = df["platforms"].fillna("")
    df["n_platforms"] = plats.map(lambda s: len(s.split(";")) if s else np.nan)
    df["plat_social"] = plats.str.contains("Social Media|E-commerce")
    df["plat_mobile"] = plats.str.contains("Mobile")
    df["plat_financial"] = plats.str.contains("Financial|Database")
    df["plat_computer_web"] = plats.str.contains("Computer|Website|Network|Cloud|Server|Email")
    for c in ["plat_social", "plat_mobile", "plat_financial", "plat_computer_web"]:
        df[c] = df[c].astype(int).where(v)

    df["loss_band"] = blank(raw["loss_band"])
    df["loss_ordinal"] = df["loss_band"].map(lambda b: LOSS[b][0], na_action="ignore")
    df["loss_mid_ngn"] = df["loss_band"].map(lambda b: LOSS[b][1], na_action="ignore")
    df["sensitive"] = blank(raw["sensitive_raw"]).map({"Yes": 1, "No": 0})
    df["disrupted"] = blank(raw["disrupted_raw"]).map({"Yes": 1, "No": 0})
    df["awareness"] = blank(raw["awareness"])
    df["report_channel"] = [channel(a, b) if vi else np.nan
                            for a, b, vi in zip(raw["report_channel_raw"], raw["report_channel_other"], v)]
    df["ever_reported"] = blank(raw["ever_reported_raw"]).map({"Yes": 1, "No": 0})
    df["not_reported"] = 1 - df["ever_reported"]
    df["reasons_not_reported"] = blank(raw["reasons_not_reported"]).map(clean_multi, na_action="ignore")
    for c in ["language", "pref_mode", "pref_channel", "encouragers"]:
        df[c] = blank(raw[c]).map(clean_multi, na_action="ignore")
    df["satisfaction"] = blank(raw["satisfaction"]).replace({"No satisfied": "Not satisfied"})
    for c in [c for c in raw.columns if c.startswith("rate_")]:
        df[c] = pd.to_numeric(blank(raw[c]), errors="coerce")

    check_no_identifiers(df)
    C.CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.CLEAN_CSV, index=False)
    print(f"[prepare] wrote {C.CLEAN_CSV.name}: {len(df)} rows, {int(df.victim.sum())} victims, "
          f"{df.shape[1]} columns (identifiers removed: {', '.join(IDENTIFYING)})")
    return df


def check_no_identifiers(df: pd.DataFrame):
    text = df.astype(str).to_csv(index=False)
    assert "@" not in text, "an email address survived de-identification"
    assert not re.search(r"\b0[789][01]\d{8}\b", text), "a phone number survived de-identification"
    assert not set(df.columns) & {"name", "email", "phone", "state", "timestamp"}, "identifier column survived"


if __name__ == "__main__":
    main()
