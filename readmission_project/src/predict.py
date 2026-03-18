import pandas as pd
import numpy as np
import joblib
import os

print(" PREDICTION STARTED")

# ======================
# LOAD MODELS
# ======================
model_cls = joblib.load("models/model_cls.pkl")
model_reg = joblib.load("models/model_reg.pkl")
model_bucket = joblib.load("models/model_bucket.pkl")
model_reason = joblib.load("models/model_reason.pkl")
scaler = joblib.load("models/scaler.pkl")
le_reason = joblib.load("models/label_encoder_reason.pkl")

# ======================
# LOAD DATA
# ======================
patients = pd.read_csv("data/raw/patients.csv")
encounters = pd.read_csv("data/raw/encounters.csv")
observations = pd.read_csv("data/raw/observations.csv")
medications = pd.read_csv("data/raw/medications.csv")
conditions = pd.read_csv("data/raw/conditions.csv")

# ======================
# SAFE DATETIME
# ======================
def safe_datetime(col):
    return pd.to_datetime(col, errors="coerce", format="mixed")

observations["DATE"] = observations["DATE"].astype(str).str.replace("Z", "", regex=False)
observations["DATE"] = observations["DATE"].str.replace(" ", "T", regex=False)

encounters["START"] = safe_datetime(encounters["START"])
encounters["STOP"] = safe_datetime(encounters["STOP"])
observations["DATE"] = safe_datetime(observations["DATE"])
medications["START"] = safe_datetime(medications["START"])
conditions["START"] = safe_datetime(conditions["START"])

# Drop invalid
encounters = encounters.dropna(subset=["START"])
observations = observations.dropna(subset=["DATE"])
medications = medications.dropna(subset=["START"])
conditions = conditions.dropna(subset=["START"])

# ======================
# FILTER EMERGENCY
# ======================
encounters = encounters[encounters["ENCOUNTERCLASS"] == "emergency"]
encounters = encounters.sort_values(["PATIENT", "START"])

# ======================
# TEMPORAL FEATURES
# ======================
encounters["PREV_START"] = encounters.groupby("PATIENT")["START"].shift(1)
encounters["LAST_VISIT_GAP"] = (encounters["START"] - encounters["PREV_START"]).dt.days

encounters = encounters.set_index("START")
encounters["VISIT_LAST_30D"] = encounters.groupby("PATIENT")["PATIENT"]\
    .rolling("30D").count().values
encounters = encounters.reset_index()

encounters["TOTAL_VISITS"] = encounters.groupby("PATIENT")["PATIENT"].transform("count")

# ======================
# FEATURE ENGINEERING
# ======================

# Observations
observations["DESC_CODE"] = observations["DESCRIPTION"].astype("category").cat.codes
observations["VALUE"] = pd.to_numeric(observations["VALUE"], errors="coerce")

obs_feat = observations.groupby("PATIENT").agg({
    "DESC_CODE": "mean",
    "VALUE": "mean"
}).reset_index()

# Medications
med_feat = medications.groupby("PATIENT").agg({
    "CODE": "count"
}).reset_index().rename(columns={"CODE": "MED_COUNT"})

# Conditions
cond_feat = conditions.groupby("PATIENT").agg({
    "CODE": "count"
}).reset_index().rename(columns={"CODE": "COND_COUNT"})

# Patients
patients["GENDER"] = patients["GENDER"].astype("category").cat.codes

# ======================
# MERGE
# ======================
df = encounters.merge(obs_feat, on="PATIENT", how="left")
df = df.merge(med_feat, on="PATIENT", how="left")
df = df.merge(cond_feat, on="PATIENT", how="left")
df = df.merge(patients[["Id", "GENDER"]], left_on="PATIENT", right_on="Id", how="left")

df = df.fillna(0)

# ======================
# FEATURES
# ======================
features = [
    "GENDER", "DESC_CODE", "VALUE",
    "MED_COUNT", "COND_COUNT",
    "LAST_VISIT_GAP", "VISIT_LAST_30D", "TOTAL_VISITS"
]

X = df[features]

# Scale
X_scaled = scaler.transform(X)

# ======================
# PREDICTIONS
# ======================
df["PRED_PROB"] = model_cls.predict_proba(X_scaled)[:, 1]
df["PRED_CLASS"] = model_cls.predict(X_scaled)
df["PRED_DAYS"] = model_reg.predict(X)
df["PRED_BUCKET"] = model_bucket.predict(X)

reason_encoded = model_reason.predict(X)
df["PRED_REASON"] = le_reason.inverse_transform(reason_encoded)

# ======================
# RISK
# ======================
def risk(days):
    if days <= 30:
        return "HIGH"
    elif days <= 90:
        return "MEDIUM"
    else:
        return "LOW"

df["RISK"] = df["PRED_DAYS"].apply(risk)

# ======================
# SAVE
# ======================
os.makedirs("output", exist_ok=True)
df.to_csv("output/predictions.csv", index=False)

print(" Predictions saved: output/predictions.csv")
print(" PREDICTION COMPLETE")