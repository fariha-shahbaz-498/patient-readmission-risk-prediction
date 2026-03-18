import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

print(" PIPELINE STARTED")

# ======================
# LOAD DATA
# ======================
patients = pd.read_csv("data/raw/patients.csv")
encounters = pd.read_csv("data/raw/encounters.csv")
observations = pd.read_csv("data/raw/observations.csv")
medications = pd.read_csv("data/raw/medications.csv")
conditions = pd.read_csv("data/raw/conditions.csv")

print(" Data loaded")

# ======================
# SAFE DATETIME FIX 
# ======================
def safe_datetime(col):
    return pd.to_datetime(col, errors="coerce", format="mixed")

# Fix malformed strings first
observations["DATE"] = observations["DATE"].astype(str).str.replace("Z", "", regex=False)
observations["DATE"] = observations["DATE"].str.replace(" ", "T", regex=False)

encounters["START"] = safe_datetime(encounters["START"])
encounters["STOP"] = safe_datetime(encounters["STOP"])

observations["DATE"] = safe_datetime(observations["DATE"])
medications["START"] = safe_datetime(medications["START"])
conditions["START"] = safe_datetime(conditions["START"])

# Drop broken rows
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
# TARGET CREATION
# ======================
encounters["NEXT_START"] = encounters.groupby("PATIENT")["START"].shift(-1)
encounters["DAYS_TO_NEXT"] = (encounters["NEXT_START"] - encounters["START"]).dt.days
encounters["NEXT_EXISTS"] = encounters["DAYS_TO_NEXT"].notnull().astype(int)

# ======================
# TEMPORAL FEATURES 
# ======================
encounters["PREV_START"] = encounters.groupby("PATIENT")["START"].shift(1)
encounters["LAST_VISIT_GAP"] = (encounters["START"] - encounters["PREV_START"]).dt.days

# visit density last 30 days
encounters = encounters.set_index("START")
encounters["VISIT_LAST_30D"] = encounters.groupby("PATIENT")["PATIENT"]\
    .rolling("30D").count().values
encounters = encounters.reset_index()

# total visits
encounters["TOTAL_VISITS"] = encounters.groupby("PATIENT")["PATIENT"].transform("count")

print(" Targets + temporal features created")

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
# ENCODE TARGETS
# ======================
le_reason = LabelEncoder()
df["REASON_ENC"] = le_reason.fit_transform(df["REASONCODE"].astype(str))

# ======================
# FEATURES
# ======================
features = [
    "GENDER", "DESC_CODE", "VALUE",
    "MED_COUNT", "COND_COUNT",
    "LAST_VISIT_GAP", "VISIT_LAST_30D", "TOTAL_VISITS"
]

X = df[features]

y_cls = df["NEXT_EXISTS"]
y_reg = df["DAYS_TO_NEXT"].fillna(0)
y_reason = df["REASON_ENC"]

print(" Dataset ready:", X.shape)

# ======================
# SCALE
# ======================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================
# CLASSIFICATION
# ======================
print("--- CLASSIFICATION MODEL ---")

model_cls = LogisticRegression(max_iter=200, class_weight="balanced")
model_cls.fit(X_scaled, y_cls)

probs = model_cls.predict_proba(X_scaled)[:, 1]
brier = brier_score_loss(y_cls, probs)

print(f"Brier Score: {brier:.4f}")

# ======================
# CALIBRATION CURVE 
# ======================
prob_true, prob_pred = calibration_curve(y_cls, probs, n_bins=10)

os.makedirs("output", exist_ok=True)

plt.figure()
plt.plot(prob_pred, prob_true, marker='o')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("Calibration Curve")
plt.savefig("output/calibration_curve.png")
plt.close()

print(" Calibration curve saved")

# ======================
# REGRESSION
# ======================
print("--- REGRESSION MODEL ---")

model_reg = RandomForestRegressor(n_estimators=50, max_depth=8)
model_reg.fit(X, y_reg)

print("Regression trained")

# ======================
# TIME BUCKET MODEL
# ======================
print("--- TIME BUCKET MODEL ---")

def bucket(days):
    if days <= 30:
        return 0
    elif days <= 90:
        return 1
    else:
        return 2

y_bucket = y_reg.apply(bucket)

model_bucket = RandomForestClassifier()
model_bucket.fit(X, y_bucket)

print(" Time bucket model trained")

# ======================
# REASON MODEL
# ======================
print("--- REASON MODEL ---")

model_reason = RandomForestClassifier()
model_reason.fit(X, y_reason)

print(" Reason model trained")

# ======================
# PREDICTIONS
# ======================
df["PRED_PROB"] = probs
df["PRED_CLASS"] = model_cls.predict(X_scaled)
df["PRED_DAYS"] = model_reg.predict(X)
df["PRED_BUCKET"] = model_bucket.predict(X)
df["PRED_REASON"] = model_reason.predict(X)

# ======================
# RISK STRATIFICATION
# ======================
def risk(days):
    if days <= 30:
        return "HIGH"
    elif days <= 90:
        return "MEDIUM"
    else:
        return "LOW"

df["RISK"] = df["PRED_DAYS"].apply(risk)

print("--- RISK STRATIFICATION ---")
print(df["RISK"].value_counts())

# ======================
# SAVE OUTPUT
# ======================
df.to_csv("output/pipeline_output.csv", index=False)
import joblib
os.makedirs("models", exist_ok=True)

joblib.dump(model_cls, "models/model_cls.pkl")
joblib.dump(model_reg, "models/model_reg.pkl")
joblib.dump(model_bucket, "models/model_bucket.pkl")
joblib.dump(model_reason, "models/model_reason.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(le_reason, "models/label_encoder_reason.pkl")
print(" FINAL 10/10 PIPELINE COMPLETE")