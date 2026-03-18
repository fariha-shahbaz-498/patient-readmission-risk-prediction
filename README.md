# patient-readmission-risk-prediction
# Patient Readmission Risk Prediction

An end-to-end Machine Learning system for predicting hospital readmission risk using healthcare data, with explainable AI (SHAP).

---

##  Features

-  Predict emergency readmission risk (90 days)
-  Predict time bucket of readmission
-  Classify readmission reason (Cardiac / Non-Cardiac)
-  Explain predictions using SHAP
-  Works on small and large datasets

---

## 📂 Project Structure
readmission_project/
│
├── data/raw/
│ ├── patients.csv
│ ├── encounters.csv
│ ├── conditions.csv
│ ├── medications.csv
│ └── observations.csv
│
├── src/
│ ├── run_pipeline.py
│ └── predict.py
│
├── models/
├── shap_outputs/
└── README.md

---

##  Installation

```bash
pip install pandas numpy scikit-learn matplotlib shap joblib
Run Pipeline
python src/run_pipeline.py
✔ Trains models
✔ Saves models in /models
✔ Generates SHAP plots

 Run Prediction
python src/predict.py
Example Output
EMERGENCY_RISK_SCORE: 0.78
EMERGENCY_RISK_LEVEL: HIGH

TIME_BUCKET_PREDICTION: 2

READMISSION_REASON: CARDIAC
REASON_CONFIDENCE: 0.73
### Features Used
Age

Gender

Total Conditions

Medications Count

Observations Count

Visit Count

Engineered Features:
Visit Intensity

Health Burden

Utilization Score

#### Model
Random Forest Classifier

Metrics:

AUROC

AUPRC

Brier Score

### Explainability
SHAP is used to:

Interpret predictions

Identify important features

Outputs saved in:

shap_outputs/
