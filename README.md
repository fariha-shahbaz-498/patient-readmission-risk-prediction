Predicting Emergency Readmission Risk from EHR Data
### Overview
This project develops a complete machine learning pipeline to predict patient emergency department readmissions using synthetic Electronic Health Record (EHR) data.

The system leverages temporal patient history, clinical measurements, and healthcare interactions to generate actionable predictions for proactive healthcare planning and resource allocation.

### Objectives
The project addresses three key predictive tasks:

Classification

Predict whether a patient will have a next emergency visit

Output: Yes / No

Regression

Predict the number of days until the next emergency visit

Risk Stratification

Categorize patients into:

  High Risk (0–30 days)

   Medium Risk (31–90 days)

   Low Risk (90+ days)

Additional Task

Predict reason code (condition associated with admission)

### Dataset
Synthetic healthcare dataset inspired by EHR systems.

Source Files:
patients.csv → Demographics

encounters.csv → Visit history (filtered to emergency only)

observations.csv → Clinical measurements

medications.csv → Prescriptions

conditions.csv → Diagnoses

 ### Data Processing & Feature Engineering
✔ Data Cleaning
Robust datetime parsing (handles inconsistent formats)

Missing value handling

Emergency encounter filtering

✔ Feature Engineering
Clinical Features
Encoded observation descriptions (DESC_CODE)

Numerical clinical values (VALUE)

Medication count (MED_COUNT)

Condition count (COND_COUNT)

  ### Temporal Features (Key Contribution)
LAST_VISIT_GAP → Days since previous visit

VISIT_LAST_30D → Visit density in last 30 days

TOTAL_VISITS → Patient visit frequency

These features capture patient history over time, improving predictive performance.

 ### Models
1. Classification Model
Logistic Regression

Target: NEXT_EXISTS

Balanced class weights

2. Regression Model
Random Forest Regressor

Target: DAYS_TO_NEXT

3. Time Bucket Model
Predicts:

0 → High risk

1 → Medium risk

2 → Low risk

4. Reason Prediction Model
Random Forest Classifier

Predicts admission reason code

 ### Model Evaluation
Classification
Brier Score: ~0.19

Calibration curve generated for probability validation

Regression
Predicts time-to-next visit (days)

✔ Calibration (Important in Healthcare)
Ensures predicted probabilities reflect real-world likelihoods

 ### Risk Stratification
Based on predicted time to next visit:

Risk Level	Days
High	0–30
Medium	31–90
Low	90+
### Conclusion
This project demonstrates how temporal EHR data can be transformed into meaningful predictive insights. By combining clinical features with time-based patterns, the system provides a robust and interpretable solution for healthcare risk prediction.
