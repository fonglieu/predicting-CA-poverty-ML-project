# California Poverty Risk Prediction Using Machine Learning

## Project Overview
This project builds a machine learning model to predict whether an individual in California falls below the federal poverty threshold using demographic and employment characteristics from the American Community Survey (ACS) 2024 5-Year PUMS. The goal is to demonstrate how predictive analytics can support early identification of economically vulnerable populations and assist policymakers in allocating resources more effectively.

**Streamlit App:** https://predicting-ca-poverty-aziza-fong-marcela.streamlit.app/

---

## Research Questions

**Primary:** Can demographic characteristics, employment conditions, education level, and health insurance coverage predict whether an individual in California falls below the poverty threshold?

**Supporting:**
- How do education, employment status, hours worked, and wages influence poverty likelihood?
- How do age, marital status, citizenship, and insurance coverage create differences in risk across the population?

---

## Data
- **Source:** ACS 2024 5-Year PUMS, California (psam_p06.csv)
- **Raw file:** 285 variables across the full California population
- **Sample:** 25,000 records extracted using DuckDB
- **Final dataset:** 23,266 records, 26 predictors, 12.3% poverty class
- **Split:** 80/20 stratified train/test — 18,612 train / 4,654 test
- **Target variable:** `poverty_status` — binary indicator derived from POVPIP (1 = below poverty threshold, 0 = above)

---

## Features
Features were selected across four domains based on economic theory and the ACS codebook:

- **Education:** SCHL
- **Employment and income:** ESR, WKHP, WAGP, COW, INDP, OCCP
- **Demographics:** AGEP, SEX, RAC1P, MAR, NATIVITY, CIT, ENG, LANX, DIS, DRIVESP
- **Health insurance:** HICOV, HINS1 through HINS7

---

## Models
Three classification models were trained and compared:

| Model | Poverty Recall | ROC-AUC | False Negatives |
|---|---|---|---|
| Logistic Regression | 0.17 | 0.840 | 472 |
| Random Forest | 0.62 | 0.853 | 220 |
| XGBoost (Final) | 0.85 | 0.859 | 85 |

Overall accuracy was not used as the primary metric due to the 87.7% majority class. ROC-AUC and poverty recall were prioritized since the goal is to correctly identify individuals actually at risk.

**XGBoost** was selected as the final model. It was trained with `scale_pos_weight=7.13` to compensate for the 7:1 class imbalance and further optimized using RandomizedSearchCV with 30 iterations and 5-fold cross-validation scoring directly on poverty recall. After tuning, the model correctly identified 487 out of 572 individuals below the poverty threshold.

---

## Key Predictors
Based on XGBoost feature importance and SHAP analysis:
1. Employer-sponsored insurance (HINS1) — 18.6%
2. Wage income (WAGP) — 17.1%
3. Medicaid coverage (HINS4) — 15.7%
4. Overall insurance coverage (HICOV) — 8.0%
5. Marital status (MAR) — 6.8%

---

## Streamlit Deployment
An interactive prediction interface allows users to estimate individual poverty risk based on ACS demographic inputs.

- Accepts ACS demographic inputs
- Explains Census category codes
- Returns poverty classification and estimated probability

**Live app:** https://predicting-ca-poverty-aziza-fong-marcela.streamlit.app/

---

## Repository Structure

├── MLFinalProject.ipynb              # Main notebook (EDA, modeling, evaluation)
├── app.py                            # Streamlit app
├── requirements.txt                  # Project dependencies
├── ca_poverty_subset_final.csv       # Final cleaned dataset
├── xgb_poverty_model.pkl             # Trained XGBoost model
├── random_forest_poverty_model.pkl   # Trained Random Forest model
├── feature_names.pkl                 # Feature order for inference
├── ML Final Project Report.pdf       # Final written report
└── README.md

---

## Technologies
Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, joblib, DuckDB

---

## Limitations
- Model predicts poverty at a single point in time and does not capture poverty cycling over time
- 25,000-row sample would need validation against the full ACS dataset before real deployment
- False positive rate is high at the current threshold and would need adjustment for specific policy contexts
- Does not account for regional cost-of-living differences across California counties

---

## Social Impact
Accurate poverty prediction models help identify vulnerable populations earlier and support more effective allocation of housing, employment, and healthcare resources. This project demonstrates how machine learning can complement traditional survey methods to improve poverty monitoring and intervention planning.
