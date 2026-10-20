# Student Dropout Risk Prediction

**Name:** Saurab Thapa  
**Register number:** URK24CS9026

## Project idea

This project predicts whether a student may drop out. It also compares different models, checks the quality of predicted probabilities, explains the results, and groups students into Low, Medium, and High risk tiers.

## Verified project status

This project was verified locally by creating a virtual environment from `requirements.txt`, installing the dependencies, and launching the Streamlit dashboard successfully from the project root. The app loads the dataset, trains the calibrated logistic regression model, and serves the dashboard at `http://localhost:8501` without import or runtime errors.

The latest verified model results in `results/tables/model_comparison.csv` and `results/tables/hybrid_comparison.csv` are consistent with the README summary: the best recall model remains the class-weighted Logistic Regression (ROC-AUC 0.934, PR-AUC 0.900), and the hybrid risk engine improves to ROC-AUC 0.936 and PR-AUC 0.904 while preserving a lower Brier score than the ML-only baseline.

## Dataset

The project uses the UCI student dropout dataset with 4,424 records. The original target has three outcomes: Dropout, Enrolled, and Graduate. This project creates a binary target where Dropout is 1 and the other outcomes are 0.

## Folder structure

```text
student-dropout-risk/
├── README.md
├── requirements.txt
├── .gitignore
├── data/raw/data.csv
├── notebooks/
│   └── 05_hybrid_risk_engine.ipynb
├── src/
│   └── custom_algorithms.py
├── results/figures/
│   ├── fig15_hybrid_comparison.png
│   ├── fig16_afsi_sensitivity.png
│   ├── fig17_recommendation_frequency.png
│   └── fig18_hybrid_tier_dropout.png
└── results/tables/
```

## Methodology

1. Explore the data, class balance, group differences, correlations, and statistical tests.
2. Create student profiles using scaling, PCA, and KMeans.
3. Add approval rates, grade change, and financial risk features.
4. Compare Logistic Regression, Random Forest, and XGBoost with class weights and SMOTE.
5. Calibrate probabilities, explain predictions with SHAP, and create risk tiers.
6. Add a transparent hybrid risk engine and a model-based counterfactual recommender.

## Hybrid risk engine

Notebook 05 rebuilds the final Logistic Regression with balanced class weights and
isotonic calibration using the same seed-42 stratified 80/20 split. The
**Academic-Financial Stress Index (AFSI)** is a 0-100 score based on first- and
second-semester approval rates, a financial-risk flag, and grade decline. A
training-only five-fold PR-AUC search chooses the hybrid weight (`alpha`) before
one test-set evaluation. The notebook compares ML-only, AFSI-only, hybrid, and
Random Forest scores, checks AFSI-weight sensitivity, and compares Low, Medium,
and High tiers.

The recommender greedily tests only actionable changes (tuition status, debtor
status, approved units, and second-semester grade) for up to three steps. These
are model-based associations, not causal claims. They are decision support only
and must not be used for punitive decisions or to change protected/non-actionable
attributes.

## Results

| Model | Imbalance strategy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---:|---:|---:|---:|---:|
| Logistic Regression | Class weights | 0.789 | 0.845 | 0.816 | 0.934 | 0.900 |
| Logistic Regression | SMOTE | 0.795 | 0.835 | 0.814 | 0.932 | 0.898 |
| Random Forest | Class weights | 0.794 | 0.827 | 0.810 | 0.930 | 0.900 |
| XGBoost | SMOTE | 0.851 | 0.824 | 0.837 | 0.939 | 0.913 |

The full comparison is saved in `results/tables/model_comparison.csv`.

The new hybrid comparison is saved in `results/tables/hybrid_comparison.csv`.
In the test run, the hybrid achieved ROC-AUC 0.936 and PR-AUC 0.904, compared
with 0.935 and 0.900 for ML-only. The hybrid's Brier score was 0.0839 versus
0.0842 for ML-only. This small improvement should not be treated as proof that
the hybrid will generalise beyond this dataset.

The final model is **Logistic Regression with class weights**. It was chosen because it had the highest recall (0.845), while remaining simple and interpretable. Recall is important because missing an at-risk student is the more costly error.

Isotonic calibration gave the lowest Brier score (0.0842). The test-set dropout rates were 7.4% for Low risk, 48.8% for Medium risk, and 90.0% for High risk.

## Figures

- `fig01_class_balance.png`
- `fig02_correlation_heatmap.png`
- `fig03_group_dropout_rates.png`
- `fig04_cluster_selection.png`
- `fig05_student_profiles.png`
- `fig06_model_comparison.png`
- `fig07_best_confusion_matrix.png`
- `fig08_best_roc_curve.png`
- `fig09_calibration_curve.png`
- `fig10_shap_summary.png`
- `fig11_shap_bar.png`
- `fig12_shap_waterfall_high.png`
- `fig13_shap_waterfall_medium.png`
- `fig14_shap_waterfall_low.png`
- `fig15_hybrid_comparison.png`
- `fig16_afsi_sensitivity.png`
- `fig17_recommendation_frequency.png`
- `fig18_hybrid_tier_dropout.png`

## Tables

- `results/tables/model_comparison.csv`
- `results/tables/calibration_scores.csv`
- `results/tables/risk_tier_summary.csv`
- `results/tables/hybrid_comparison.csv`
- `results/tables/afsi_sensitivity.csv`
- `results/tables/hybrid_tier_summary.csv`
- `results/tables/recommender_summary.csv`
- `results/tables/recommender_examples.csv`

## How to run

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run the command from the project root. Streamlit prints a local URL, normally
`http://localhost:8501`, which opens the dashboard in a browser. This runs the
dashboard locally; no deployment step is required.

To run the analysis notebooks instead, start Jupyter:

```powershell
jupyter notebook
```

Run the notebooks in this order:

1. `01_data_understanding_eda.ipynb`
2. `02_unsupervised_student_profiles.ipynb`
3. `03_model_building_and_comparison.ipynb`
4. `04_calibration_explainability_risk_tiers.ipynb`
5. `05_hybrid_risk_engine.ipynb`

Run them with the project root as the working directory.

## Limitations

This is an academic project, not a decision-making system. It uses first- and second-semester performance, so it is a late warning rather than an early warning. The data comes from one institution and one period, so the results may not generalise. Gender and nationality are present in the data, so risk tiers must not be used for punitive decisions. The Medium risk tier is small, so its dropout rate is less reliable. The binary target treats Enrolled students as non-dropouts.

The hybrid score inherits the limitations of the calibrated model and adds a
hand-designed index whose weights are not causal estimates. Alpha is selected
from training folds, but all reported performance still comes from one split.
The counterfactual recommender evaluates hypothetical feature edits against the
model; it cannot establish that a student changing those features would actually
avoid dropout. Small tier groups and data from one institution and period also
limit reliability and generalisation.
