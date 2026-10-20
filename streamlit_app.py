from pathlib import Path
import sys

import pandas as pd
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

PROJECT_IMPORT_ERROR = None
try:
    from custom_algorithms import afsi, hybrid_score, recommend
    from data_utils import CATEGORICAL_CODE_COLUMNS, load_data, prepare_model_data, split_data
    from model_utils import make_pipeline
except ImportError as error:
    PROJECT_IMPORT_ERROR = error


SEED = 42
ALPHA = 0.9

FIGURE_GROUPS = {
    "Data Exploration": [
        ("fig01_class_balance.png", "Class balance"),
        ("fig02_correlation_heatmap.png", "Feature correlation heatmap"),
        ("fig03_group_dropout_rates.png", "Dropout rates across groups"),
    ],
    "Student Profiles": [
        ("fig04_cluster_selection.png", "K-means cluster selection"),
        ("fig05_student_profiles.png", "Student profiles"),
    ],
    "Model Performance": [
        ("fig06_model_comparison.png", "Model comparison"),
        ("fig07_best_confusion_matrix.png", "Best-recall model confusion matrix"),
        ("fig08_best_roc_curve.png", "Best-recall model ROC curve"),
    ],
    "Calibration and Explainability": [
        ("fig09_calibration_curve.png", "Calibration curve"),
        ("fig10_shap_summary.png", "SHAP summary"),
        ("fig11_shap_bar.png", "SHAP feature importance"),
        ("fig12_shap_waterfall_high.png", "High-risk SHAP explanation"),
        ("fig13_shap_waterfall_medium.png", "Medium-risk SHAP explanation"),
        ("fig14_shap_waterfall_low.png", "Low-risk SHAP explanation"),
    ],
    "Custom Algorithm": [
        ("fig15_hybrid_comparison.png", "Hybrid comparison"),
        ("fig16_afsi_sensitivity.png", "AFSI sensitivity"),
        ("fig17_recommendation_frequency.png", "Recommendation frequency"),
        ("fig18_hybrid_tier_dropout.png", "Hybrid tier dropout rates"),
    ],
}


st.set_page_config(
    page_title="Student Dropout Risk Prediction",
    page_icon="📊",
    layout="wide",
)

if PROJECT_IMPORT_ERROR is not None:
    st.error(f"A required project module or dependency is unavailable: {PROJECT_IMPORT_ERROR}")
    st.stop()


@st.cache_data
def load_project_data() -> pd.DataFrame:
    return load_data()


@st.cache_data
def load_table(filename: str) -> pd.DataFrame | None:
    path = ROOT / "results" / "tables" / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def train_models():
    data = load_project_data()
    model_data = prepare_model_data(data)
    X_train, _, y_train, _ = split_data(model_data, random_state=SEED)

    base_model = make_pipeline(
        "logistic_regression", X_train.columns, "class_weight", SEED
    )
    calibrated_model = CalibratedClassifierCV(
        estimator=base_model, method="isotonic", cv=5
    )
    calibrated_model.fit(X_train, y_train)
    return calibrated_model, X_train


def model_input(raw_row: pd.DataFrame) -> pd.DataFrame:
    prepared_row = raw_row.copy()
    if "Target" not in prepared_row.columns:
        prepared_row["Target"] = "Graduate"
    return prepare_model_data(prepared_row).drop(columns=["dropout"], errors="ignore")


def risk_tier(probability: float) -> str:
    if probability <= 0.30:
        return "Low"
    if probability <= 0.60:
        return "Medium"
    return "High"


def support_action(tier: str) -> str:
    return {
        "Low": "Routine check-in and normal academic resources.",
        "Medium": "Advisor meeting with targeted academic and financial support.",
        "High": "Prompt advisor outreach, financial review, and intensive support planning.",
    }[tier]


def training_defaults(training_rows: pd.DataFrame) -> dict:
    defaults = {}
    for column in training_rows.columns:
        if column in {"Target", "dropout"}:
            continue
        if column in CATEGORICAL_CODE_COLUMNS or not pd.api.types.is_numeric_dtype(training_rows[column]):
            defaults[column] = training_rows[column].mode(dropna=True).iloc[0]
        else:
            defaults[column] = float(training_rows[column].median())
    return defaults


def display_result(raw_row: pd.DataFrame, calibrated_model) -> None:
    ml_probability = float(calibrated_model.predict_proba(model_input(raw_row))[:, 1][0])
    afsi_score = float(afsi(raw_row).iloc[0])
    hybrid_probability = float(hybrid_score([ml_probability], [afsi_score], ALPHA)[0])
    ml_tier = risk_tier(ml_probability)
    hybrid_tier = risk_tier(hybrid_probability)

    st.subheader("Prediction")
    cards = st.columns(4)
    cards[0].metric("Dropout probability", f"{ml_probability:.1%}")
    cards[1].metric("Risk tier", ml_tier)
    cards[2].metric("AFSI score", f"{afsi_score:.1f} / 100")
    cards[3].metric("Hybrid probability", f"{hybrid_probability:.1%}")
    st.progress(ml_probability, text=f"Calibrated ML probability: {ml_probability:.1%}")

    st.write(f"**Hybrid risk tier:** {hybrid_tier}")
    st.write(f"**Interpretation:** The calibrated model estimates a {ml_probability:.1%} dropout probability for this profile.")
    st.write(f"**Suggested support:** {support_action(hybrid_tier)}")
    st.caption("This is an educational decision-support estimate, not a definitive outcome or causal prediction.")

    if hybrid_tier == "High":
        if st.button("Generate Support Suggestions", key="support_suggestions"):
            recommendation_row = raw_row.iloc[0].copy()
            if "Target" not in recommendation_row:
                recommendation_row["Target"] = "Graduate"
            suggestions = recommend(
                recommendation_row, calibrated_model, max_steps=3, target=0.30
            )
            if suggestions:
                for suggestion in suggestions:
                    st.write(f"- {suggestion}")
            else:
                st.info("No permitted feature change lowered the model probability in the search.")


def predictor_page(data: pd.DataFrame, calibrated_model, X_train: pd.DataFrame) -> None:
    st.subheader("Student Risk Predictor")
    training_rows = data.loc[X_train.index]
    defaults = training_defaults(training_rows)
    if "student_form" not in st.session_state:
        st.session_state.student_form = defaults.copy()

    if st.button("Load Example Student"):
        example = data.iloc[0].to_dict()
        st.session_state.student_form = {**defaults, **example}
        st.session_state.example_loaded = True
        st.session_state.pop("prediction_row", None)
        st.rerun()
    if st.session_state.get("example_loaded"):
        st.caption("Example loaded from an actual dataset row; it is not a newly collected student.")

    courses = sorted(data["Course"].dropna().unique().tolist())
    with st.form("student_predictor"):
        st.subheader("Academic details")
        academic = st.columns(3)
        course = academic[0].selectbox("Course", courses, index=courses.index(st.session_state.student_form["Course"]))
        age = academic[1].number_input("Age at enrolment", min_value=15.0, max_value=80.0, value=float(st.session_state.student_form["Age at enrollment"]), step=1.0)
        grade_1 = academic[2].number_input("First-semester grade", min_value=0.0, max_value=20.0, value=float(st.session_state.student_form["Curricular units 1st sem (grade)"]), step=0.1)
        units = st.columns(4)
        enrolled_1 = units[0].number_input("1st sem enrolled", min_value=0, max_value=50, value=int(st.session_state.student_form["Curricular units 1st sem (enrolled)"]))
        approved_1 = units[1].number_input("1st sem approved", min_value=0, max_value=50, value=int(st.session_state.student_form["Curricular units 1st sem (approved)"]))
        enrolled_2 = units[2].number_input("2nd sem enrolled", min_value=0, max_value=50, value=int(st.session_state.student_form["Curricular units 2nd sem (enrolled)"]))
        approved_2 = units[3].number_input("2nd sem approved", min_value=0, max_value=50, value=int(st.session_state.student_form["Curricular units 2nd sem (approved)"]))
        grade_2 = st.number_input("Second-semester grade", min_value=0.0, max_value=20.0, value=float(st.session_state.student_form["Curricular units 2nd sem (grade)"]), step=0.1)

        st.subheader("Financial details")
        finance = st.columns(3)
        tuition = finance[0].selectbox("Tuition fees up to date", [1, 0], index=[1, 0].index(int(st.session_state.student_form["Tuition fees up to date"])), format_func=lambda value: "Yes" if value == 1 else "No")
        debtor = finance[1].selectbox("Debtor status", [0, 1], index=[0, 1].index(int(st.session_state.student_form["Debtor"])), format_func=lambda value: "No" if value == 0 else "Yes")
        scholarship = finance[2].selectbox("Scholarship holder", [0, 1], index=[0, 1].index(int(st.session_state.student_form["Scholarship holder"])), format_func=lambda value: "No" if value == 0 else "Yes")
        submitted = st.form_submit_button("Predict Dropout Risk")

    if submitted:
        if approved_1 > enrolled_1 or approved_2 > enrolled_2:
            st.error("Approved units cannot exceed enrolled units.")
            return
        row = defaults.copy()
        row.update({
            "Course": course,
            "Age at enrollment": age,
            "Curricular units 1st sem (enrolled)": enrolled_1,
            "Curricular units 1st sem (approved)": approved_1,
            "Curricular units 1st sem (grade)": grade_1,
            "Curricular units 2nd sem (enrolled)": enrolled_2,
            "Curricular units 2nd sem (approved)": approved_2,
            "Curricular units 2nd sem (grade)": grade_2,
            "Tuition fees up to date": tuition,
            "Debtor": debtor,
            "Scholarship holder": scholarship,
        })
        st.session_state.student_form = row
        st.session_state.prediction_row = row

    if "prediction_row" in st.session_state:
        display_result(
            pd.DataFrame([st.session_state.prediction_row]),
            calibrated_model,
        )


def visual_page() -> None:
    st.subheader("Visual Analytics")
    for group, figures in FIGURE_GROUPS.items():
        st.subheader(group)
        available_figures = [
            (ROOT / "results" / "figures" / filename, caption)
            for filename, caption in figures
            if (ROOT / "results" / "figures" / filename).exists()
        ]
        if not available_figures:
            st.info(f"No figures are available for {group}.")
            continue
        for start in range(0, len(available_figures), 2):
            columns = st.columns(2)
            for column, (path, caption) in zip(columns, available_figures[start:start + 2]):
                with column:
                    st.image(str(path), caption=caption, width="stretch")


def evaluation_page() -> None:
    st.subheader("Model Evaluation")
    primary_tables = [
        ("model_comparison.csv", "Model comparison"),
        ("calibration_scores.csv", "Calibration scores"),
        ("risk_tier_summary.csv", "Risk-tier summary"),
        ("hybrid_comparison.csv", "Hybrid comparison"),
    ]
    for filename, title in primary_tables:
        table = load_table(filename)
        if table is None:
            st.info(f"{filename} is not available.")
            continue
        st.subheader(title)
        st.dataframe(table, hide_index=True, width="stretch")

    with st.expander("Additional analyses"):
        secondary_tables = [
        ("afsi_sensitivity.csv", "AFSI sensitivity"),
        ("hybrid_tier_summary.csv", "Hybrid tier summary"),
        ("recommender_summary.csv", "Recommender summary"),
        ("recommender_examples.csv", "Recommender examples"),
        ]
        for filename, title in secondary_tables:
            table = load_table(filename)
            if table is None:
                st.info(f"{filename} is not available.")
                continue
            st.subheader(title)
            st.dataframe(table, hide_index=True, width="stretch")

    model_results = load_table("model_comparison.csv")
    hybrid_results = load_table("hybrid_comparison.csv")
    if model_results is not None:
        best = model_results.sort_values(["recall", "pr_auc"], ascending=False).iloc[0]
        st.write(f"The selected model is **{best['model']} with {best['imbalance_strategy']}** because it has the highest recall ({best['recall']:.3f}).")
    if hybrid_results is not None:
        hybrid = hybrid_results.loc[hybrid_results["scorer"] == "Hybrid"].iloc[0]
        st.write(f"The held-out hybrid result is ROC-AUC {hybrid['roc_auc']:.3f}, PR-AUC {hybrid['pr_auc']:.3f}, and Brier score {hybrid['brier_score']:.3f}.")


try:
    data = load_project_data()
    calibrated_model, X_train = train_models()
except Exception as error:
    st.error(f"The dashboard could not load the dataset or project model: {error}")
    st.stop()

with st.sidebar:
    st.caption("Student Dropout Risk Prediction")
    page = st.radio("Navigate", ["Student Risk Predictor", "Visual Analytics", "Model Evaluation"])
    st.caption("Saurab Thapa | URK24CS9026")

st.title("Student Dropout Risk Prediction")
if page == "Student Risk Predictor":
    predictor_page(data, calibrated_model, X_train)
elif page == "Visual Analytics":
    visual_page()
else:
    evaluation_page()

st.divider()
st.caption("Saurab Thapa | URK24CS9026 | Data Science Ecosystem (23CS2024)")