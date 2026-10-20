"""Small, transparent algorithms for the hybrid dropout risk engine."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

from data_utils import add_engineered_features, prepare_model_data


AFSI_WEIGHTS = {
    "approval_1st": 0.30,
    "approval_2nd": 0.40,
    "financial_risk": 0.20,
    "grade_decline": 0.10,
}


def afsi(data: pd.DataFrame, weights: Mapping[str, float] = AFSI_WEIGHTS) -> pd.Series:
    """Return the 0-100 Academic-Financial Stress Index."""
    engineered = add_engineered_features(data)
    stress = (
        weights["approval_1st"] * (1 - engineered["1st_sem_approval_rate"])
        + weights["approval_2nd"] * (1 - engineered["2nd_sem_approval_rate"])
        + weights["financial_risk"] * engineered["financial_risk_flag"]
        + weights["grade_decline"] * (engineered["grade_change"] < 0)
    )
    return (100 * stress).clip(0, 100)


def hybrid_score(
    calibrated_probability: np.ndarray,
    afsi_score: pd.Series | np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Combine calibrated ML probability and AFSI using alpha."""
    return alpha * np.asarray(calibrated_probability) + (1 - alpha) * (
        np.asarray(afsi_score) / 100
    )


def _model_probability(row: pd.Series, model) -> float:
    """Prepare one raw student row and return its calibrated probability."""
    prepared = prepare_model_data(pd.DataFrame([row])).drop(
        columns=["dropout"], errors="ignore"
    )
    return float(model.predict_proba(prepared)[:, 1][0])


def recommend(
    student_row: pd.Series,
    model,
    max_steps: int = 3,
    target: float = 0.30,
) -> list[str]:
    """Greedily suggest actionable changes that lower the model probability."""
    row = student_row.copy()
    probability = _model_probability(row, model)
    recommendations = []
    actions = [
        ("Tuition fees up to date", 1, "Clear tuition fee dues"),
        ("Debtor", 0, "Clear debtor flag"),
        (
            "Curricular units 1st sem (approved)",
            min(row["Curricular units 1st sem (enrolled)"], row["Curricular units 1st sem (approved)"] + 1),
            "Approve one more first-semester unit",
        ),
        (
            "Curricular units 2nd sem (approved)",
            min(row["Curricular units 2nd sem (enrolled)"], row["Curricular units 2nd sem (approved)"] + 1),
            "Approve one more second-semester unit",
        ),
        (
            "Curricular units 2nd sem (grade)",
            min(18, row["Curricular units 2nd sem (grade)"] + 1),
            "Raise second-semester grade by one point",
        ),
    ]
    for _ in range(max_steps):
        if probability < target:
            break
        best = None
        for column, value, label in actions:
            if value == row[column]:
                continue
            candidate = row.copy()
            candidate[column] = value
            candidate_probability = _model_probability(candidate, model)
            if best is None or candidate_probability < best[0]:
                best = (candidate_probability, column, value, label)
        if best is None or best[0] >= probability:
            break
        new_probability, column, value, label = best
        recommendations.append(
            f"{label}: {probability:.0%} to {new_probability:.0%}"
        )
        row[column] = value
        probability = new_probability
        actions = [
            (name, (value if name == column else current), text)
            for name, current, text in actions
        ]
    return recommendations
