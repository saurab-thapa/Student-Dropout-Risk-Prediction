from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


TARGET = "Target"
DROP_COLUMNS = [TARGET, "dropout"]
CATEGORICAL_CODE_COLUMNS = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
]


def load_data(path: str | Path = "data/raw/data.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    data = pd.read_csv(path, sep=";")
    data.columns = data.columns.str.strip()
    data["dropout"] = (data[TARGET].str.strip() == "Dropout").astype(int)
    return data


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for semester in ("1st", "2nd"):
        approved = f"Curricular units {semester} sem (approved)"
        enrolled = f"Curricular units {semester} sem (enrolled)"
        result[f"{semester}_sem_approval_rate"] = (
            result[approved] / result[enrolled].replace(0, np.nan)
        ).fillna(0)
    result["grade_change"] = (
        result["Curricular units 2nd sem (grade)"]
        - result["Curricular units 1st sem (grade)"]
    )
    result["financial_risk_flag"] = (
        (result["Tuition fees up to date"] == 0) | (result["Debtor"] == 1)
    ).astype(int)
    return result


def split_data(
    data: pd.DataFrame, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = data.drop(columns=DROP_COLUMNS, errors="ignore")
    target = data["dropout"]
    return train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
        stratify=target,
    )


def prepare_model_data(data: pd.DataFrame) -> pd.DataFrame:
    return add_engineered_features(data).drop(columns=[TARGET])
