from collections.abc import Iterable

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from data_utils import CATEGORICAL_CODE_COLUMNS


def make_preprocessor(columns: Iterable[str]) -> ColumnTransformer:
    columns = list(columns)
    categorical = [column for column in CATEGORICAL_CODE_COLUMNS if column in columns]
    numeric = [column for column in columns if column not in categorical]
    return ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )


def make_model(name: str, imbalance: str = "none", random_state: int = 42):
    model_kwargs = {
        "logistic_regression": dict(max_iter=1000, random_state=random_state),
        "random_forest": dict(
            n_estimators=250, max_depth=8, random_state=random_state, n_jobs=-1
        ),
        "xgboost": dict(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=2,
        ),
    }
    classes = {
        "logistic_regression": LogisticRegression,
        "random_forest": RandomForestClassifier,
        "xgboost": XGBClassifier,
    }
    if name not in classes:
        raise ValueError(f"Unknown model: {name}")
    estimator_kwargs = model_kwargs[name]
    if imbalance == "class_weight" and name != "xgboost":
        estimator_kwargs["class_weight"] = "balanced"
    if imbalance == "class_weight" and name == "xgboost":
        estimator_kwargs["scale_pos_weight"] = 1.0
    estimator = classes[name](**estimator_kwargs)
    return estimator


def make_pipeline(
    name: str, columns: Iterable[str], imbalance: str = "none", random_state: int = 42
):
    estimator = make_model(name, imbalance, random_state)
    steps = [("preprocess", make_preprocessor(columns))]
    if imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("model", estimator))
    return Pipeline(steps)
