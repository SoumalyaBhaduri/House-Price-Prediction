from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "model_training" / "train.csv"


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data[data["SQUARE_FT"] < 200000].copy()
    data["SQUARE_FT"] = np.log1p(data["SQUARE_FT"])
    data["BHK_NO."] = np.log1p(data["BHK_NO."])
    return data


def build_pipeline() -> Pipeline:
    numeric_features = [
        "UNDER_CONSTRUCTION",
        "RERA",
        "BHK_NO.",
        "SQUARE_FT",
        "READY_TO_MOVE",
        "RESALE",
    ]
    categorical_features = ["POSTED_BY", "BHK_OR_RK"]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        random_state=42,
        n_jobs=1,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )


def rmsle(y_true: pd.Series, y_pred: np.ndarray) -> float:
    clipped_predictions = np.clip(y_pred, a_min=0, a_max=None)
    return float(
        np.sqrt(
            np.mean(
                (np.log1p(y_true.to_numpy()) - np.log1p(clipped_predictions)) ** 2
            )
        )
    )


def main() -> None:
    data = load_data(DATA_PATH)

    features = data.drop(
        columns=["TARGET(PRICE_IN_LACS)", "ADDRESS", "LONGITUDE", "LATITUDE"]
    )
    target = data["TARGET(PRICE_IN_LACS)"] * 100000

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    cv_scores = cross_val_score(
        pipeline,
        features,
        target,
        cv=5,
        scoring="r2",
        n_jobs=1,
    )

    print(f"Rows used: {len(data)}")
    print(f"Train R2: {pipeline.score(X_train, y_train):.4f}")
    print(f"Test R2: {r2_score(y_test, predictions):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, predictions):,.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):,.2f}")
    print(f"RMSLE: {rmsle(y_test, predictions):.4f}")
    print(f"CV R2 Mean: {cv_scores.mean():.4f}")
    print(f"CV R2 Std: {cv_scores.std():.4f}")


if __name__ == "__main__":
    main()
