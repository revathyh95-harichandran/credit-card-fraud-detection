# Trains the final fraud model once, on all the development data, and saves
# it to outputs/models/fraud_model.joblib.
#
# How to run (from the project folder, with the project's Python):
#     .venv\Scripts\python.exe src\train.py
#
# Training needs the answers (the Class column), takes a little while, and is
# only run when the model should be rebuilt. To score new transactions with the
# saved model, use predict.py instead: it never trains anything.
#
# What is saved is a scikit-learn Pipeline: the preparation step (scale Amount,
# pass V1 to V28 through unchanged) followed by the XGBoost model, as one
# object. predict.py loads that one object, so new data always gets exactly the
# same preparation, using the Amount mean and spread learned here.
#
# The test group (data/processed/test.csv) is never opened here.

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from walk_forward import feature_columns, fit_final_model, make_model, prepare_with_scaler

# This file sits in the src folder, so the project folder is one level up.
# Building paths from here means the script finds its files wherever it is
# started from.
project_folder = Path(__file__).resolve().parent.parent
development_path = project_folder / "data" / "processed" / "development.csv"
model_path = project_folder / "outputs" / "models" / "fraud_model.joblib"


def build_pipeline(learn_y):
    # Preparation: V1 to V28 pass through unchanged, Amount is scaled. Listing
    # the V columns first keeps the column order V1 ... V28, Amount, exactly as
    # in every notebook. Any other column (row_id, Time, Class, block, in_gap)
    # is dropped (remainder="drop", also the default, written out to be clear).
    v_columns = []
    for column in feature_columns:
        if column != "Amount":
            v_columns.append(column)

    preparation = ColumnTransformer(
        [
            ("unchanged", "passthrough", v_columns),
            ("scale_amount", StandardScaler(), ["Amount"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # Hand the model a table with the real column names, as in the notebooks.
    preparation.set_output(transform="pandas")

    # The model: the same function the notebooks used, so the settings are
    # guaranteed to be the same (XGBoost, class weighting from learn_y).
    model = make_model("XGBoost", learn_y)

    return Pipeline([("preparation", preparation), ("model", model)])


development = pd.read_csv(development_path)
learn_X = development[feature_columns]
learn_y = development["Class"]
print(f"development data: {len(development):,} rows, {learn_y.sum()} fraud")

pipeline = build_pipeline(learn_y)
pipeline.fit(learn_X, learn_y)
amount_scaler = pipeline.named_steps["preparation"].named_transformers_["scale_amount"]
print(f"trained. Amount scaler learned mean {amount_scaler.mean_[0]:.2f}, spread {amount_scaler.scale_[0]:.2f}")

# Check: this must be exactly the model evaluated in notebooks 06 to 08
# (built there with fit_final_model). Compared on the development data only,
# so the test group is not scored again. If the scores differ at all, stop
# without saving.
pipeline_scores = pipeline.predict_proba(learn_X)[:, 1]
reference_model, reference_scaler = fit_final_model(development, feature_columns)
reference_X = prepare_with_scaler(development, reference_scaler, feature_columns)
reference_scores = reference_model.predict_proba(reference_X)[:, 1]
largest_gap = np.abs(pipeline_scores - reference_scores).max()
print("largest score difference from the evaluated model:", largest_gap)
if largest_gap != 0:
    sys.exit("Stopped: the pipeline does not reproduce the evaluated model exactly, so it was not saved.")

joblib.dump(pipeline, model_path)
size_kb = model_path.stat().st_size / 1024
print(f"saved: {model_path.relative_to(project_folder)} ({size_kb:,.0f} KB)")
