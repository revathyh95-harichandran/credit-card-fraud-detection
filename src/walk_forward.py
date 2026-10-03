# Shared steps for walk-forward validation, used by every notebook from
# Phase 5 onwards so that each phase runs exactly the same code.
#
# Walk-forward validation (set up in notebooks/03_time_split.ipynb): the
# development data is cut into 5 time-ordered blocks; in round r (1 to 4)
# a model learns from blocks 1 to r and is checked on block r + 1. Rows in
# the first 10 minutes after a boundary (in_gap) are never scored.

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# The features every model uses: V1 to V28, then Amount. Time is left out
# (every check block's Time is later than anything learned from, so a model
# could latch onto "bigger Time" in a way that means nothing real).
feature_columns = []
for number in range(1, 29):
    feature_columns.append("V" + str(number))
feature_columns.append("Amount")


def prepare_round(development, round_number, columns):
    # columns: the feature columns to use (normally feature_columns above;
    # notebooks/05_models.ipynb also tests adding hour_of_day).
    # One round's learning blocks, and its check block without the gap rows.
    learn = development[development["block"] <= round_number]
    check = development[(development["block"] == round_number + 1) & (~development["in_gap"])]

    # The Amount scaler learns its mean and standard deviation from the
    # learning blocks only, then the same two numbers are applied to both.
    scaler = StandardScaler()
    scaler.fit(learn[["Amount"]])

    learn_X = learn[columns].copy()
    learn_X["Amount"] = scaler.transform(learn[["Amount"]])[:, 0]
    check_X = check[columns].copy()
    check_X["Amount"] = scaler.transform(check[["Amount"]])[:, 0]

    return learn_X, learn["Class"], check_X, check["Class"]


def aucpr_low_recall(answers, scores, recall_limit):
    # Area under the precision-recall curve for recall 0 to recall_limit,
    # divided by recall_limit so it runs from 0 to 1. With recall_limit 1.0
    # it equals scikit-learn's average_precision_score (checked in
    # notebooks/04_imbalance.ipynb).
    precision, recall, thresholds = precision_recall_curve(answers, scores)
    step_top = np.minimum(recall[:-1], recall_limit)
    step_bottom = np.minimum(recall[1:], recall_limit)
    area = np.sum((step_top - step_bottom) * precision[:-1])
    return area / recall_limit


def make_model(model_name, learn_y):
    # Builds a new, untrained model by name. Every model uses class weighting
    # (carried forward from notebooks/04_imbalance.ipynb). Settings are
    # explained in notebooks/05_models.ipynb; none are tuned, on purpose.
    # learn_y (the round's learning answers) is only used by XGBoost, to set
    # its class weighting from the learning data.
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    elif model_name == "Random Forest":
        # n_jobs=-1 uses every processor core; it changes speed, not results.
        return RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_leaf=1,
                                      max_features="sqrt", bootstrap=True,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
    elif model_name == "XGBoost":
        # scale_pos_weight = genuine rows / fraud rows in the learning blocks,
        # the same ratio as "balanced" class weights in the other two models.
        fraud_rows = learn_y.sum()
        genuine_rows = len(learn_y) - fraud_rows
        # The other values are XGBoost 3.4.1's own defaults, written out so
        # they are visible (confirmed by reading a trained model's settings).
        return XGBClassifier(n_estimators=100, learning_rate=0.3, max_depth=6,
                             min_child_weight=1, reg_lambda=1, reg_alpha=0, gamma=0,
                             subsample=1, colsample_bytree=1,
                             scale_pos_weight=genuine_rows / fraud_rows,
                             random_state=42, n_jobs=-1)
    else:
        raise ValueError("Unknown model: " + model_name)


def run_walk_forward(development, model_name, recall_limit, columns):
    # Runs all 4 walk-forward rounds for one model. In each round the scaler
    # and the model are fitted on the learning blocks only, then the check
    # block (gap rows excluded) is scored.
    # Returns: a table of round-by-round results, the pooled check-block
    # answers, the pooled check-block scores, and the 4 trained models.
    round_rows = []
    answers_by_round = []
    scores_by_round = []
    trained_models = []

    for round_number in range(1, 5):
        learn_X, learn_y, check_X, check_y = prepare_round(development, round_number, columns)
        model = make_model(model_name, learn_y)
        model.fit(learn_X, learn_y)
        scores = model.predict_proba(check_X)[:, 1]

        answers_by_round.append(check_y.to_numpy())
        scores_by_round.append(scores)
        trained_models.append(model)

        # Only some models record how many learning steps they took;
        # hasattr checks whether this one does before reading it.
        if hasattr(model, "n_iter_"):
            learning_steps = model.n_iter_[0]
        else:
            learning_steps = None

        round_rows.append({
            "model": model_name,
            "round": round_number,
            "check_fraud": check_y.sum(),
            "low_recall_aucpr": aucpr_low_recall(check_y, scores, recall_limit),
            "average_precision": average_precision_score(check_y, scores),
            "learning_steps": learning_steps,
        })

    round_table = pd.DataFrame(round_rows)
    pooled_answers = np.concatenate(answers_by_round)
    pooled_scores = np.concatenate(scores_by_round)
    return round_table, pooled_answers, pooled_scores, trained_models


def bootstrap_metrics(answers, scores_by_name, recall_limit, number_of_resamples, seed):
    # Resamples the pooled check rows with replacement, number_of_resamples
    # times. In each resample every model in scores_by_name (a dictionary of
    # name -> scores) is scored on exactly the same rows, so the results can
    # be compared in pairs. Returns name -> {"low_recall": [...],
    # "average_precision": [...]}, one value per resample.
    rng = np.random.default_rng(seed)
    number_of_rows = len(answers)

    results = {}
    for name in scores_by_name:
        results[name] = {"low_recall": [], "average_precision": []}

    for resample_number in range(number_of_resamples):
        positions = rng.integers(0, number_of_rows, size=number_of_rows)
        resampled_answers = answers[positions]
        for name in scores_by_name:
            resampled_scores = scores_by_name[name][positions]
            results[name]["low_recall"].append(aucpr_low_recall(resampled_answers, resampled_scores, recall_limit))
            results[name]["average_precision"].append(average_precision_score(resampled_answers, resampled_scores))

    return results
