# Scores transactions with the saved fraud model and flags the most suspicious
# ones, using the cutoff chosen in notebooks/07_threshold.ipynb.
#
# How to run (from the project folder, with the project's Python):
#     .venv\Scripts\python.exe src\predict.py transactions.csv predictions.csv
#     .venv\Scripts\python.exe src\predict.py transactions.csv predictions.csv --cutoff cost-based
#
# Prediction never learns anything and never needs the answers: it loads the
# model saved by train.py and applies it. Any Class column in the input is
# ignored.
#
# The input CSV needs the columns V1 to V28 and Amount, in any order. Other
# columns are ignored, except row_id, which is copied to the output if present.
#
# The output CSV has one row per input row, in the same order:
#     row_id       (only if the input had one)
#     fraud_score  the model's score, 0 to 1 (higher = more suspicious)
#     rank         1 = the most suspicious transaction in this file
#     flagged      True for the transactions to send to the fraud team
#
# How flagging works: both cutoffs were chosen as ALERT RATES, "flag the top
# X% most suspicious transactions", not as fixed score values (notebook 07
# explains why). So the rule is applied to the whole input file at once, which
# only makes sense for a batch of transactions, such as a day's worth.
#
# Only load model files you trust: loading a joblib file can run code stored
# inside it.

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

project_folder = Path(__file__).resolve().parent.parent
model_path = project_folder / "outputs" / "models" / "fraud_model.joblib"

# The two cutoffs, chosen on the pooled walk-forward check blocks in notebook
# 07 (176,175 check transactions): the main "plateau" line flagged the top 159
# (about 0.090%); the cost-based line flagged the top 346 (about 0.196%).
alert_rates = {
    "plateau": 159 / 176175,
    "cost-based": 346 / 176175,
}

# Below this many alerts, a "top X%" rule is dominated by rounding (for
# example, 0.090% of 5,000 is 4.5 alerts, so one alert more or less is a 20%
# swing), so the script warns. The rule was chosen on check blocks of about
# 44,000 transactions each, which gave about 40 alerts at the plateau line.
fewest_reliable_alerts = 10

# Read the command: argparse takes the words typed after the script's name.
parser = argparse.ArgumentParser(description="Score transactions with the saved fraud model and flag the most suspicious.")
parser.add_argument("input_csv", help="CSV with columns V1 to V28 and Amount")
parser.add_argument("output_csv", help="where to write the scores and flags")
parser.add_argument("--cutoff", choices=["plateau", "cost-based"], default="plateau",
                    help="plateau (default): flag the top 0.090%%; cost-based: flag the top 0.196%%")
arguments = parser.parse_args()

pipeline = joblib.load(model_path)
needed_columns = list(pipeline.feature_names_in_)

transactions = pd.read_csv(arguments.input_csv)

# Stop with a clear message if anything the model needs is missing or empty.
missing_columns = []
for column in needed_columns:
    if column not in transactions.columns:
        missing_columns.append(column)
if len(missing_columns) > 0:
    sys.exit("Stopped: the input file is missing these columns: " + ", ".join(missing_columns))

empty_columns = []
for column in needed_columns:
    if transactions[column].isna().any():
        empty_columns.append(column)
if len(empty_columns) > 0:
    sys.exit("Stopped: these columns have empty values (the model was never trained on any): " + ", ".join(empty_columns))

# Score. The pipeline scales Amount with the mean and spread learned in
# training, then applies the model. Nothing is re-learned from this file.
scores = pipeline.predict_proba(transactions[needed_columns])[:, 1]

# Rank 1 = highest score. kind="stable" keeps tied scores in file order, so the
# ranking is the same on every run (the same rule as notebook 07).
order = np.argsort(-scores, kind="stable")
rank = np.zeros(len(scores), dtype=int)
for position in range(len(order)):
    rank[order[position]] = position + 1

# Flag the top alert-rate share of this file. round() rounds to the nearest
# whole alert; an exact half goes to the even number (2.5 -> 2), as in
# notebook 07.
alert_rate = alert_rates[arguments.cutoff]
number_of_alerts = round(alert_rate * len(scores))
flagged = rank <= number_of_alerts

predictions = pd.DataFrame()
if "row_id" in transactions.columns:
    predictions["row_id"] = transactions["row_id"]
predictions["fraud_score"] = scores
predictions["rank"] = rank
predictions["flagged"] = flagged
predictions.to_csv(arguments.output_csv, index=False)

print(f"scored {len(scores):,} transactions with {model_path.name}")
print(f"cutoff: {arguments.cutoff}, flag the top {alert_rate:.3%} -> {number_of_alerts} flagged")
if number_of_alerts < fewest_reliable_alerts:
    print(f"warning: only {number_of_alerts} alerts. A top-percentage rule is unreliable on a file this small "
          f"({len(scores):,} rows); it was designed for batches like a day of transactions.")
print(f"written: {arguments.output_csv}")
