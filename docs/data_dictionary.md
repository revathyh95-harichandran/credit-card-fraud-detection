# Data Dictionary

This project uses the **Credit Card Fraud Detection** dataset, created by the
Machine Learning Group of the Université Libre de Bruxelles (ULB) together
with the payment company Worldline, and published on
[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It holds
card transactions made over about two days, each labelled as either genuine
or fraud. The raw file has 284,807 rows and 31 columns (checked when loading
it in cleaning step 1).

The descriptions below come from the dataset creators' published
description. Anything we have measured ourselves on the real file is added
only after that check has actually been run, and is labelled as such.

## Files

| File | What it is |
|---|---|
| `data/raw/creditcard.csv` | The original download, never modified. 284,807 rows, 31 columns. |
| `data/staging/creditcard_with_row_id.csv` | The raw data with `row_id` added. 284,807 rows, 32 columns. |
| `data/staging/removed_duplicate_rows.csv` | The 1,081 rows removed as exact duplicate copies, kept as a record. |
| `data/processed/creditcard_clean.csv` | The final cleaned file. 283,726 rows, 32 columns. Exact duplicate copies removed (one copy of each kept). Split by time into the two files below. |
| `data/processed/development.csv` | The earliest 80% (226,980 rows, 399 fraud, hours 0 to 40.34), plus the `block` and `in_gap` columns. Every choice in the project is made on this file, with walk-forward validation. |
| `data/processed/test.csv` | The latest 20% (56,746 rows, 74 fraud, hours 40.34 to 48.00), plus the `in_gap` column. Opened once only, for the final result. |
| `data/staging/superseded_60_20_20/` | `train.csv` and `validation.csv` from a first 60/20/20 split, replaced by walk-forward validation. Kept only as a record; not used. |

## Columns

Types were checked on the real data in cleaning step 3: `int64` means whole
numbers, `float64` means decimal numbers. No column has any missing values.

| Column | Source | Type | Description |
|---|---|---|---|
| `row_id` | Added by us (cleaning step 2) | int64 | The row's position in the original raw file, starting at 0. This is our own label, not an ID from the bank. It never changes, so any transaction can always be traced back to the raw file, even after rows are removed. |
| `Time` | Original | float64 | Seconds between this transaction and the **first transaction in the file**. It is **not** the time of day, and the clock time of the first transaction is not given. Stored as a decimal, but every value is a whole number of seconds. Checked in cleaning step 5: runs from 0 to 172,792 (about 48 hours), the file is already in time order, and many transactions share the same second. Exploration notebook 02: transactions follow a clear daily cycle, with the quietest hours exactly 24 hours apart (hours 4 and 28 after the first transaction, about 1,100 transactions each, against up to 9,875 in the busiest hour). The quiet stretches are consistent with night-time, but the clock time is not known. |
| `V1` to `V28` | Original | float64 | Anonymized features created with PCA from confidential transaction details. We genuinely can't say what each one individually represents. See the section below. |
| `Amount` | Original | float64 | The transaction amount. The currency is not stated in the published description, so none is assumed here. In the cleaned file (exploration notebook 02): ranges from 0 to 25,691.16, median 22.00, mean 88.47, 99% of transactions at or below 1,018.97. Heavily right-skewed (skewness 16.98). By the IQR rule (Q1 5.60, Q3 77.51), 31,685 transactions (11.17%) are above the upper fence of 185.38 and none below the lower fence; these are real, larger purchases, kept in the data. The fraud rate among them is 0.275%, against 0.153% for all other transactions. Fraud versus genuine (full table in `outputs/tables/amount_by_class.csv`): fraud has a lower median (9.82 vs 22.00) but a higher mean (123.87 vs 88.41), with non-overlapping 95% bootstrap confidence intervals for both; fraud has more exactly-zero amounts (5.29% vs 0.63%) and more above 185.38 (18.39% vs 11.16%), and its largest amount is 2,125.87. |
| `block` | Added by us (split notebook 03), development file only | int64 | Which of the 5 time-ordered walk-forward blocks the transaction belongs to, 1 (earliest) to 5 (latest). Each block has about 45,400 rows; transactions from the same second are always in the same block. |
| `in_gap` | Added by us (split notebook 03), development and test files | bool | `True` for transactions in the first 10 minutes after a boundary (the starts of blocks 2 to 5, and of the test group). These rows are not scored when checking or testing, so a fraud burst can't be learned on one side of a boundary and scored on the other. They are still used for learning. |
| `Class` | Original | int64 | The label the model tries to predict: `1` = confirmed fraud, `0` = genuine transaction. These are the only two values that appear. In the cleaned file: 283,253 genuine and 473 fraud (0.1667% fraud, about 1 in 599), counted in cleaning step 9. |

## Why the V columns are anonymized

The bank's original transaction details (things that could identify real
cardholders or merchants) could not be released publicly. Before
publishing, the creators transformed them with **PCA (principal component
analysis)**, which blends all the original columns into new ones. Each V
column is a weighted mix of many original details, and only the bank knows
the recipe, so the originals cannot be recovered.

What this means in practice:

- A V column's value is a position on a scale relative to other
  transactions, not a readable measurement. That is why the values are small
  decimals on both sides of zero.
- The columns are numbered in order of how much of the variation between
  transactions each one captures: V1 the most, V28 the least. Checked in
  cleaning step 6: the standard deviations do shrink steadily, from about
  1.96 for V1 to about 0.33 for V28.
- Every V column's mean is essentially exactly zero (all within about
  0.000000000000005 of it), as PCA output should be. Checked in cleaning
  step 6.
- The V columns are essentially uncorrelated with each other, as PCA
  output should be: across all 378 pairs, the largest correlation is 0.019
  (V8 with V21). Time and Amount, which were not part of PCA, do correlate
  with some V columns (Time with V3 at −0.42, Amount with V2 at −0.53).
  Checked in exploration notebook 02.
- The V columns are **not** all on the same scale as each other (see the
  standard deviations above), even though they are all centred on zero.
- The V columns can still help a model tell fraud from genuine, but if a
  later analysis finds, say, V14 important, the honest explanation stops at
  "V14", not at a real-world meaning.
- `Time` and `Amount` were **not** put through PCA, so they are on their own,
  very different scales.

## What this data cannot tell us

- **Who the cardholders are.** There is no personal or demographic
  information, so no fairness audit (checking whether the model treats
  different groups of people differently) is possible with this data. A real
  deployment would need one.
- **The currency** of `Amount`.
- **The real date or time of day** of any transaction.
