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
| `data/processed/creditcard_clean.csv` | The final cleaned file used by every later phase. 283,726 rows, 32 columns. Exact duplicate copies removed (one copy of each kept). |

## Columns

Types were checked on the real data in cleaning step 3: `int64` means whole
numbers, `float64` means decimal numbers. No column has any missing values.

| Column | Source | Type | Description |
|---|---|---|---|
| `row_id` | Added by us (cleaning step 2) | int64 | The row's position in the original raw file, starting at 0. This is our own label, not an ID from the bank. It never changes, so any transaction can always be traced back to the raw file, even after rows are removed. |
| `Time` | Original | float64 | Seconds between this transaction and the **first transaction in the file**. It is **not** the time of day, and the clock time of the first transaction is not given. Stored as a decimal, but every value is a whole number of seconds. Checked in cleaning step 5: runs from 0 to 172,792 (about 48 hours), the file is already in time order, and many transactions share the same second. |
| `V1` to `V28` | Original | float64 | Anonymized features created with PCA from confidential transaction details. We genuinely can't say what each one individually represents. See the section below. |
| `Amount` | Original | float64 | The transaction amount. The currency is not stated in the published description, so none is assumed here. |
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
