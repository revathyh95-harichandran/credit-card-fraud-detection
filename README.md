# Credit Card Fraud Detection

A fraud detection model built on the public credit card transactions dataset
released by the Machine Learning Group of the Université Libre de Bruxelles
(ULB) together with Worldline: two days of real, anonymized card
transactions, in which fraud is extremely rare.

## Project status

**Phases 0 (setup), 1 (cleaning), 2 (exploration), 3 (time-based split)
and 4 (handling the class imbalance) are complete.** The final model has
not been chosen or trained yet. This README is updated at the end of every
phase with what that phase actually produced.

Phase 1 produced:
- `notebooks/01_data_cleaning.ipynb`: every cleaning check, run on the real
  data, with its real output.
- `docs/data_dictionary.md`: every column explained, with checked facts.
- A cleaned dataset (kept locally, not in this repository): 283,726
  transactions after removing 1,081 exact duplicate copies. 473 are fraud,
  **0.1667%**, about 1 in every 599. No missing values and no contradictory
  labels were found.

## What the data looks like (Phase 2)

All from `notebooks/02_exploration.ipynb`, on the cleaned data.

- **Fraud is rare:** 473 of 283,726 transactions (0.17%), about 1 in 599.
  [Chart](outputs/figures/01_class_balance.png)
- **Amount is heavily right-skewed** (skewness 16.98): median 22.00, mean
  88.47, largest 25,691.16. By the IQR rule, 11.17% of transactions are
  unusually large (above 185.38); they are real purchases and are kept.
  [Distribution](outputs/figures/02_amount_distribution.png),
  [boxplot](outputs/figures/03_amount_boxplot.png)
- **Transactions follow a daily cycle,** with the quietest hours exactly 24
  hours apart. **Fraud does not:** its share rises to about 1.3% to 1.6% of
  transactions in the quiet hours, against 0.17% overall. (The data's clock
  time is unknown, so these can't be named as night-time for certain.)
  [Chart](outputs/figures/04_time_distribution.png)
- **Fraud looks different in Amount, but not simply "bigger":** a lower
  median than genuine (9.82 vs 22.00) and a higher mean (123.87 vs 88.41),
  with non-overlapping 95% bootstrap confidence intervals for both. Fraud
  has more exactly-zero amounts and more large ones.
  [Chart](outputs/figures/05_amount_by_class.png),
  [table](outputs/tables/amount_by_class.csv)
- **The anonymized V1 to V28 columns are uncorrelated with each other**
  (largest correlation 0.019 across all 378 pairs), as PCA output should be.
  [Heatmap](outputs/figures/06_v_correlation_heatmap.png)

## How the data is split (Phase 3)

From `notebooks/03_time_split.ipynb`. The data is split **by time, never
randomly**, because fraud patterns change over time and a random split
would let a model learn from the future.

| Part | Rows | Fraud | Used for |
|---|---|---|---|
| Development data (earliest 80%) | 226,980 | 399 | every choice, via walk-forward validation |
| Test group (latest 20%) | 56,746 | 74 | the final result, looked at once, at the end |

- **Walk-forward validation:** the development data is cut into 5
  time-ordered blocks. In each of 4 rounds a model learns from all earlier
  blocks and is checked on the next, so choices are judged on 250 fraud
  cases while never peeking at the future. A first single 60/20/20 split
  was replaced because its validation group held only 57 fraud cases.
- **Clean boundaries:** transactions from the same second always stay
  together, and the first 10 minutes after each boundary are not scored,
  so a burst of fraud can't be learned on one side and scored on the
  other.
- **The fraud rate changes over time:** from 0.10% to 0.31% across blocks,
  highest in the two blocks containing night-time stretches. The first
  block's rate (0.31%) is higher than every other part's, and the second
  night block's (0.20%) higher than its two daytime neighbours', with
  non-overlapping 95% bootstrap confidence intervals; the other
  differences may be noise.
  [Chart](outputs/figures/07_fraud_rate_over_time.png)

## Handling the class imbalance (Phase 4)

From `notebooks/04_imbalance.ipynb`, on the development data only. Four
techniques were compared, each in all four walk-forward rounds with the
same simple model (Logistic Regression), fitted on each round's learning
blocks only and judged on the pooled check blocks (250 fraud cases).

- **Metric:** AUCPR over recall 0 to 0.2 (how precise the model is among
  its most confident flags, the part a fraud team with limited time
  actually uses), with average precision over the whole curve as a
  pre-declared tie-breaker.
- **Comparison test:** 95% bootstrap confidence intervals of the paired
  difference between two techniques scored on the same transactions.
- **Result: no single clear winner.** ADASYN was clearly worse on the main
  metric, Borderline-SMOTE clearly worse on average precision, and SMOTE
  and class weighting were tied on both. No technique won every round.
  Contrary to what the literature would suggest, neither Borderline-SMOTE
  nor ADASYN beat plain SMOTE on this data.
- **Carried forward: class weighting**, as a practical tie-break between
  two equally good options: it creates no synthetic data and keeps
  training fast.
- **Data leakage shown on purpose:** fitting SMOTE before separating the
  learning and check data inflated average precision by 0.018, a real
  difference (95% interval +0.006 to +0.032) with no real improvement
  behind it.

Results: [by round](outputs/tables/imbalance_by_round.csv),
[pooled](outputs/tables/imbalance_pooled.csv),
[paired differences](outputs/tables/imbalance_paired_differences.csv).

## Planned approach

These are the plans, not results yet:

- **XGBoost** as the main model, compared against simpler baselines, using
  class weighting and the same walk-forward comparison.
- **The final model** trained once on all the development data, then
  scored once on the locked test group with AUCPR over the low-recall
  range.

## Fairness: what this project cannot check

A fair fraud model should not flag some groups of people (for example by
age, gender, ethnicity, or where they live) far more often than others for
the same behaviour. **This project cannot check for that.**

The dataset contains no usable personal or demographic information. The
only readable columns are `Time`, `Amount`, and `Class`; everything else
(`V1` to `V28`) was anonymized by the dataset's creators using PCA, which
blends the original details together so they can't be recovered. Whatever
personal details went into those columns, we can't see them, so we can't
measure whether the model treats any group differently.

That does not mean the model is fair, only that its fairness is unknown.
Anonymized features can still carry patterns linked to who someone is,
and even `Amount` or the timing of purchases can differ between groups.
A real deployment of a model like this would need a proper fairness audit,
using data that includes the relevant characteristics, handled under
appropriate privacy safeguards. That audit is outside what this dataset
makes possible.

## Project structure

```
data/
  raw/          original download, never modified (not in git)
  staging/      in-between cleaning output (not in git)
  processed/    final cleaned data used for modelling (not in git)
notebooks/      Jupyter notebooks for exploration and explanation
src/            Python scripts for the repeatable steps
outputs/
  figures/      every chart, saved as an image file
  tables/       summary tables, saved as CSV files
  models/       the final trained model
docs/           project documentation, such as the data dictionary
```

## Getting the data

The dataset is not included in this repository. Download it from Kaggle:
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
and place the downloaded `archive.zip` in `data/raw/`. Unzip it there to get
`creditcard.csv`.

## License

The code and documentation in this project are released under the MIT
License, see [LICENSE](LICENSE). The dataset is not covered by this license;
it belongs to its creators and has its own terms.
