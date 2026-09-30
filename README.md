# Credit Card Fraud Detection

A fraud detection model built on the public credit card transactions dataset
released by the Machine Learning Group of the Université Libre de Bruxelles
(ULB) together with Worldline: two days of real, anonymized card
transactions, in which fraud is extremely rare.

## Project status

**Phases 0 (setup), 1 (cleaning) and 2 (exploration) are complete.** No
modelling has been done yet. This README is updated at the end of every
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

## Planned approach

These are the plans, not results yet:

- **Time-ordered data splits** (train, validation, test by transaction time)
  instead of a random split, because fraud patterns change over time.
- **AUCPR over a realistic low-recall range** as the main metric, since a
  fraud team can only review a limited number of alerts. Plain accuracy is
  not used.
- **An honest comparison of four ways to handle the class imbalance:**
  SMOTE, Borderline-SMOTE, ADASYN, and class weighting.
- **XGBoost** as the main model, compared against simpler baselines.

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
