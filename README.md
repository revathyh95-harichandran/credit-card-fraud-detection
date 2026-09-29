# Credit Card Fraud Detection

A fraud detection model built on the public credit card transactions dataset
released by the Machine Learning Group of the Université Libre de Bruxelles
(ULB) together with Worldline: two days of real, anonymized card
transactions, in which fraud is extremely rare.

## Project status

**Phase 0 (setup) and Phase 1 (cleaning) are complete.** No modelling has
been done yet. This README is updated at the end of every phase with what
that phase actually produced.

Phase 1 produced:
- `notebooks/01_data_cleaning.ipynb`: every cleaning check, run on the real
  data, with its real output.
- `docs/data_dictionary.md`: every column explained, with checked facts.
- A cleaned dataset (kept locally, not in this repository): 283,726
  transactions after removing 1,081 exact duplicate copies. 473 are fraud,
  **0.1667%**, about 1 in every 599. No missing values and no contradictory
  labels were found.

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
