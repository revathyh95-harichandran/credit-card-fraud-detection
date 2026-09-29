# Credit Card Fraud Detection

A fraud detection model built on the public credit card transactions dataset
released by the Machine Learning Group of the Université Libre de Bruxelles
(ULB) together with Worldline: two days of real, anonymized card
transactions, in which fraud is extremely rare.

## Project status

**Phase 0 (setup) is complete.** The project structure, version control and
license are in place, and the dataset has been checked and unzipped locally.
No analysis or modelling has been done yet. This README is updated at the
end of every phase with what that phase actually produced.

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
