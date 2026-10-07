# Credit Card Fraud Detection

A fraud detection model built on the public credit card transactions dataset
released by the Machine Learning Group of the Université Libre de Bruxelles
(ULB) together with Worldline: two days of real, anonymized card
transactions, in which fraud is extremely rare.

## Project status

**Phases 0 to 8 are complete:** setup, cleaning, exploration, the
time-based split, handling the class imbalance, comparing models, the
final model's one-time evaluation on the test group, choosing the
decision thresholds, and explaining the model. The final tidy-up (Phase 9)
is still to come. This README is updated at the end of every phase with
what that phase actually produced.

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

## Comparing models (Phase 5)

From `notebooks/05_models.ipynb`, on the development data only, with
class weighting and the same walk-forward rounds for every model. The
shared steps live in `src/walk_forward.py`, so every model runs exactly
the same code.

- **Three models, standard settings:** Logistic Regression, Random Forest
  and XGBoost, each with reasonable, explained default settings rather
  than a search for the best ones. With only 250 fraud cases to judge on,
  tuning would partly fit those particular cases. These results show
  standard-setting performance, not each model's ceiling.

| Model | AUCPR, recall 0 to 0.2 (95% CI) | Average precision (95% CI) |
|---|---|---|
| Logistic Regression | 0.997 (0.983 to 1.000) | 0.692 (0.631 to 0.749) |
| Random Forest | 0.960 (0.896 to 0.998) | 0.758 (0.698 to 0.810) |
| XGBoost | 0.969 (0.912 to 1.000) | 0.753 (0.694 to 0.804) |

- **Result:** all three are tied on the main metric (no paired difference
  proven). On average precision, XGBoost and Random Forest are genuinely
  ahead of Logistic Regression overall, though not in every round: in one
  later round Logistic Regression was genuinely better. XGBoost and
  Random Forest could not be told apart.
- **Chosen: XGBoost**, as a practical tie-break with Random Forest: its
  scores are smooth (useful for choosing a decision threshold), it was best
  or tied on the main metric in every round, and it is fast.
- **Scaling Amount** didn't measurably change performance, but without it
  Logistic Regression failed to finish learning in three of four rounds,
  so it is kept.
- **An hour-of-day feature** (fraud's share rises at night) was tested and
  **left out**: it didn't measurably improve XGBoost.

Results: [by round](outputs/tables/models_by_round.csv),
[pooled](outputs/tables/models_pooled.csv),
[paired differences](outputs/tables/models_paired_differences.csv),
[hour-of-day test](outputs/tables/hour_of_day_paired_differences.csv).

## The final result (Phase 6)

From `notebooks/06_final_evaluation.ipynb`. The final model (XGBoost,
class weighting, standard settings, 29 features) was trained once on all
the development data, then scored **once** on the locked test group: the
latest 20% of the data, 55,372 transactions with 74 fraud cases, never
used for any decision. Nothing was changed after seeing the result.

| Metric (test group) | Result | 95% confidence interval |
|---|---|---|
| **AUCPR over recall 0 to 0.2 (headline)** | **1.00** | 1.00 to 1.00 |
| Average precision (whole curve) | 0.80 | 0.71 to 0.88 |

- **What the headline means:** working down the model's list of most
  suspicious transactions, every top alert was real fraud. In fact the
  first 37 alerts, half of all the fraud, contained no false alarm. The
  interval is a single point because the metric is at its ceiling; that
  means no top alert was wrong in this test period, not that the model is
  certain to be perfect on new data.
- **Why accuracy isn't used:** a model that calls everything genuine
  scores 99.87% accuracy on this test group while catching no fraud.
- **The cost of catching everything:** 80% of the fraud takes 117 alerts
  (about half of them real fraud); all 74 take 22,970 alerts, because the
  last few fraud cases look completely ordinary to the model.
  [Alert depth table](outputs/tables/test_alert_depth.csv),
  [precision-recall curve](outputs/figures/08_precision_recall_curve_test.png)
- **Against the walk-forward estimate** (0.97 headline, 0.75 average
  precision): the test scores are higher, but the confidence intervals
  overlap, so they are consistent rather than proven different.

## From risk score to decision (Phase 7)

From `notebooks/07_threshold.ipynb`. The model gives every transaction a
risk score; a **cutoff** turns that into a yes/no decision (flag it for
the fraud team, or let it through). Both cutoffs below were chosen on the
walk-forward check blocks only, expressed as "flag the top X% most
suspicious transactions", then applied **once** to the saved test scores.

- **Main cutoff: flag the top 0.090%.** On the check blocks, alerts stay
  93% to 96% real fraud until about 60% of the fraud is found, then
  precision falls off a cliff between 70% and 80%. This line sits at the
  end of that stable stretch, with a safety margin.
  [Precision-recall curve](outputs/figures/09_precision_recall_curve_check_blocks.png),
  [depth table](outputs/tables/check_blocks_alert_depth.csv)
- **Cost-based cutoff: flag the top 0.196%.** Under stated business
  assumptions (a missed fraud costs its Amount + 20, a false alarm 8, a
  caught fraud 3 for review time), this is where total cost is lowest on
  the check blocks. These costs are assumptions, not bank data.
  [Cost curve](outputs/figures/10_cost_curve_check_blocks.png),
  [table](outputs/tables/cost_based_cutoff_check_blocks.csv)

**Test group result** (55,372 transactions, 74 fraud; 95% bootstrap
intervals in brackets):

| | Main cutoff (top 0.090%) | Cost-based (top 0.196%) | Flag nothing |
|---|---|---|---|
| Alerts | 50 | 109 | 0 |
| Fraud caught / missed | 49 / 25 | 59 / 15 | 0 / 74 |
| False alarms | 1 | 50 | 0 |
| Precision | 0.98 (0.93 to 1.00) | 0.54 (0.45 to 0.63) | — |
| Recall | 0.66 (0.56 to 0.76) | 0.80 (0.71 to 0.88) | 0 |
| MCC | 0.81 (0.73 to 0.87) | 0.66 (0.58 to 0.73) | — |
| Total cost | 4,579 (1,743 to 8,316) | 2,641 (973 to 5,433) | 9,208 |

- **Both cutoffs held up on unseen data**, close to what the check blocks
  predicted (main: 60% caught at 94% precision on the check blocks, 66% at
  98% on test; cost-based: 79% at 57%, then 80% at 54%).
- **The trade-off:** the cost-based line catches 10 more frauds at the
  price of 49 more false alarms. Total cost intervals are wide, because
  cost depends mostly on the Amounts of a few missed frauds.
- **The best cost-based line moves with the fraud rate** (from 0.09% to
  0.48% of transactions across the walk-forward rounds). In a real
  deployment, the fraud rate and alert queue should be monitored weekly
  and the line re-tuned regularly.
  [Confusion matrices](outputs/figures/11_confusion_matrix_test.png),
  [results](outputs/tables/test_final_cutoffs.csv),
  [intervals](outputs/tables/test_final_cutoffs_intervals.csv)

## What the model pays attention to (Phase 8)

From `notebooks/08_shap.ipynb`, using **SHAP** on the final model over
the development data (all 226,980 transactions; the test group was not
used again). For each transaction, SHAP measures how much each feature
pushed the model's score towards fraud or towards genuine; the pushes add
up exactly to the model's score (checked, and matched against XGBoost's
own calculation).

- **V14 matters far more than any other feature:** 16% of all pushing,
  and on fraud transactions three times the next feature. Low V14 values
  push strongly towards fraud. V14, V4, V12 and V10 together do 40% of
  the work. [Summary chart](outputs/figures/12_shap_summary.png),
  [ranking](outputs/figures/13_shap_importance_bar.png),
  [table](outputs/tables/shap_feature_importance.csv)
- **What V14 means can't be said:** the V columns are anonymized, so SHAP
  shows *that* they matter, not what real-world behaviour they stand for.
  SHAP also explains the model, not the causes of fraud.
- **Amount** is 6th of 29, a supporting feature rather than a leading one.
  Middle amounts (10 to 100) push towards genuine; exactly-zero and large
  amounts push towards fraud, matching where fraud is actually more
  common. Its largest pushes all go to frauds of exactly 99.99, which may
  be one repeated attack the model partly remembers.
  [Chart](outputs/figures/14_shap_amount_dependence.png),
  [by band](outputs/tables/shap_amount_by_band.csv)
- **Time** is not one of the model's features, so SHAP says nothing about
  it directly. The model is still slightly more suspicious of genuine
  transactions in the night hours where fraud peaks, a pattern that
  reaches it through the V columns; the effect is small.
  [Chart](outputs/figures/15_scores_by_hour.png),
  [by hour](outputs/tables/scores_by_hour.csv)

The notebook separates what the SHAP numbers show from what is only a
reasonable guess (for example, that zero amounts reflect card testing).

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
src/            Python code for the repeatable steps (walk_forward.py)
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
