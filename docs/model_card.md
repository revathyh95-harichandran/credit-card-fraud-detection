# Model Card: Credit Card Fraud Ranking Model

A model card is a short, standard document that says what a trained model
is, what it is for and not for, how it was tested, and where it falls
short, so anyone thinking of using it can judge it honestly. This one
follows the structure of *Model Cards for Model Reporting* (Mitchell et al.,
2019). Every number below comes from the project's notebooks; nothing here
was calculated only for this card.

## Model details

- **What it is:** an XGBoost classifier (gradient-boosted decision trees)
  that gives each card transaction a fraud score from 0 to 1.
- **Settings:** 100 trees, learning rate 0.3, maximum depth 6, and XGBoost's
  other default settings, written out in `src/walk_forward.py`. No search
  for the best settings was done, on purpose (see Limitations).
- **Class weighting:** each fraud case counted 567.87 times while learning
  (226,581 genuine ÷ 399 fraud in the development data), so the rare class
  isn't ignored. Chosen in notebook 04: it tied with plain SMOTE on both
  measures, while ADASYN and Borderline-SMOTE each came out clearly worse
  on one of them; class weighting was preferred over SMOTE because it
  creates no synthetic data and keeps training fast.
- **Inputs (29 features):** the anonymized columns `V1` to `V28`, and
  `Amount` scaled with the mean (90.93) and spread (250.77) learned from the
  development data. `Time` is not used; an hour-of-day feature was tested
  and left out (notebook 05).
- **The saved file:** `outputs/models/fraud_model.joblib` (about 249 KB), a
  scikit-learn pipeline holding the Amount scaling and the model together,
  so new data is always prepared exactly as in training. Its scores were
  checked to be identical to the model evaluated in the notebooks.
- **Built with:** Python 3.14.7; pandas 3.0.5, numpy 2.5.3, scikit-learn
  1.9.1, xgboost 3.4.1, shap 0.52.0, joblib 1.6.0 (all pinned in
  `requirements.txt`).
- **Author and date:** Revathy Harichandran, October 2026. A student
  portfolio project, built to learn data science.
- **License:** the code and this documentation are MIT licensed (see
  `LICENSE`); the dataset has its own license (see Data).

## Intended use

- **What it is for:** ranking a **batch** of card transactions (for
  example, a day's worth) from most to least suspicious, so that a fraud
  team with limited time reviews the most suspicious ones first. The
  `flagged` output marks the top 0.090% of a batch (the main cutoff) or the
  top 0.196% (the cost-based cutoff).
- **Who it is for:** anyone studying how to build and evaluate a fraud
  model honestly on heavily imbalanced, time-ordered data; and as a
  starting point for discussion, not a finished product.

## Not intended for

- **Automatic decisions without a person reviewing them,** such as
  blocking a payment or a card on the score alone.
- **Scoring single transactions one at a time.** Both cutoffs are a share
  of a batch, not a fixed score, so they need a batch to mean anything.
- **Live use on today's transactions.** It learned from two days in 2013;
  fraud patterns change, and it has never been retrained or monitored on
  newer data.
- **Any other bank's or country's data,** whose transactions and
  anonymized columns would not match these.
- **Decisions about people,** such as credit, insurance or employment.
- **Reading the score as the real chance of fraud.** It ranks suspicion;
  see Limitations.

## Data

- **Source:** the *Credit Card Fraud Detection* dataset, collected during a
  research collaboration of Worldline and the Machine Learning Group of the
  Université Libre de Bruxelles (ULB), published on Kaggle
  ([mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)).
  The creators describe it as transactions made by credit cards in
  **September 2013 by European cardholders**, over two days: 492 frauds out
  of 284,807 transactions (the same counts were found in the raw file here).
- **License:** the dataset page lists its license as "Database: Open
  Database, Contents: Database Contents", linking to the Open Data Commons
  **Database Contents License (DbCL) v1.0**
  (opendatacommons.org/licenses/dbcl/1.0/). It is a published, openly
  licensed academic dataset; this repository does not redistribute it (it
  is downloaded separately; see the README).
- **Privacy:** the dataset contains **no identifiable cardholder
  information**. The creators state that "due to confidentiality issues"
  the original features could not be provided: `V1` to `V28` are principal
  components from a PCA transformation, and the only untransformed columns
  are `Time` (seconds since the first transaction), `Amount` and `Class`.
  There are no names, card numbers, merchants, locations or demographic
  details.
- **Cleaning** (notebook 01): a stable `row_id` added; 1,081 extra copies
  of exact duplicate rows removed (19 of them fraud; every fraud
  transaction keeps one copy, and no duplicates disagreed on the label); no
  missing values. Result: 283,726 transactions, 473 fraud (0.1667%).
- **Split by time, never randomly** (notebook 03):
  - **Development data**, the earliest 80%: 226,980 transactions, 399
    fraud. Every choice was made here with **walk-forward validation**: 5
    time-ordered blocks; in each of 4 rounds a model learns from the
    earlier blocks and is checked on the next (250 check fraud cases in
    total). The final model was then trained once on all of it.
  - **Test group**, the latest 20%: 56,746 transactions, 74 fraud, of
    which 55,372 (all 74 fraud) are scored, after leaving out the first 10
    minutes so a fraud burst can't straddle the boundary. Scored **once**,
    at the end, and never used to make any choice.

## How it was measured, and why

- **Headline: AUCPR over recall 0 to 0.2,** the area under the
  precision-recall curve over its first fifth, scaled to run from 0 to 1.
  It measures how clean the model's most confident alerts are, the part of
  the list a fraud team with limited time actually works through. The
  dataset's creators also recommend the precision-recall curve for this
  data.
- **Tie-breaker: average precision** over the whole precision-recall curve.
- **Secondary: MCC** (Matthews correlation coefficient), once a cutoff
  turns scores into yes/no decisions.
- **Never used: plain accuracy.** Calling every transaction genuine scores
  99.87% accuracy on the test group while catching no fraud.
- **Uncertainty:** every result has a 95% bootstrap confidence interval
  (1,000 resamples). Comparisons used the paired difference between two
  approaches scored on the same transactions; one was only called better
  when that interval excluded zero.

## Results

**Test group** (55,372 transactions, 74 fraud; notebook 06):

| Measure | Result | 95% confidence interval |
|---|---|---|
| AUCPR over recall 0 to 0.2 (headline) | 1.00 | 1.00 to 1.00 |
| Average precision | 0.80 | 0.71 to 0.88 |

- The first 37 alerts, half of all the fraud, contained no false alarm.
  The interval is a single point because the measure is at its ceiling:
  no top alert was wrong in this test period, which does not mean the
  model is certain to be perfect on new data.
- Catching 80% of the fraud takes 117 alerts (about half real fraud);
  catching all 74 takes 22,970, because the last few look completely
  ordinary to the model.
- **The walk-forward estimate before testing** was 0.969 (0.912 to 1.000)
  on the headline and 0.753 (0.694 to 0.804) on average precision. The
  test results are higher, but the intervals overlap, so they are
  consistent rather than proven different.

**At the two cutoffs** (both chosen on the walk-forward check blocks, then
applied once to the test scores; notebook 07):

| | Main cutoff (top 0.090%) | Cost-based cutoff (top 0.196%) |
|---|---|---|
| Alerts | 50 | 109 |
| Fraud caught / missed | 49 / 25 | 59 / 15 |
| False alarms | 1 (of 55,298 genuine) | 50 (of 55,298 genuine) |
| Precision | 0.98 (0.93 to 1.00) | 0.54 (0.45 to 0.63) |
| Recall | 0.66 (0.56 to 0.76) | 0.80 (0.71 to 0.88) |
| MCC | 0.81 (0.73 to 0.87) | 0.66 (0.58 to 0.73) |
| Total cost (assumed costs) | 4,579 (1,743 to 8,316) | 2,641 (973 to 5,433) |

- **Main cutoff:** the end of the stretch where alerts stayed 93% to 96%
  real fraud on the check blocks, before precision falls off a cliff.
- **Cost-based cutoff:** the lowest total cost on the check blocks under
  **assumed** costs (a missed fraud costs its Amount + 20, a false alarm 8,
  a caught fraud 3; flagging nothing costs 9,208 on the test group).
- Both behaved on the test group close to what the check blocks predicted
  (main: 60% caught at 94% precision on the check blocks; cost-based: 79%
  at 57%).

## What the model relies on

From SHAP, calculated on the development data (notebook 08):

- **V14 dominates:** 16% of all the pushing towards fraud or genuine, and
  on fraud transactions three times the next feature; low V14 values push
  strongly towards fraud. V14, V4, V12 and V10 together do 40%.
- **Amount** is 6th of 29, a supporting feature: middle amounts (10 to 100)
  push towards genuine, exactly-zero and large amounts towards fraud.
- **Time** is not a feature; the model is still slightly more suspicious of
  genuine transactions in the quiet, probably night-time hours, through
  the V columns. The effect is small.
- What V14 or any V column means in real life cannot be said: they are
  anonymized. SHAP explains the model, not the causes of fraud.

## Limitations

- **Two days of 2013 data from one source.** Whether the patterns hold
  over longer periods, today, or elsewhere is unknown.
- **Few fraud cases to measure with:** 74 in the test group and 250 in the
  check blocks, so several intervals are wide.
- **Anonymized features:** the model's reasoning can be measured but not
  explained in real-world terms, and no domain-based features could be
  built.
- **No search for the best settings,** on purpose: with so few fraud cases
  to judge on, tuning would partly fit those particular cases. These are
  standard-setting results, not the best possible.
- **The score is not the real chance of fraud.** Class weighting pushes
  scores upwards; how far they differ from real chances was not measured.
- **Possible memorising of one attack:** 27 development frauds have an
  Amount of exactly 99.99, and the model gives that amount its largest
  Amount pushes; a different attacker would not trigger them.
- **Cutoffs are shares of a batch,** and the best cost-based share moves
  with the fraud rate (from 0.09% to 0.48% of transactions across the
  walk-forward rounds).
- **The costs are assumptions,** not real bank figures, so the cost-based
  cutoff is only as good as they are.
- **Concept drift:** the fraud rate already varied from 0.10% to 0.31%
  between blocks within these two days; real fraud patterns change over
  time, and this model has no way to adapt on its own.

## Fairness and ethical considerations

- **No fairness audit was possible.** The dataset is fully anonymized and
  contains no personal or demographic information, so it is impossible to
  check whether the model flags some groups of people more often than
  others for the same behaviour. That does not mean the model is fair, only
  that its fairness is unknown: anonymized columns, and even `Amount` or
  timing, can still carry patterns linked to who someone is. **A real
  deployment would need a proper fairness audit**, on data that includes
  the relevant characteristics, handled with appropriate privacy
  safeguards.
- **False alarms have a real cost to real people:** a genuine customer
  whose payment is questioned or blocked is inconvenienced and may lose
  trust in their bank. That is why the main cutoff favours clean alerts,
  and why alerts should go to a person, not trigger an automatic block.
- **Missed fraud has a cost too,** to customers and the bank; the
  cost-based cutoff makes that trade-off explicit, under stated
  assumptions.

## Recommendations for anyone using it

- Send alerts to a **person for review**; never act on the score alone.
- **Score in batches** (such as a day's transactions), as the cutoffs
  assume.
- **Monitor weekly:** the fraud rate, the number of alerts, and how many
  alerts turn out to be real fraud.
- **Retune the cutoff and retrain** regularly on recent, labelled data,
  using the same time-ordered method, and re-check before trusting it
  again.
- **Carry out a fairness audit** before any real-world use.

## How to use it

```
.venv\Scripts\python.exe src\predict.py transactions.csv predictions.csv
```

The input needs columns `V1` to `V28` and `Amount`; the output gives each
transaction's `fraud_score`, `rank` and `flagged`. Add
`--cutoff cost-based` for the 0.196% cutoff. Set-up steps are in the README.
Only load model files you trust: loading a joblib file can run code stored
inside it.

## References

- Dataset: Machine Learning Group, ULB, and Worldline. *Credit Card Fraud
  Detection.* Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- The first work the dataset's creators ask users to cite: Dal Pozzolo, A.,
  Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating
  Probability with Undersampling for Unbalanced Classification. Symposium
  on Computational Intelligence and Data Mining (CIDM), IEEE.
- Dal Pozzolo, A., Caelen, O., Le Borgne, Y.-A., Waterschoot, S., &
  Bontempi, G. (2014). Learned lessons in credit card fraud detection from
  a practitioner perspective. Expert Systems with Applications, 41(10),
  4915-4928.
- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson,
  B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model Cards for Model
  Reporting. Proceedings of the Conference on Fairness, Accountability, and
  Transparency (FAT*).
