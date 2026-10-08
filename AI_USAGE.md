# AI Usage Declaration: Credit Card Fraud Detection

**Author:** Revathy Harichandran
**AI tool used:** Claude Code (Anthropic)
**Last updated:** October 2026

## 1. Summary

I used Claude Code as an assistant during this project, mainly for writing code,
drafting documentation and figures, and routine tasks. The project idea, research
question, dataset selection, analysis plan, interpretation and all final decisions are
my own. I directed the work step by step through prompts I wrote myself, reviewed every
output, and asked for corrections whenever something was wrong or unclear. I understand
and can explain every part of the code, analysis and results in this project, and I take
full responsibility for its content.

## 2. Why I used an AI tool

I used the tool to make my work more productive and less time-consuming. Letting it
handle the routine work meant I could spend more of my time on the parts that need
human judgement: brainstorming, asking analytical and logical questions, and exploring
the data more deeply and quickly than I could have alone. I treated the tool as an
assistant, not as the owner of the project.

## 3. What I did

- **Project idea and research question:** with limited analyst time, how can a bank
  catch as much fraud as possible without wrongly flagging genuine customers?
- **Literature review:** I used the tool to help search for and summarise research before
  starting. I read the key sources myself and checked every reference.
- **Dataset selection:** I chose the ULB/Worldline credit card dataset from Kaggle and
  downloaded the data myself.
- **Data licensing and privacy:** I made sure the data is used in line with its license
  (Open Data Commons DbCL v1.0) and kept it out of the public repository.
- **Key definitions:** I defined the development data (the earliest 80%) and the test
  group (the latest 20%, used once), and the headline measure: precision among the most
  confident alerts (AUCPR over recall 0 to 0.2).
- **Planning every stage:** I decided the order of the project (setup, cleaning,
  exploration, time-based split, class imbalance, model comparison, final evaluation,
  decision thresholds, explainability, finishing) and wrote specific instructions for
  each stage.
- **Methods and rules:** I specified the rules to follow: no random splits, fitting only
  on training data, comparing all four imbalance methods, and bootstrap confidence
  intervals for every comparison.
- **Analysis questions:** I chose what to examine: four ways of handling the imbalance,
  three model types, an hour-of-day feature, and two kinds of decision threshold.
- **Evaluation plan:** I designed how the results would be tested: a time-ordered split,
  walk-forward validation, a test set used once, and paired bootstrap comparisons.
- **Decisions:** whenever the tool offered options, I made the final call. Examples are
  in section 6.
- **Review and quality control:** I reviewed every output and asked for errors to be
  explained before going further.
- **Interpretation and conclusions:** I decided what the findings mean and what to
  recommend. The tool drafted the wording, which I reviewed and had rewritten in my own
  plain, simple style.
- **Publishing:** I created the GitHub repository and signed in myself. The AI tool
  never had access to my password.

## 4. What the AI tool did

- **Wrote code** to my instructions: the notebooks for cleaning, exploration, splitting,
  modelling, evaluation and explanation; the shared module; and the training and
  prediction scripts.
- **Ran the code** and reported the results back to me.
- **Suggested approaches** for some technical details, for example a 10-minute gap at
  each time boundary so a fraud burst can't leak across it. I reviewed each suggestion
  before it was used.
- **Flagged problems** it found while running the code, for example that my first
  60/20/20 split left only 57 fraud cases for validation, which I then reviewed.
- **Drafted documentation:** the README, the data dictionary and the model card, which I
  reviewed, edited and approved.
- **Created charts** from the data, to the requirements I set.
- **Helped with research** for the literature review. I verified all sources.
- **Ran Git commands** when I asked. I did any sign-in myself.

## 5. How I stayed in control

- Every prompt was written by me. The tool did not act on its own goals.
- I worked in small, checked steps rather than asking for everything at once.
- For the write-ups and charts, I asked the tool to show them to me **before** they were
  written into files or published, and I approved them.
- Every number in the project can be reproduced by re-running the notebooks and scripts,
  so nothing rests on the AI's word alone.
- Only public, anonymised data was used. No personal or confidential data was shared
  with the AI tool, and it never had access to my passwords.

## 6. Examples of decisions I made

- **Walk-forward validation:** I replaced a single validation split with walk-forward
  validation, so choices rest on 250 fraud cases instead of 57.
- **A tie-breaker fixed in advance:** I set average precision as the tie-breaker
  *before* any comparison results existed, because the headline measure often hit its
  ceiling.
- **Class weighting:** I carried forward class weighting rather than SMOTE when the two
  tied, because it creates no synthetic data.
- **XGBoost:** I chose XGBoost over Random Forest as a practical tie-break, for its
  smooth scores, which matter for thresholds.
- **The main threshold:** I proposed basing it on where precision stays stable in the
  precision-recall curve, and chose the end of that plateau (flag the top 0.090%).
- **The cost assumptions:** I set the business costs for the cost-based threshold (a
  false alarm costs 8 units, for analyst time and customer trust; a missed fraud costs
  its amount + 20).

## 7. What I learned

Through this project I learned to lock away the test data before modelling and use it
only once; that a single number can mislead, so comparisons need confidence intervals;
and to report results honestly even when they contradict expectations.

## 8. Who did what

| Part of the project | Me | AI tool (Claude Code) |
|---|---|---|
| Project idea and research question | ✔ Did it | — |
| Literature review | ✔ Read and verified sources | Helped search and summarise |
| Dataset choice | ✔ Did it | — |
| Data cleaning | ✔ Set the rules and approved decisions | Wrote the code, suggested options |
| Analysis and model comparison | ✔ Chose what to examine | Wrote the code, ran it |
| Evaluation design and thresholds | ✔ Designed it, set the costs | Wrote and ran the code |
| Interpretation and recommendations | ✔ Decided them | Drafted the wording |
| Documentation | ✔ Reviewed and approved | Drafted it |
| Publishing / GitHub | ✔ Created and signed in | Ran commands |
