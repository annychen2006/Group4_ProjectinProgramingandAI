# Fatigue regression results

We fit 31 linear regressions to see which daily covariates best predict a person's fatigue score (1–10). Each regression uses a different combination of covariates. They are ranked below from highest to lowest R².

## Data

Each row is one person on one day: 8 person-days from 4 people (Anny, Bianca, Taeyun, Zaid), logged between 14 and 19 October 2026. The outcome is the fatigue score each person recorded for that day.

| Name | Date | Fatigue | Academic % | Work % | Social % | Interactions | Weekend |
|---|---|---|---|---|---|---|---|
| Taeyun | Wed 2026-10-14 | 8 | 10.42 | 0.00 | 0.00 | 3 | 0 |
| Anny | Sat 2026-10-17 | 5 | 12.50 | 0.00 | 6.94 | 4 | 1 |
| Taeyun | Sun 2026-10-18 | 9 | 0.00 | 0.00 | 27.08 | 2 | 1 |
| Anny | Sun 2026-10-18 | 2 | 0.00 | 0.00 | 6.25 | 2 | 1 |
| Bianca | Sun 2026-10-18 | 8 | 0.00 | 8.33 | 5.21 | 5 | 1 |
| Zaid | Mon 2026-10-19 | 9 | 22.57 | 0.00 | 7.99 | 12 | 0 |
| Anny | Mon 2026-10-19 | 4 | 4.86 | 0.00 | 1.39 | 4 | 0 |
| Bianca | Mon 2026-10-19 | 9 | 4.17 | 6.25 | 0.69 | 6 | 0 |

## Covariates

| Covariate | Definition |
|---|---|
| `count` | Number of interactions logged that day |
| `academic` | Minutes of academic interactions ÷ 1,440 × 100. Academic = group work, study, lecture, class discussion, teacher |
| `work` | Minutes of work interactions ÷ 1,440 × 100. Work = work meeting |
| `social` | Minutes of social interactions ÷ 1,440 × 100. Social = friend, family |
| `weekend` | 1 for Saturday or Sunday, 0 for Monday to Friday |

1,440 is the number of minutes in a day (24 × 60). Interaction types are matched regardless of capitalization, so "lecture" and "Lecture" count as the same type.

## Method

Each of the 31 non-empty combinations of the 5 covariates is fit as its own ordinary least squares (OLS) regression with an intercept, predicting `fatigue_score`. A covariate counts as statistically significant if its coefficient's p-value is below 0.05; the intercept is not counted.

## Results

| Rank | Covariates | # covariates | R² | Adj R² | F-test p | # significant | Significant covariates |
|---|---|---|---|---|---|---|---|
| 1 | count + academic + work + social + weekend | 5 | 0.965 | 0.877 | 0.086 | 3 | work, social, weekend |
| 2 | academic + work + social + weekend | 4 | 0.841 | 0.628 | 0.144 | 0 | – |
| 3 | count + work + social + weekend | 4 | 0.735 | 0.382 | 0.287 | 0 | – |
| 4 | work + social + weekend | 3 | 0.714 | 0.500 | 0.138 | 0 | – |
| 5 | count + academic + work + social | 4 | 0.638 | 0.155 | 0.426 | 0 | – |
| 6 | academic + work + social | 3 | 0.620 | 0.335 | 0.234 | 0 | – |
| 7 | count + work + social | 3 | 0.491 | 0.109 | 0.394 | 0 | – |
| 8 | count + academic + social + weekend | 4 | 0.443 | −0.300 | 0.692 | 0 | – |
| 9 | count + social + weekend | 3 | 0.416 | −0.023 | 0.497 | 0 | – |
| 10 | count + academic + work + weekend | 4 | 0.343 | −0.532 | 0.806 | 0 | – |
| 11 | academic + work + weekend | 3 | 0.343 | −0.151 | 0.602 | 0 | – |
| 12 | academic + social + weekend | 3 | 0.335 | −0.164 | 0.613 | 0 | – |
| 13 | count + academic + work | 3 | 0.328 | −0.175 | 0.622 | 0 | – |
| 14 | social + weekend | 2 | 0.328 | 0.060 | 0.370 | 0 | – |
| 15 | count + academic + social | 3 | 0.328 | −0.176 | 0.623 | 0 | – |
| 16 | academic + work | 2 | 0.326 | 0.056 | 0.373 | 0 | – |
| 17 | count + work + weekend | 3 | 0.324 | −0.183 | 0.629 | 0 | – |
| 18 | count + social | 2 | 0.317 | 0.044 | 0.385 | 0 | – |
| 19 | count + work | 2 | 0.304 | 0.025 | 0.404 | 0 | – |
| 20 | work + social | 2 | 0.271 | −0.020 | 0.453 | 0 | – |
| 21 | work + weekend | 2 | 0.249 | −0.051 | 0.488 | 0 | – |
| 22 | count + academic + weekend | 3 | 0.233 | −0.343 | 0.759 | 0 | – |
| 23 | count + academic | 2 | 0.219 | −0.094 | 0.539 | 0 | – |
| 24 | count + weekend | 2 | 0.207 | −0.110 | 0.560 | 0 | – |
| 25 | count | 1 | 0.200 | 0.066 | 0.267 | 0 | – |
| 26 | academic + social | 2 | 0.159 | −0.177 | 0.648 | 0 | – |
| 27 | work | 1 | 0.142 | −0.001 | 0.357 | 0 | – |
| 28 | academic + weekend | 2 | 0.102 | −0.257 | 0.764 | 0 | – |
| 29 | weekend | 1 | 0.087 | −0.065 | 0.477 | 0 | – |
| 30 | academic | 1 | 0.063 | −0.093 | 0.549 | 0 | – |
| 31 | social | 1 | 0.063 | −0.093 | 0.550 | 0 | – |

Coefficients, standard errors and p-values for every model are in [`regression_results.csv`](regression_results.csv).

### Coefficients of the top-ranked model (#1)

| Term | Coefficient | Std. error | p-value |
|---|---|---|---|
| Intercept | 5.030 | 0.931 | 0.033 |
| count | −0.739 | 0.278 | 0.117 |
| academic | 0.431 | 0.119 | 0.069 |
| work | 1.063 | 0.195 | 0.032 |
| social | 0.359 | 0.060 | 0.027 |
| weekend | −4.273 | 0.991 | 0.050 (just under 0.05) |

Read literally, each extra percentage point of the day spent on work interactions goes with about 1.06 more fatigue points, and weekend days average about 4.3 points lower, holding the other covariates fixed. The caveats below explain why these estimates should not be relied on yet.

## What the columns mean

- **# covariates:** how many covariates the model uses (not counting the intercept).
- **R²:** the share of the day-to-day variation in fatigue that the model explains, from 0 to 1. R² never goes down when a covariate is added, so larger models rank higher almost automatically.
- **Adj R²:** R² adjusted for the number of covariates. It goes down when a covariate adds less than chance would, and it can be negative when a model does worse than simply predicting the average. It is the fairer way to compare models of different sizes.
- **F-test p:** tests whether the covariates, taken together, predict fatigue better than always guessing the average score (6.75). A p-value below 0.05 means the improvement is unlikely to be chance. Above 0.05, we cannot rule out that the fit is luck.
- **# significant / Significant covariates:** how many covariates, and which ones, have an individual p-value below 0.05. Each of those p-values tests whether that one covariate matters, given the others in the model.

### The F-test in more detail

The F-test compares two models: the baseline, which predicts the average fatigue score for every day, and the regression. The regression always makes at least slightly smaller errors than the baseline. The F-test checks whether the reduction is larger than chance would produce, given how many covariates were used:

$$F = \frac{(\text{SS}_\text{baseline} - \text{SS}_\text{model}) / k}{\text{SS}_\text{model} / (n - k - 1)}$$

SS is the sum of squared prediction errors, *k* is the number of covariates, and *n* is the number of person-days (8). The p-value comes from an F-distribution with (*k*, *n* − *k* − 1) degrees of freedom.

The F-test and the per-covariate tests can disagree. In model #1, three covariates are individually significant, but the F-test p is 0.086. With 8 data points and 6 estimated numbers, only 2 degrees of freedom remain, so the overall test has very little power.

## Interpretation and caveats

- **No model is statistically significant overall.** No model's F-test reaches p < 0.05; the best is 0.086 (model #1). The data does not yet show that any combination of these covariates reliably predicts fatigue.
- **The top R² comes from the largest model.** Model #1 uses all 5 covariates on 8 data points and comes close to fitting the data exactly. By adjusted R², models #1 (0.877), #2 (0.628) and #4 (0.500) do best.
- **Significance appears only in the full model.** None of the other 30 models has any covariate with p < 0.05. Results that appear only when every covariate is included, with 2 degrees of freedom left over, are fragile.
- **The estimates are unstable.** The interaction count coefficient is +0.37 on its own (model #25) but −0.74 in the full model. A sign that flips between models usually means there is too little data to separate the effects.
- **`work` reflects one person.** Work minutes are nonzero only on Bianca's two days, so this covariate mostly distinguishes Bianca from everyone else rather than measuring the effect of work.
- **Repeated people.** Anny contributes 3 of the 8 days. OLS treats each day as independent, which overstates how much information the data contains.
- **Next step:** collect more days per person, then rerun `python analysis/regressions.py`. With more data, a model that accounts for differences between people (for example, a mixed-effects model with a random intercept per person) would be worth trying.
