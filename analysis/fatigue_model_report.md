# Social Battery v1: fatigue model results

## Data

- Self-logged by the team (Anny, Bianca, Taeyun, Zaid), 2026-10-14 to 2026-10-19.
- `interactions.csv`: 38 interactions. `daily_fatigue.csv`: 8 person-days (the outcome).
- 8 rows is under 30, so no 70/30 split: v1 is fit on all 8 rows. Leave-one-out (predict each day from the other 7) is reported as a check on new days.

| Person | Date | Hours of interactions | Fatigue | v1 predicted | v1 decision |
|---|---|---|---|---|---|
| Taeyun | 2026-10-14 | 2.50 | 8 | 5.85 | stay quiet |
| Anny | 2026-10-17 | 4.67 | 5 | 7.42 | nudge |
| Taeyun | 2026-10-18 | 6.50 | 9 | 8.75 | nudge |
| Anny | 2026-10-18 | 1.50 | 2 | 5.13 | stay quiet |
| Bianca | 2026-10-18 | 3.25 | 8 | 6.40 | stay quiet |
| Zaid | 2026-10-19 | 7.33 | 9 | 9.35 | nudge |
| Anny | 2026-10-19 | 1.50 | 4 | 5.13 | stay quiet |
| Bianca | 2026-10-19 | 2.67 | 9 | 5.97 | stay quiet |

### Data dictionary (`fatigue_model_data.csv`)

| Column | Meaning |
|---|---|
| `Name` | who logged the day |
| `date` | the day (YYYY-MM-DD) |
| `fatigue_score` | OUTCOME: self-rated fatigue that day, 1 = fully charged, 10 = drained |
| `total_hours` | hours of logged interactions that day (v1's feature) |
| `interaction_count` | number of interactions logged that day |
| `social_day_share` | friend + family minutes / 1,440 |
| `academic_day_share` | lecture + study + group work + teacher minutes / 1,440 |
| `work_day_share` | work meeting minutes / 1,440 |
| `weekend_indicator` | 1 = Saturday or Sunday, 0 = weekday |
| `predicted_v1` | v1's predicted fatigue |
| `nudge_v1` | v1's decision: nudge if predicted fatigue >= 7 |

## Choosing v1: the team spreadsheet's models vs total hours

M1-M5 reproduce `daily_fatigue_regression.xlsx` (day share = minutes / 1,440; class discussion left out of the shares; weekend = Saturday/Sunday). Leave-one-out predicts each day from a model fit on the other 7, which shows how a model does on a day it has not seen.

| Model | Features | Residual df | R² | Adjusted R² | Typical error, all days | Typical error, held-out days | Worst held-out miss |
|---|---|---|---|---|---|---|---|
| Baseline | average fatigue | 7 | 0.000 | 0.000 | 2.54 | 2.90 | 5.4 |
| M1 Count only | interaction_count | 6 | 0.200 | 0.066 | 2.27 | 3.00 | 4.8 |
| M2 Time shares only | social_day_share, academic_day_share, work_day_share | 4 | 0.595 | 0.291 | 1.61 | 6.03 | 15.4 |
| M3 Calendar only | weekend_indicator | 6 | 0.087 | -0.065 | 2.42 | 3.23 | 5.3 |
| M4 Interaction features | interaction_count, social_day_share, academic_day_share, work_day_share | 3 | 0.600 | 0.066 | 1.61 | 6.46 | 15.5 |
| M5 Full model | interaction_count, social_day_share, academic_day_share, work_day_share, weekend_indicator | 2 | 0.976 | 0.918 | 0.39 | 39.31 | 111.0 |
| v1 Total hours | total_hours | 6 | 0.350 | 0.242 | 2.05 | 2.56 | 4.3 |

M5 Full model has the highest adjusted R² (0.918), but with 6 numbers fit to 8 days it misses held-out days by 39.3 points on average on a 1-10 scale. Every spreadsheet model does worse than the baseline on held-out days. Total hours is the only candidate that beats the baseline there, so it is v1.

![model comparison](fatigue_model_comparison.png)

## Model v1

`fatigue_score ~ total_hours` (OLS)

| Term | Coefficient | p-value |
|---|---|---|
| Intercept | 4.040 | 0.057 |
| total_hours | 0.725 | 0.122 |

R² 0.350, adjusted R² 0.242. Each extra hour of interactions adds 0.72 fatigue points. v1 predicts a tired day (7+) once a day reaches 4.1 hours of interactions.

## Fit, baseline vs v1

| | R² | Typical error (RMSE, fatigue points) |
|---|---|---|
| Baseline (average 6.75), all 8 days | 0.000 | 2.54 |
| v1, all 8 days | 0.350 | 2.05 |
| Baseline, leave-one-out | -0.306 | 2.90 |
| v1, leave-one-out | -0.022 | 2.56 |

![chart](fatigue_model_chart.png)

## Decision and value

Rule: nudge if predicted fatigue >= 7, else stay quiet (`should_nudge`). 5 of 8 logged days were tired days.

Value per user-day (ASSUMED): nudge on a tired day $0.50, nudge on a fine day -$0.20, no nudge on a tired day -$0.50, no nudge on a fine day $0.00. Per year = per user-day x 1,000 users (ASSUMED) x 365 days.

| Rule | Nudges | Tired days caught | False alarms | Tired days missed | Value, 8 days | Per user-day | Per year |
|---|---|---|---|---|---|---|---|
| Baseline | 0 | 0 | 0 | 5 | -$2.50 | -$0.31 | -$114,062 |
| v1 | 3 | 2 | 1 | 3 | -$0.70 | -$0.09 | -$31,938 |
| v1, leave-one-out | 3 | 2 | 1 | 3 | -$0.70 | -$0.09 | -$31,938 |

The leave-one-out row predicts each day from a model fit on the other 7. It makes the same nudge decision as v1 on every day, so the value is the same.

## New input

weekday: 2 h lecture, 1.5 h study, 1 h group work, 1.5 h with friends = 6.0 hours. v1 predicts fatigue 8.39: **nudge**. Baseline: stay quiet.
