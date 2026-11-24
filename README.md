# Social Battery deployment skeleton

A small Flask page that reads the team's existing Railway Postgres data. Choose a person to see the number of interactions, total interaction minutes, and average recorded fatigue. The page queries the database on each request; it does not read local CSV files.

## Run locally

1. Create and activate a Python virtual environment.
2. Install packages: `pip install -r requirements.txt`.
3. Create an ignored `.env` file with `DATABASE_URL=<Railway DATABASE_PUBLIC_URL>`.
4. Run `python app.py` and open `http://127.0.0.1:5000`.

## Load the sample data

The team's sample logs are in `data/interactions.csv` and `data/daily_fatigue.csv`. With `DATABASE_URL` set, run `python seed.py` to load them. It creates the tables from `schema.sql` if needed and only adds rows the database doesn't already have, so it is safe to run again. Interaction types are stored in title case, so "group work" and "Group Work" become one type.

Do not commit `.env` or paste the database URL into source code. The app expects the schema's `users`, `interactions`, and `daily_fatigue` tables. The selected person is passed as `user_id` in the page URL; SQL parameters keep it separate from query text.

## Deploy on Railway

Add this GitHub repository as a new service **inside the Railway project that already contains Postgres**. In the app service's Variables, set `DATABASE_URL` to `${{Postgres.DATABASE_URL}}` (use the database service's actual name if different). Railway will build the Dockerfile. In the app service's networking settings, generate a public domain and open that domain to verify the page.

The app service's public URL is the assignment deliverable; the Postgres public URL is a database connection string and should not be submitted.

## Group model v1: should the app tell this user to recharge?

`analysis/fatigue_model.py` is the team's first model. It loads our logs (`data/interactions.csv`, `data/daily_fatigue.csv`; 8 person-days logged 14–19 October 2026) and fits v1: M5 from the team spreadsheet, a linear regression of `fatigue_score` on interaction count, social/academic/work shares of the day and a weekend indicator. With fewer than 30 rows there is no 70/30 split, so v1 is fit on all 8 days, with leave-one-out as a check on new days. It compares v1 with the baseline (the average fatigue, so the app never nudges), reproduces the spreadsheet's M1–M5 for comparison, and values both rules under assumed dollar figures. `should_nudge` returns the decision for a new day.

Run `pip install -r analysis/requirements.txt`, then `python analysis/fatigue_model.py`. It writes `analysis/fatigue_model_report.md` (every number), `analysis/fatigue_model_results.json`, `analysis/fatigue_model_data.csv`, `analysis/fatigue_model_chart.png` (predicted vs actual) and `analysis/fatigue_model_comparison.png` (all candidate models). The dollar values and user count are assumptions, set at the top of the script.
