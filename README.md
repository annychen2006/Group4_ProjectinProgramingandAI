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

## Decision tree: clinic no-shows

`analysis/decision_tree.py` answers the clinic overbooking assignment using `data/clinic_appointments.csv` (400 appointments). It does a stratified 70/30 train/test split (seed 42), fits a depth-3 classification tree, compares it with the hold-every-slot baseline on the test appointments, applies the overbook/hold rule to appointments D and E, and values both policies with the assignment's cost table.

Run `pip install -r analysis/requirements.txt`, then `python analysis/decision_tree.py`. It writes `analysis/decision_tree_report.md` (every number), `analysis/decision_tree_results.json` and `analysis/decision_tree.png` (the tree). The per-year figure assumes 10,000 appointments a year; change `APPOINTMENTS_PER_YEAR` to use another volume.
