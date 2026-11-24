"""Load data/interactions.csv and data/daily_fatigue.csv into Postgres.

Usage: set DATABASE_URL (in .env or the environment), then run
    python seed.py

Safe to run more than once: it creates the schema if missing and only adds
rows that are not already in the database. It never deletes or changes
existing rows; fatigue scores that differ from the CSV are reported instead.
"""

import csv
import os
from collections import Counter
from datetime import date
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def read_csv(name):
    with open(DATA / name, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def type_name(raw):
    # schema.sql stores interaction types in title case ("group work" -> "Group Work").
    return " ".join(raw.split()).title()


def user_id_for(cursor, name, cache):
    if name not in cache:
        cursor.execute("SELECT id FROM users WHERE name = %s ORDER BY id LIMIT 1", (name,))
        row = cursor.fetchone()
        if row is None:
            cursor.execute("INSERT INTO users (name) VALUES (%s) RETURNING id", (name,))
            row = cursor.fetchone()
            print(f"  added user {name}")
        cache[name] = row[0]
    return cache[name]


def type_id_for(cursor, name, cache):
    if name not in cache:
        cursor.execute(
            """
            INSERT INTO interaction_types (name) VALUES (%s)
            ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
            RETURNING id
            """,
            (name,),
        )
        cache[name] = cursor.fetchone()[0]
    return cache[name]


def main():
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set (add it to .env or the environment)")

    interactions = read_csv("interactions.csv")
    fatigue = read_csv("daily_fatigue.csv")

    with psycopg.connect(url) as connection:  # one transaction; rolls back on any error
        with connection.cursor() as cursor:
            cursor.execute((ROOT / "schema.sql").read_text(encoding="utf-8"))
            users, types = {}, {}

            # Interactions have no natural key and the CSV has real repeats
            # (e.g. two 10-minute Friend chats on one day), so compare counts
            # of identical rows and insert only the missing copies.
            wanted = Counter()
            for row in interactions:
                key = (
                    user_id_for(cursor, row["Name"].strip(), users),
                    date.fromisoformat(row["date"]),
                    type_id_for(cursor, type_name(row["type"]), types),
                    int(row["duration_min"]),
                )
                wanted[key] += 1

            added_interactions = 0
            for (user_id, day, type_id, minutes), count in wanted.items():
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM interactions
                    WHERE user_id = %s AND interaction_date = %s
                      AND type_id = %s AND duration_min = %s
                    """,
                    (user_id, day, type_id, minutes),
                )
                missing = count - cursor.fetchone()[0]
                for _ in range(max(missing, 0)):
                    cursor.execute(
                        """
                        INSERT INTO interactions (user_id, interaction_date, type_id, duration_min)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (user_id, day, type_id, minutes),
                    )
                    added_interactions += 1

            added_fatigue = 0
            for row in fatigue:
                user_id = user_id_for(cursor, row["Name"].strip(), users)
                day = date.fromisoformat(row["date"])
                score = int(row["fatigue_score"])
                cursor.execute(
                    """
                    INSERT INTO daily_fatigue (user_id, log_date, fatigue_score)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (user_id, log_date) DO NOTHING
                    """,
                    (user_id, day, score),
                )
                if cursor.rowcount:
                    added_fatigue += 1
                else:
                    cursor.execute(
                        "SELECT fatigue_score FROM daily_fatigue WHERE user_id = %s AND log_date = %s",
                        (user_id, day),
                    )
                    existing = cursor.fetchone()[0]
                    if existing != score:
                        print(
                            f"  kept existing fatigue {existing} for {row['Name']} on {day} "
                            f"(CSV says {score})"
                        )

            # The CSV's interaction_count / total_duration_min are derived values;
            # check them against what the database now holds.
            mismatches = 0
            for row in fatigue:
                cursor.execute(
                    """
                    SELECT interaction_count, total_duration_min FROM daily_summary
                    WHERE user_id = %s AND log_date = %s
                    """,
                    (users[row["Name"].strip()], date.fromisoformat(row["date"])),
                )
                count, minutes = cursor.fetchone()
                expected = (int(row["interaction_count"]), int(row["total_duration_min"]))
                if (count, minutes) != expected:
                    mismatches += 1
                    print(
                        f"  {row['Name']} {row['date']}: database has {count} interactions / "
                        f"{minutes} min, CSV says {expected[0]} / {expected[1]}"
                    )

    print(
        f"Done: {added_interactions} interactions and {added_fatigue} fatigue days added; "
        f"{mismatches} daily summary mismatches."
    )


if __name__ == "__main__":
    main()
