"""Small database-backed Social Battery deployment skeleton."""

import os

import psycopg
from dotenv import load_dotenv
from flask import Flask, render_template, request
from psycopg.rows import dict_row

load_dotenv()
app = Flask(__name__)


def database_url():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    return url


@app.get("/")
def index():
    with psycopg.connect(database_url(), row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, name FROM users ORDER BY name, id")
            people = cursor.fetchall()

            selected_id = request.args.get("user_id", type=int)
            if selected_id is None and people:
                selected_id = people[0]["id"]

            selected_person = next(
                (person for person in people if person["id"] == selected_id), None
            )
            summary = None
            if selected_person:
                cursor.execute(
                    """
                    SELECT COUNT(*) AS interaction_count,
                           COALESCE(SUM(duration_min), 0) AS total_duration_min
                    FROM interactions
                    WHERE user_id = %s
                    """,
                    (selected_id,),
                )
                interaction_totals = cursor.fetchone()
                cursor.execute(
                    """
                    SELECT COUNT(*) AS fatigue_days,
                           ROUND(AVG(fatigue_score)::numeric, 1) AS average_fatigue
                    FROM daily_fatigue
                    WHERE user_id = %s
                    """,
                    (selected_id,),
                )
                summary = {**interaction_totals, **cursor.fetchone()}

    return render_template(
        "index.html",
        people=people,
        selected_person=selected_person,
        summary=summary,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")))
