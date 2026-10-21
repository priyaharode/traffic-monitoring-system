from sqlalchemy import text
from database import engine

with engine.begin() as connection:
    columns = connection.execute(
        text("PRAGMA table_info(violations)")
    ).fetchall()
    column_names = [column[1] for column in columns]
    if "evidence_hash" not in column_names:
        connection.execute(
            text(
                "ALTER TABLE violations "
                "ADD COLUMN evidence_hash VARCHAR(64)"
            )
        )
        print("evidence_hash column added.")
    else:
        print("evidence_hash column already exists.")
