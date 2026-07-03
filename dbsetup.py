"""
dbsetup.py  –  Standout #3: MySQL database persistence layer.

Replaces all SQLite .db files and flat .txt/.csv records with a single
MySQL database (dra_mlops) containing three tables:

    ingested_files        – audit trail of every source file processed
    model_scores          – F1 score recorded after every scoring run
    diagnostics_snapshots – latency, missing-value %, Evidently drift per run

Connection settings are read from config.json → "database" block so the
credentials never appear hard-coded in application scripts.

Fallback behaviour
──────────────────
Every public function wraps its MySQL call in a try/except. When the
MySQL server is unavailable (e.g. during unit testing or CI), it logs a
warning and continues without raising – the rest of the pipeline keeps
working because the flat-file outputs (ingestedfiles.txt, latestscore.txt,
olddiagnostics/*.json) are always written first.

Local setup (one-time, run as MySQL root)
──────────────────────────────────────────
  CREATE DATABASE dra_mlops CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
  CREATE USER 'dra_user'@'localhost' IDENTIFIED BY 'dra_password_2026';
  GRANT ALL PRIVILEGES ON dra_mlops.* TO 'dra_user'@'localhost';
  FLUSH PRIVILEGES;
"""

import json
import os
from datetime import datetime

import mysql.connector
from mysql.connector import Error as MySQLError

# ── Load config ────────────────────────────────────────────────────────────
with open("config.json", "r") as _f:
    _cfg = json.load(_f)

_DB = _cfg["database"]          # host / port / user / password / database / charset


# ── Connection factory ─────────────────────────────────────────────────────
def get_conn():
    """Return a live mysql.connector connection using config.json credentials."""
    return mysql.connector.connect(
        host=_DB["host"],
        port=int(_DB["port"]),
        user=_DB["user"],
        password=_DB["password"],
        database=_DB["database"],
        charset=_DB.get("charset", "utf8mb4"),
        connection_timeout=10,
        autocommit=False,
    )


# ── Schema bootstrap ───────────────────────────────────────────────────────
_TABLES = {
    "ingested_files": """
        CREATE TABLE IF NOT EXISTS ingested_files (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            filepath    VARCHAR(1024)   NOT NULL,
            ingested_at DATETIME        NOT NULL,
            INDEX idx_filepath (filepath(255))
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "model_scores": """
        CREATE TABLE IF NOT EXISTS model_scores (
            id          INT AUTO_INCREMENT PRIMARY KEY,
            f1_score    DOUBLE          NOT NULL,
            scored_at   DATETIME        NOT NULL,
            model_path  VARCHAR(512),
            data_path   VARCHAR(512),
            INDEX idx_scored_at (scored_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
    "diagnostics_snapshots": """
        CREATE TABLE IF NOT EXISTS diagnostics_snapshots (
            id               INT AUTO_INCREMENT PRIMARY KEY,
            recorded_at      DATETIME        NOT NULL,
            ingestion_time   DOUBLE,
            training_time    DOUBLE,
            avg_missing_pct  DOUBLE,
            summary_stats    LONGTEXT,
            missing_data     LONGTEXT,
            evidently_drift  LONGTEXT,
            INDEX idx_recorded_at (recorded_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """,
}


def init_all_tables():
    """Create all three tables in dra_mlops if they don't already exist."""
    try:
        conn = get_conn()
        cursor = conn.cursor()
        for name, ddl in _TABLES.items():
            cursor.execute(ddl)
            print(f"[MySQL] Table ready: {name}")
        conn.commit()
        cursor.close()
        conn.close()
    except MySQLError as e:
        print(f"[MySQL] WARNING – could not initialise tables: {e}")


# ── ingested_files ─────────────────────────────────────────────────────────
def insert_ingested_files(filepaths: list) -> None:
    """Record each filepath in ingested_files with the current timestamp."""
    if not filepaths:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [(fp, ts) for fp in filepaths]
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO ingested_files (filepath, ingested_at) VALUES (%s, %s)",
            rows,
        )
        conn.commit()
        print(f"[MySQL] Recorded {len(filepaths)} ingested file(s) → ingested_files")
        cursor.close()
        conn.close()
    except MySQLError as e:
        print(f"[MySQL] WARNING – insert_ingested_files failed: {e}")


def get_ingested_files() -> list:
    """Return every filepath ever recorded in ingested_files."""
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT filepath FROM ingested_files")
        rows = [r[0] for r in cursor.fetchall()]
        cursor.close()
        conn.close()
        return rows
    except MySQLError as e:
        print(f"[MySQL] WARNING – get_ingested_files failed: {e}")
        return []


def write_ingested_txt(output_folder: str, filepaths: list) -> str:
    """Write ingestedfiles.txt (backward-compat) AND insert into MySQL."""
    txt_path = os.path.join(output_folder, "ingestedfiles.txt")
    with open(txt_path, "w") as f:
        for fp in filepaths:
            f.write(fp + "\n")
    insert_ingested_files(filepaths)
    return txt_path


# ── model_scores ───────────────────────────────────────────────────────────
def insert_score(f1: float, model_path: str = "", data_path: str = "") -> None:
    """Append one F1-score row to model_scores."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO model_scores
               (f1_score, scored_at, model_path, data_path)
               VALUES (%s, %s, %s, %s)""",
            (float(f1), ts, model_path, data_path),
        )
        conn.commit()
        print(f"[MySQL] Recorded F1={f1:.4f} → model_scores")
        cursor.close()
        conn.close()
    except MySQLError as e:
        print(f"[MySQL] WARNING – insert_score failed: {e}")


def get_score_history() -> list:
    """Return all rows from model_scores ordered by scored_at (oldest first)."""
    try:
        conn = get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT f1_score, scored_at, model_path "
            "FROM model_scores ORDER BY scored_at ASC"
        )
        rows = cursor.fetchall()
        # Convert datetime objects to ISO strings for JSON-serializability
        for r in rows:
            if hasattr(r.get("scored_at"), "isoformat"):
                r["scored_at"] = r["scored_at"].isoformat()
        cursor.close()
        conn.close()
        return rows
    except MySQLError as e:
        print(f"[MySQL] WARNING – get_score_history failed: {e}")
        return []


def write_score_txt(model_path_dir: str, f1: float) -> str:
    """Write latestscore.txt AND insert into MySQL."""
    txt_path = os.path.join(model_path_dir, "latestscore.txt")
    with open(txt_path, "w") as f:
        f.write(str(f1))
    insert_score(f1, model_path=model_path_dir)
    return txt_path


# ── diagnostics_snapshots ──────────────────────────────────────────────────
def insert_diagnostics(
    ingestion_time: float,
    training_time: float,
    summary_stats: list,
    missing_data: list,
    evidently_drift: dict = None,
) -> None:
    """Append one diagnostics snapshot row."""
    import json as _j

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    miss_vals = [m.get("missing_pct", 0) for m in missing_data]
    avg_miss = sum(miss_vals) / len(miss_vals) if miss_vals else 0.0

    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO diagnostics_snapshots
               (recorded_at, ingestion_time, training_time, avg_missing_pct,
                summary_stats, missing_data, evidently_drift)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            (
                ts,
                float(ingestion_time),
                float(training_time),
                float(avg_miss),
                _j.dumps(summary_stats, default=str),
                _j.dumps(missing_data, default=str),
                _j.dumps(evidently_drift or {}, default=str),
            ),
        )
        conn.commit()
        print(f"[MySQL] Diagnostics snapshot recorded → diagnostics_snapshots")
        cursor.close()
        conn.close()
    except MySQLError as e:
        print(f"[MySQL] WARNING – insert_diagnostics failed: {e}")


def get_diagnostics_history() -> list:
    """Return all snapshot rows ordered chronologically."""
    import json as _j

    try:
        conn = get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """SELECT recorded_at, ingestion_time, training_time,
                      avg_missing_pct, evidently_drift
               FROM diagnostics_snapshots
               ORDER BY recorded_at ASC"""
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        result = []
        for r in rows:
            if hasattr(r.get("recorded_at"), "isoformat"):
                r["recorded_at"] = r["recorded_at"].isoformat()
            try:
                r["evidently_drift"] = _j.loads(r["evidently_drift"] or "{}")
            except Exception:
                r["evidently_drift"] = {}
            result.append(r)
        return result
    except MySQLError as e:
        print(f"[MySQL] WARNING – get_diagnostics_history failed: {e}")
        return []


# ── Convenience: migrate data from SQLite (run once) ──────────────────────
def migrate_from_sqlite():
    """
    One-time migration: reads existing SQLite .db files (if present) and
    inserts their rows into MySQL, then renames the .db files to .db.bak.
    Safe to run multiple times – skips already-migrated files.
    """
    import sqlite3, glob as _glob

    db_dir = _cfg.get("output_folder_path", "ingesteddata")
    migrated = 0

    for db_file in _glob.glob(os.path.join(db_dir, "*.db")):
        bak = db_file + ".bak"
        if os.path.exists(bak):
            print(f"[migrate] Skipping {db_file} (already migrated)")
            continue
        print(f"[migrate] Migrating {db_file} → MySQL …")
        try:
            sq = sqlite3.connect(db_file)
            sq.row_factory = sqlite3.Row

            # ingested_files
            try:
                rows = sq.execute("SELECT filepath FROM ingested_files").fetchall()
                fps = [r["filepath"] for r in rows]
                if fps:
                    insert_ingested_files(fps)
            except Exception:
                pass

            # model_scores
            try:
                rows = sq.execute(
                    "SELECT f1_score, model_path FROM model_scores"
                ).fetchall()
                for r in rows:
                    insert_score(r["f1_score"], model_path=r["model_path"] or "")
            except Exception:
                pass

            # diagnostics_snapshots
            try:
                rows = sq.execute(
                    "SELECT ingestion_time, training_time, avg_missing_pct, "
                    "       summary_stats, missing_data, evidently_drift "
                    "FROM diagnostics_snapshots"
                ).fetchall()
                import json as _j
                for r in rows:
                    insert_diagnostics(
                        r["ingestion_time"] or 0,
                        r["training_time"] or 0,
                        _j.loads(r["summary_stats"] or "[]"),
                        _j.loads(r["missing_data"] or "[]"),
                        _j.loads(r["evidently_drift"] or "{}"),
                    )
            except Exception:
                pass

            sq.close()
            os.rename(db_file, bak)
            print(f"[migrate] Done – renamed to {bak}")
            migrated += 1
        except Exception as ex:
            print(f"[migrate] ERROR on {db_file}: {ex}")

    print(f"[migrate] Migration complete. {migrated} file(s) processed.")


# ── CLI entry-point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DRA MySQL DB setup utility")
    parser.add_argument(
        "--migrate-sqlite",
        action="store_true",
        help="One-time migration of existing SQLite .db files into MySQL",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DRA MLOps – MySQL Database Setup")
    print("=" * 60)
    print(f"  Host     : {_DB['host']}:{_DB['port']}")
    print(f"  Database : {_DB['database']}")
    print(f"  User     : {_DB['user']}")
    print()

    # Test connection first
    try:
        c = get_conn()
        c.close()
        print("[MySQL] Connection: OK")
    except MySQLError as e:
        print(f"[MySQL] Connection FAILED: {e}")
        print()
        print("Run the following SQL as MySQL root to create the DB and user:")
        print()
        print(f"  CREATE DATABASE {_DB['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        print(f"  CREATE USER '{_DB['user']}'@'localhost' IDENTIFIED BY '{_DB['password']}';")
        print(f"  GRANT ALL PRIVILEGES ON {_DB['database']}.* TO '{_DB['user']}'@'localhost';")
        print(f"  FLUSH PRIVILEGES;")
        raise SystemExit(1)

    init_all_tables()

    if args.migrate_sqlite:
        migrate_from_sqlite()

    # Smoke-test round-trip
    print()
    print("Running smoke tests …")
    insert_ingested_files(["smoke_test/dummy.csv"])
    files = get_ingested_files()
    assert any("dummy.csv" in f for f in files), "ingested_files round-trip FAILED"
    print("  ingested_files : PASS")

    insert_score(0.9999, model_path="smoke_test")
    scores = get_score_history()
    assert any(abs(s["f1_score"] - 0.9999) < 1e-6 for s in scores), "model_scores round-trip FAILED"
    print("  model_scores   : PASS")

    insert_diagnostics(0.1, 0.2, [], [], {"test": True})
    hist = get_diagnostics_history()
    assert len(hist) > 0, "diagnostics_snapshots round-trip FAILED"
    print("  diagnostics    : PASS")

    print()
    print("All checks passed. MySQL backend is ready.")
