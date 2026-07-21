#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from semsearch.share.config import get_settings

COMPOSE = ["podman", "compose"]
TOKENIZER = "semsearch_llmlingua2"


class MigrationError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace a tsvector database with a fresh VectorChord BM25 database."
    )
    parser.add_argument(
        "backup_dir",
        type=Path,
        help="new directory that will receive database.dump, chunks.csv, and pgdata",
    )
    return parser.parse_args()


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _capture(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_command_output(command: list[str], path: Path) -> None:
    with path.open("wb") as output:
        subprocess.run(command, check=True, stdout=output)


def _read_command_input(command: list[str], path: Path) -> None:
    with path.open("rb") as input_file:
        subprocess.run(command, check=True, stdin=input_file)


def _db_command(program: str, *args: str) -> list[str]:
    return [
        *COMPOSE,
        "exec",
        "-T",
        "db",
        "sh",
        "-c",
        'exec "$0" -U "$POSTGRES_USER" -d "$POSTGRES_DB" "$@"',
        program,
        *args,
    ]


def _psql_command(sql: str) -> list[str]:
    return _db_command(
        "psql",
        "-X",
        "--quiet",
        "--tuples-only",
        "--no-align",
        "--set=ON_ERROR_STOP=1",
        "--command",
        sql,
    )


def _wait_for_database() -> None:
    command = _db_command("pg_isready", "--timeout=5")
    for _ in range(30):
        if (
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        ):
            return
        time.sleep(2)
    raise MigrationError("fresh database did not become ready within 60 seconds")


def _validated_paths(backup_arg: Path) -> tuple[Path, Path]:
    raw_pgdata = os.environ.get("PGDATA_DIR")
    if not raw_pgdata:
        raise MigrationError("PGDATA_DIR must be exported")

    pgdata_input = Path(raw_pgdata).expanduser()
    if not pgdata_input.is_absolute():
        raise MigrationError("PGDATA_DIR must be absolute")
    if pgdata_input.is_symlink():
        raise MigrationError("PGDATA_DIR must not be a symlink")
    pgdata = pgdata_input.resolve(strict=True)
    if pgdata == Path("/") or not pgdata.is_dir():
        raise MigrationError("PGDATA_DIR must be an existing database directory")

    backup_input = backup_arg.expanduser()
    if not backup_input.is_absolute():
        raise MigrationError("backup_dir must be absolute")
    backup = backup_input.resolve(strict=False)
    if backup.exists():
        raise MigrationError("backup_dir must not already exist")
    if backup == pgdata or backup.is_relative_to(pgdata):
        raise MigrationError("backup_dir must be outside PGDATA_DIR")
    return pgdata, backup


def _require_tools() -> None:
    missing = [name for name in ("mv", "podman", "uv") if shutil.which(name) is None]
    if missing:
        raise MigrationError(f"missing required command(s): {', '.join(missing)}")


def _require_quiescent_database() -> None:
    active_clients = _capture(
        _psql_command(
            """
            SELECT count(*)
            FROM pg_stat_activity
            WHERE datname = current_database()
              AND backend_type = 'client backend'
              AND pid <> pg_backend_pid()
            """
        )
    )
    if active_clients != "0":
        raise MigrationError(
            f"database still has {active_clients} external client session(s)"
        )


def _export_database(backup: Path) -> tuple[Path, Path]:
    dump_path = backup / "database.dump"
    chunks_path = backup / "chunks.csv"

    print(f"Writing full dump to {dump_path}")
    _write_command_output(
        _db_command("pg_dump", "--format=custom"),
        dump_path,
    )
    print(f"Writing projected chunks to {chunks_path}")
    _write_command_output(
        _psql_command(
            """
            COPY (
                SELECT id, page_id, start_offset, content_length, embedding::text
                FROM chunks
                ORDER BY id
            ) TO STDOUT WITH (FORMAT csv, HEADER true)
            """
        ),
        chunks_path,
    )
    return dump_path, chunks_path


def _restore_database(dump_path: Path, chunks_path: Path, embedding_dim: int) -> None:
    _run(["uv", "run", "semsearch", "init-db"])
    _run(
        _psql_command(
            """
            DROP INDEX chunks_embedding_hnsw_idx;
            DROP INDEX chunks_search_vector_bm25_idx;
            """
        )
    )

    _read_command_input(
        _db_command(
            "pg_restore",
            "--data-only",
            "--exit-on-error",
            "--single-transaction",
            "--disable-triggers",
            "--no-owner",
            "--no-privileges",
            "--table=sites",
            "--table=crawl_jobs",
            "--table=pages",
        ),
        dump_path,
    )

    _run(
        _psql_command(
            """
            CREATE UNLOGGED TABLE chunks_migration (
                id bigint PRIMARY KEY,
                page_id bigint NOT NULL,
                start_offset int NOT NULL,
                content_length int NOT NULL,
                embedding text NOT NULL
            )
            """
        )
    )
    _read_command_input(
        _psql_command(
            """
            COPY chunks_migration
                (id, page_id, start_offset, content_length, embedding)
            FROM STDIN WITH (FORMAT csv, HEADER true)
            """
        ),
        chunks_path,
    )
    _run(
        _psql_command(
            f"""
            INSERT INTO chunks
                (id, page_id, start_offset, content_length, embedding, search_vector)
            SELECT m.id,
                   m.page_id,
                   m.start_offset,
                   m.content_length,
                   m.embedding::halfvec({embedding_dim}),
                   tokenize(
                       substring(
                           p.content
                           FROM m.start_offset + 1
                           FOR m.content_length
                       ),
                       '{TOKENIZER}'
                   )::bm25vector
            FROM chunks_migration m
            JOIN pages p ON p.id = m.page_id
            ORDER BY m.id;

            DROP TABLE chunks_migration;

            CREATE INDEX chunks_embedding_hnsw_idx
                ON chunks USING hnsw (embedding halfvec_cosine_ops);
            CREATE INDEX chunks_search_vector_bm25_idx
                ON chunks USING bm25 (search_vector bm25_ops);

            SELECT setval(
                pg_get_serial_sequence('sites', 'id'),
                COALESCE(max(id), 1),
                max(id) IS NOT NULL
            ) FROM sites;
            SELECT setval(
                pg_get_serial_sequence('crawl_jobs', 'id'),
                COALESCE(max(id), 1),
                max(id) IS NOT NULL
            ) FROM crawl_jobs;
            SELECT setval(
                pg_get_serial_sequence('pages', 'id'),
                COALESCE(max(id), 1),
                max(id) IS NOT NULL
            ) FROM pages;
            SELECT setval(
                pg_get_serial_sequence('chunks', 'id'),
                COALESCE(max(id), 1),
                max(id) IS NOT NULL
            ) FROM chunks;

            ANALYZE;
            """
        )
    )


def migrate(pgdata: Path, backup: Path) -> None:
    settings = get_settings()
    _require_tools()
    _require_quiescent_database()

    backup.mkdir(parents=True)
    dump_path, chunks_path = _export_database(backup)

    print("Stopping database")
    _run([*COMPOSE, "stop", "db"])
    try:
        old_pgdata = backup / "pgdata"
        print(f"Moving {pgdata} to {old_pgdata}")
        _run(["mv", "--", str(pgdata), str(old_pgdata)])
        pgdata.mkdir(mode=0o700)

        print("Starting fresh VectorChord database")
        _run([*COMPOSE, "up", "--force-recreate", "-d", "db"])
        _wait_for_database()
        _restore_database(dump_path, chunks_path, settings.embedding_dim)
    except MigrationError, OSError, subprocess.CalledProcessError:
        subprocess.run([*COMPOSE, "stop", "db"], check=False)
        raise
    print("Migration complete; restart application services after inspection")


def main() -> int:
    args = parse_args()
    try:
        pgdata, backup = _validated_paths(args.backup_dir)
        migrate(pgdata, backup)
    except (MigrationError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Migration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
