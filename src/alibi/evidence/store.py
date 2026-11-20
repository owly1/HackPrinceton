"""Read-only evidence database (DuckDB), one file per episode.

The store is built once by trusted code, closed, hashed, and afterwards only opened
read-only with external access disabled. Callers get *typed* filters with
parameterized queries and hard result caps. There is deliberately no method that
accepts SQL text: a SELECT-only check is not a boundary, because table functions can
read files.
"""

from __future__ import annotations

import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

from alibi.contracts import ArtifactRef, Event, iso, utc

_COLUMNS = (
    "id", "ts", "ts_raw", "host", "source", "actor", "action", "object", "attrs", "raw",
    "artifact", "line_no", "byte_offset", "origin_host", "origin_file", "session_id", "logical_file", "seq",
)
GROUPABLE = ("source", "host", "actor", "action", "origin_host", "logical_file", "artifact")
MAX_LIMIT = 50


def _naive(dt: datetime) -> str:
    """UTC wall time without offset: the DB stores naive UTC timestamps.

    Sub-second precision is kept when the record has it (captured logs do; simulated ones never do), so two
    records inside one second keep their real order. A whole-second time is written exactly as before.
    """
    dt = utc(dt)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f" if dt.microsecond else "%Y-%m-%d %H:%M:%S")


def build_store(db_path: Path, events: list[Event]) -> None:
    """Write events into a new DuckDB file and close it. Trusted build-time code only."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        raise FileExistsError(f"refusing to overwrite derived evidence DB {db_path}")
    tmp_csv = db_path.with_suffix(".load.csv")
    with tmp_csv.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh, quoting=csv.QUOTE_ALL)
        for ev in events:
            a = ev.attrs
            wr.writerow([
                ev.id, _naive(ev.ts), ev.ts_raw, ev.host, ev.source, ev.actor or "", ev.action, ev.object or "",
                json.dumps(a, sort_keys=True, separators=(",", ":")), ev.raw, ev.artifact_ref.artifact,
                ev.artifact_ref.line, ev.artifact_ref.byte_offset, a.get("origin_host") or "",
                a.get("origin_file") or "", a.get("session_id") or "", a.get("logical_file") or "",
                a.get("seq") if a.get("seq") is not None else "",
            ])
    con = duckdb.connect(str(db_path))
    try:
        con.execute(
            """CREATE TABLE events (
                id VARCHAR PRIMARY KEY, ts TIMESTAMP, ts_raw VARCHAR, host VARCHAR, source VARCHAR,
                actor VARCHAR, action VARCHAR, object VARCHAR, attrs VARCHAR, raw VARCHAR,
                artifact VARCHAR, line_no INTEGER, byte_offset BIGINT, origin_host VARCHAR,
                origin_file VARCHAR, session_id VARCHAR, logical_file VARCHAR, seq BIGINT)"""
        )
        _load_csv(con, tmp_csv)
        con.execute("CHECKPOINT")
    finally:
        con.close()
        tmp_csv.unlink(missing_ok=True)


def _load_csv(con: duckdb.DuckDBPyConnection, tmp_csv: Path) -> None:
    cols = (
        "{'id':'VARCHAR','ts':'VARCHAR','ts_raw':'VARCHAR','host':'VARCHAR','source':'VARCHAR',"
        "'actor':'VARCHAR','action':'VARCHAR','object':'VARCHAR','attrs':'VARCHAR','raw':'VARCHAR',"
        "'artifact':'VARCHAR','line_no':'INTEGER','byte_offset':'BIGINT','origin_host':'VARCHAR',"
        "'origin_file':'VARCHAR','session_id':'VARCHAR','logical_file':'VARCHAR','seq':'VARCHAR'}"
    )
    con.execute(
        f"""INSERT INTO events
            SELECT id, CAST(ts AS TIMESTAMP), ts_raw, host, source, actor, action, object, attrs, raw,
                   artifact, line_no, byte_offset, origin_host, origin_file, session_id, logical_file,
                   TRY_CAST(NULLIF(seq, '') AS BIGINT)
            FROM read_csv(?, header=false, quote='"', escape='"', auto_detect=false, columns={cols},
                          max_line_size=4000000)""",
        [str(tmp_csv)],
    )


class EvidenceStore:
    """Read-only, typed access to one episode's events."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._con = duckdb.connect(str(self.db_path), read_only=True, config={"enable_external_access": "false"})
        try:
            self._con.execute("SET lock_configuration = true")
        except duckdb.Error:  # pragma: no cover - defense in depth only
            pass
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._con.close()

    def __enter__(self) -> "EvidenceStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _where(filters: dict[str, Any]) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        for col in ("source", "host", "actor", "action", "origin_host", "origin_file", "session_id",
                    "logical_file", "artifact", "object"):
            val = filters.get(col)
            if val is None:
                continue
            if isinstance(val, (list, tuple, set)):
                vals = list(val)
                clauses.append(f"{col} IN ({','.join('?' * len(vals))})")
                params.extend(vals)
            else:
                clauses.append(f"{col} = ?")
                params.append(val)
        if filters.get("file"):  # the host's own file, or the collector's copies of that file
            clauses.append("(logical_file = ? OR origin_file = ?)")
            params += [str(filters["file"]), str(filters["file"])]
        if filters.get("min_bytes_out") is not None:  # flow records at least this large
            clauses.append("TRY_CAST(json_extract_string(attrs, '$.bytes_out') AS BIGINT) >= ?")
            params.append(int(filters["min_bytes_out"]))
        if filters.get("object_contains"):
            clauses.append("contains(object, ?)")
            params.append(str(filters["object_contains"]))
        if filters.get("text_contains"):
            clauses.append("contains(raw, ?)")
            params.append(str(filters["text_contains"]))
        if filters.get("t_start") is not None:
            clauses.append("ts >= CAST(? AS TIMESTAMP)")
            params.append(_naive(filters["t_start"]))
        if filters.get("t_end") is not None:
            # times are shown to the second, so an upper bound given to the second means "through that second":
            # a record at 12:00:05.3 is inside a window that ends at 12:00:05
            whole_second = utc(filters["t_end"]).microsecond == 0
            clauses.append("ts < CAST(? AS TIMESTAMP) + INTERVAL 1 SECOND" if whole_second else "ts <= CAST(? AS TIMESTAMP)")
            params.append(_naive(filters["t_end"]))
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    @staticmethod
    def _row(row: tuple) -> Event:
        d = dict(zip(_COLUMNS, row))
        ts = d["ts"]
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        ts = ts.replace(tzinfo=timezone.utc)  # stored as naive UTC
        return Event(
            id=d["id"], ts=ts, ts_raw=d["ts_raw"], host=d["host"], source=d["source"],
            actor=d["actor"] or None, action=d["action"], object=d["object"] or None,
            attrs=json.loads(d["attrs"]), raw=d["raw"],
            artifact_ref=ArtifactRef(artifact=d["artifact"], line=d["line_no"], byte_offset=d["byte_offset"]),
        )

    def _query(self, sql: str, params: list[Any]) -> list[tuple]:
        with self._lock:
            return self._con.execute(sql, params).fetchall()

    _SELECT = "SELECT " + ", ".join(_COLUMNS) + " FROM events"

    # ------------------------------------------------------------------ typed API
    def search(self, *, limit: int = MAX_LIMIT, offset: int = 0, newest_first: bool = False,
               **filters: Any) -> list[Event]:
        """Capped search for interactive/agent use."""
        limit = max(1, min(int(limit), MAX_LIMIT))
        offset = max(0, int(offset))
        where, params = self._where(filters)
        order = "DESC" if newest_first else "ASC"
        sql = f"{self._SELECT}{where} ORDER BY ts {order}, artifact, line_no LIMIT ? OFFSET ?"
        return [self._row(r) for r in self._query(sql, params + [limit, offset])]

    def scan(self, **filters: Any) -> list[Event]:
        """Uncapped scan for trusted deterministic checks. Never exposed as an agent tool."""
        where, params = self._where(filters)
        return [self._row(r) for r in self._query(f"{self._SELECT}{where} ORDER BY ts, artifact, line_no", params)]

    def count(self, **filters: Any) -> int:
        where, params = self._where(filters)
        return int(self._query(f"SELECT count(*) FROM events{where}", params)[0][0])

    def count_by(self, group_by: str, **filters: Any) -> list[dict[str, Any]]:
        if group_by not in GROUPABLE:
            raise ValueError(f"group_by must be one of {GROUPABLE}")
        where, params = self._where(filters)
        rows = self._query(
            f"SELECT {group_by}, count(*) AS n, min(ts), max(ts) FROM events{where} "
            f"GROUP BY {group_by} ORDER BY n DESC, {group_by} LIMIT 100", params)
        return [{"key": r[0], "count": int(r[1]), "first_ts": iso(r[2].replace(tzinfo=timezone.utc)),
                 "last_ts": iso(r[3].replace(tzinfo=timezone.utc))} for r in rows]

    def get(self, ids: list[str]) -> list[Event]:
        ids = [str(i) for i in ids][:200]
        if not ids:
            return []
        rows = self._query(f"{self._SELECT} WHERE id IN ({','.join('?' * len(ids))})", ids)
        by_id = {r[0]: self._row(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def existing_ids(self, ids: list[str]) -> set[str]:
        ids = list(dict.fromkeys(str(i) for i in ids))
        found: set[str] = set()
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            rows = self._query(f"SELECT id FROM events WHERE id IN ({','.join('?' * len(chunk))})", chunk)
            found.update(r[0] for r in rows)
        return found

    def context(self, artifact: str, line: int, radius: int = 4) -> list[Event]:
        radius = max(0, min(int(radius), 25))
        rows = self._query(
            f"{self._SELECT} WHERE artifact = ? AND line_no BETWEEN ? AND ? ORDER BY line_no",
            [artifact, line - radius * 2, line + radius * 2])
        return [self._row(r) for r in rows]

    def time_bounds(self) -> tuple[datetime, datetime]:
        row = self._query("SELECT min(ts), max(ts) FROM events WHERE action <> 'unparsed'", [])[0]
        return row[0].replace(tzinfo=timezone.utc), row[1].replace(tzinfo=timezone.utc)
