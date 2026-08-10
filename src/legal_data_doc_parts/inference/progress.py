"""Durable run state and the batched ``is_classified`` write-back.

Ordering invariant for the whole pipeline: **GCS upload -> manifest record -> BigQuery
flush**. So ``is_classified = TRUE`` can never be true for a document with no
``classification.parquet``. The reverse -- file written, flag not yet flipped -- is harmless
and self-heals on the next flush or via ``--sync-bq``.
"""

from __future__ import annotations

import fcntl
import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from google.api_core import exceptions as gcp_exceptions
from google.cloud import bigquery

from legal_data_doc_parts.inference.config import DOCUMENT_TABLE

logger = logging.getLogger(__name__)

MARK_CLASSIFIED_SQL = f"""
UPDATE `{DOCUMENT_TABLE}`
SET is_classified = TRUE
WHERE doc_id IN UNNEST(@ids)
"""

# BigQuery serialises DML per table; concurrent shards collide rather than corrupt.
_SERIALIZE_RETRIES = 5


@contextmanager
def _file_lock(path: Path):
    """Serialise BigQuery DML between processes on this machine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


class RunLog:
    """Append-only manifest of finished documents, plus the failure log."""

    def __init__(self, manifest_path: Path, failures_path: Path):
        self.manifest_path = manifest_path
        self.failures_path = failures_path
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def completed_ids(self) -> set[int]:
        return self._read_ids(self.manifest_path)

    def failed_ids(self) -> set[int]:
        return self._read_ids(self.failures_path)

    @staticmethod
    def _read_ids(path: Path) -> set[int]:
        if not path.exists():
            return set()
        ids = set()
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(int(json.loads(line)["doc_id"]))
                except (json.JSONDecodeError, KeyError, ValueError):
                    # A run killed mid-write can leave one torn line; skip it.
                    logger.warning("skipping unreadable line in %s", path)
        return ids

    def record_success(self, doc_id: int, **fields) -> None:
        self._append(self.manifest_path, {"doc_id": int(doc_id), **fields})

    def record_failure(self, doc_id: int, error: str, **fields) -> None:
        self._append(self.failures_path, {"doc_id": int(doc_id), "error": error, **fields})

    def _append(self, path: Path, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock, path.open("a") as handle:
            handle.write(line + "\n")
            handle.flush()


class BigQueryProgress:
    """Buffers finished doc ids and flips ``is_classified`` in batches.

    Every UPDATE scans the whole (unpartitioned, unclustered) 800 MB table regardless of how
    many ids it targets, so the flush size is what makes this affordable: 10,000 puts the
    full corpus at ~162 statements and under a dollar, where per-document updates would be
    1.6M statements and thousands of dollars.
    """

    def __init__(
        self,
        pending_path: Path,
        lock_path: Path,
        flush_size: int = 10_000,
        client: bigquery.Client | None = None,
        enabled: bool = True,
    ):
        self.pending_path = pending_path
        self.lock_path = lock_path
        self.flush_size = flush_size
        self.enabled = enabled
        self._client = client
        self._lock = threading.Lock()
        self._pending: list[int] = []
        if enabled:
            self.pending_path.parent.mkdir(parents=True, exist_ok=True)
            self._pending = sorted(self._load_pending())
            if self._pending:
                logger.info("recovered %d ids pending BigQuery update", len(self._pending))

    @property
    def client(self) -> bigquery.Client:
        if self._client is None:
            self._client = bigquery.Client(project=DOCUMENT_TABLE.split(".")[0])
        return self._client

    def _load_pending(self) -> set[int]:
        if not self.pending_path.exists():
            return set()
        ids = set()
        with self.pending_path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    try:
                        ids.add(int(line))
                    except ValueError:
                        continue
        return ids

    def add(self, doc_id: int) -> None:
        """Queue a document. Persisted immediately so a crash cannot lose the intent."""
        if not self.enabled:
            return
        with self._lock:
            self._pending.append(int(doc_id))
            with self.pending_path.open("a") as handle:
                handle.write(f"{int(doc_id)}\n")
            should_flush = len(self._pending) >= self.flush_size
        if should_flush:
            self.flush()

    def flush(self, force: bool = False) -> int:
        """Run one UPDATE for everything buffered. Returns rows marked."""
        if not self.enabled:
            return 0
        with self._lock:
            ids = sorted(set(self._pending))
            if not ids or (not force and len(ids) < self.flush_size):
                return 0
            self._pending = []

        try:
            marked = self._run_update(ids)
        except Exception:
            # Keep them queued: pending_bq.jsonl on disk is already the durable record, so
            # the next flush (or --sync-bq) retries them.
            with self._lock:
                self._pending.extend(ids)
            logger.exception("BigQuery flush of %d ids failed; ids remain pending", len(ids))
            return 0

        # Only now is the on-disk queue safe to clear.
        self.pending_path.write_text("")
        logger.info("BigQuery: marked %d documents classified", marked)
        return marked

    def _run_update(self, ids: list[int]) -> int:
        job_config = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("ids", "INT64", ids)]
        )
        for attempt in range(_SERIALIZE_RETRIES):
            try:
                with _file_lock(self.lock_path):
                    job = self.client.query(MARK_CLASSIFIED_SQL, job_config=job_config)
                    job.result()
                return job.num_dml_affected_rows or 0
            except gcp_exceptions.Conflict:
                delay = 2**attempt
                logger.warning(
                    "BigQuery DML conflict (attempt %d/%d), retrying in %ds",
                    attempt + 1, _SERIALIZE_RETRIES, delay,
                )
                time.sleep(delay)
        raise RuntimeError(f"BigQuery UPDATE still conflicting after {_SERIALIZE_RETRIES} attempts")


def sync_from_manifest(
    manifest_path: Path,
    pending_path: Path,
    lock_path: Path,
    client: bigquery.Client | None = None,
    batch_size: int = 10_000,
) -> int:
    """Reconcile ``is_classified`` from the local manifest.

    The manifest is the source of truth for what was actually written, so this makes the flag
    recoverable after any interruption. Idempotent -- re-marking an already-true row is a
    no-op in BigQuery terms.
    """
    log = RunLog(manifest_path, manifest_path.with_name("failures.jsonl"))
    ids = sorted(log.completed_ids())
    if not ids:
        logger.info("nothing to sync: manifest is empty")
        return 0

    progress = BigQueryProgress(pending_path, lock_path, flush_size=batch_size, client=client)
    total = 0
    for start in range(0, len(ids), batch_size):
        total += progress._run_update(ids[start : start + batch_size])
        logger.info("synced %d/%d", min(start + batch_size, len(ids)), len(ids))
    return total
