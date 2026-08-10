"""Selecting which documents to classify, from BigQuery."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

from legal_data_doc_parts.inference.config import DOCUMENT_TABLE, PROJECT_ID

logger = logging.getLogger(__name__)

PENDING_QUERY = f"""
SELECT doc_id, justice_kind
FROM `{DOCUMENT_TABLE}`
WHERE is_parsed = TRUE AND is_classified = FALSE
"""

PROTECTED_QUERY = f"""
SELECT doc_id
FROM `{DOCUMENT_TABLE}`
WHERE is_classified = TRUE
"""


def bigquery_client() -> bigquery.Client:
    return bigquery.Client(project=PROJECT_ID)


def load_protected_ids(path: Path, client: bigquery.Client | None = None) -> set[int]:
    """Doc ids whose classification.parquet holds Lapa-LLM ground truth.

    Snapshotted to disk on first call and read from there forever after. This must never be
    re-derived from a live ``is_classified = TRUE`` query once the pipeline has begun
    flipping that flag, or the protected set silently grows to include our own output and
    stops protecting anything.
    """
    if path.exists():
        ids = set(pd.read_parquet(path)["doc_id"].astype("int64"))
        logger.info("protected set: %d documents (from snapshot %s)", len(ids), path)
        return ids

    client = client or bigquery_client()
    frame = client.query(PROTECTED_QUERY).to_dataframe()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    logger.info("protected set: %d documents (snapshotted to %s)", len(frame), path)
    return set(frame["doc_id"].astype("int64"))


def load_catalog(
    path: Path,
    client: bigquery.Client | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """All parsed-but-unclassified documents, cached locally so restarts are free."""
    if path.exists() and not refresh:
        frame = pd.read_parquet(path)
        logger.info("catalog: %d documents (cached %s)", len(frame), path)
        return frame

    client = client or bigquery_client()
    frame = client.query(PENDING_QUERY).to_dataframe()
    frame["doc_id"] = frame["doc_id"].astype("int64")
    frame["justice_kind"] = frame["justice_kind"].astype("int64")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    logger.info("catalog: %d documents (queried, cached to %s)", len(frame), path)
    return frame


def select_work(
    catalog: pd.DataFrame,
    protected_ids: set[int],
    completed_ids: set[int],
    shard: int = 0,
    num_shards: int = 1,
    limit: int | None = None,
) -> pd.DataFrame:
    """Apply the protected set, this shard, already-done documents, and any limit."""
    work = catalog
    if protected_ids:
        work = work[~work["doc_id"].isin(protected_ids)]
    if num_shards > 1:
        work = work[work["doc_id"] % num_shards == shard]
    if completed_ids:
        work = work[~work["doc_id"].isin(completed_ids)]
    work = work.sort_values("doc_id", kind="stable").reset_index(drop=True)
    if limit is not None:
        work = work.head(limit)
    return work
