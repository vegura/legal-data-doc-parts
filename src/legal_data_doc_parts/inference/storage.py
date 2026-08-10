"""Cloud Storage access for paragraph inputs and classification outputs."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests
from google.api_core import exceptions as gcp_exceptions
from google.api_core import retry
from google.cloud import storage

from legal_data_doc_parts.inference.config import (
    BUCKET_NAME,
    INPUT_BLOB_NAME,
    OUTPUT_BLOB_NAME,
    blob_path,
)

logger = logging.getLogger(__name__)

# Columns of an existing classification.parquet, in order. Downstream consumers already read
# this path, so the contract is "the input plus section" and nothing else.
INPUT_COLUMNS = ["document_id", "paragraph_index", "paragraph_order", "numbered_text", "text"]
OUTPUT_COLUMNS = [*INPUT_COLUMNS, "section"]

_RETRY = retry.Retry(
    predicate=retry.if_exception_type(
        gcp_exceptions.ServerError,
        gcp_exceptions.TooManyRequests,
        ConnectionError,
        TimeoutError,
    ),
    initial=1.0,
    maximum=30.0,
    multiplier=2.0,
    timeout=300.0,
)


def widen_connection_pool(client: storage.Client, size: int) -> None:
    """Size the HTTPS connection pool to the number of threads that will use it.

    The default requests adapter keeps 10 connections. With more workers than that, every
    extra request finds the pool full on release, drops its connection ("Connection pool is
    full, discarding connection") and pays a fresh TLS handshake next time. Nothing is lost
    -- the request itself has already succeeded -- but on a corpus of small objects the
    handshakes are pure overhead, and the warnings bury the run's real log.

    Reaches into ``client._http`` because the library exposes no public hook for this; it is
    an ``AuthorizedSession``, i.e. a ``requests.Session``.
    """
    adapter = requests.adapters.HTTPAdapter(pool_connections=size, pool_maxsize=size)
    try:
        client._http.mount("https://", adapter)
    except AttributeError:  # private API moved: slower, still correct
        logger.warning("could not resize the GCS connection pool; expect pool warnings")


class DocumentStore:
    """Reads paragraphs.parquet and writes classification.parquet beside it."""

    def __init__(
        self,
        bucket_name: str = BUCKET_NAME,
        dry_run_dir: Path | None = None,
        pool_size: int | None = None,
    ):
        self.dry_run_dir = Path(dry_run_dir) if dry_run_dir else None
        self._client = storage.Client()
        if pool_size:
            widen_connection_pool(self._client, pool_size)
        self._bucket = self._client.bucket(bucket_name)

    def read_paragraphs(self, justice_kind: int, document_id: str) -> pd.DataFrame:
        path = blob_path(justice_kind, document_id, INPUT_BLOB_NAME)
        payload = self._bucket.blob(path).download_as_bytes(retry=_RETRY)
        frame = pd.read_parquet(io.BytesIO(payload))
        missing = set(INPUT_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}")
        return frame

    def output_exists(self, justice_kind: int, document_id: str) -> bool:
        if self.dry_run_dir is not None:
            return self._local_path(justice_kind, document_id).exists()
        path = blob_path(justice_kind, document_id, OUTPUT_BLOB_NAME)
        return self._bucket.blob(path).exists(retry=_RETRY)

    def write_classification(
        self, justice_kind: int, document_id: str, frame: pd.DataFrame
    ) -> str:
        """Serialise and upload. Called only once a document is fully classified."""
        if list(frame.columns) != OUTPUT_COLUMNS:
            raise ValueError(f"unexpected output columns {list(frame.columns)}")

        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        payload = buffer.getvalue()

        if self.dry_run_dir is not None:
            destination = self._local_path(justice_kind, document_id)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            return str(destination)

        path = blob_path(justice_kind, document_id, OUTPUT_BLOB_NAME)
        self._bucket.blob(path).upload_from_string(
            payload, content_type="application/octet-stream", retry=_RETRY
        )
        return path

    def _local_path(self, justice_kind: int, document_id: str) -> Path:
        return self.dry_run_dir / blob_path(justice_kind, document_id, OUTPUT_BLOB_NAME)
