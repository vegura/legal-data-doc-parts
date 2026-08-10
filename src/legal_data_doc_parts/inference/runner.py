"""Orchestration: download concurrently, classify in GPU-sized chunks, upload concurrently."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from legal_data_doc_parts.inference import catalog as catalog_module
from legal_data_doc_parts.inference.classifier import SectionClassifier
from legal_data_doc_parts.inference.config import PipelineConfig
from legal_data_doc_parts.inference.context import prepare_document
from legal_data_doc_parts.inference.progress import BigQueryProgress, RunLog
from legal_data_doc_parts.inference.sequential import count_corrections, enforce_sequential
from legal_data_doc_parts.inference.storage import OUTPUT_COLUMNS, DocumentStore

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    documents: int = 0
    paragraphs: int = 0
    corrections: int = 0
    documents_with_corrections: int = 0
    skipped: int = 0
    failed: int = 0
    sequential_rule: bool = True
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    def summary(self) -> str:
        rate = self.paragraphs / self.elapsed if self.elapsed else 0.0
        share = self.corrections / self.paragraphs if self.paragraphs else 0.0
        # With the rule off the counters are zero by construction; say so rather than
        # reporting a zero that reads like agreement.
        rule = (
            f"sequential rule changed {self.corrections} paragraphs ({share:.2%}) "
            f"across {self.documents_with_corrections} documents"
            if self.sequential_rule
            else "sequential rule off (raw argmax)"
        )
        return (
            f"{self.documents} documents / {self.paragraphs} paragraphs in "
            f"{self.elapsed:.0f}s ({rate:.0f} paragraphs/s); {rule}; "
            f"{self.skipped} skipped, {self.failed} failed"
        )


@dataclass
class _Document:
    doc_id: int
    justice_kind: int
    paragraphs: pd.DataFrame


class Pipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        # Both pools hit the same session, so the connection pool must cover them together.
        self.store = DocumentStore(
            dry_run_dir=config.dry_run_dir,
            pool_size=config.download_workers + config.upload_workers,
        )
        self.classifier = SectionClassifier(config.model_path, batch_size=config.batch_size)
        self.run_log = RunLog(config.manifest_path, config.failures_path)
        self.bigquery = BigQueryProgress(
            config.pending_bq_path,
            config.bq_lock_path,
            flush_size=config.bq_flush_size,
            enabled=config.update_bigquery,
        )
        self.stats = RunStats(sequential_rule=config.use_sequential_rule)
        self._stats_lock = threading.Lock()
        self._protected: set[int] = set()
        # Its own lock: _publish advances the bar while holding nothing else, and a
        # non-reentrant _stats_lock would deadlock if the two were shared.
        self._progress: tqdm | None = None
        self._progress_lock = threading.Lock()

    # ------------------------------------------------------------------ work selection

    def select_work(self) -> pd.DataFrame:
        config = self.config
        protected = catalog_module.load_protected_ids(config.protected_ids_path)
        catalog = catalog_module.load_catalog(config.catalog_path)

        if config.retry_failed:
            wanted = self.run_log.failed_ids() - self.run_log.completed_ids()
            catalog = catalog[catalog["doc_id"].isin(wanted)]
            completed: set[int] = set()
        else:
            completed = self.run_log.completed_ids()

        work = catalog_module.select_work(
            catalog,
            protected_ids=protected,
            completed_ids=completed,
            shard=config.shard,
            num_shards=config.num_shards,
            limit=config.limit,
        )
        # Belt and braces: the protected set is filtered above, but a write is unrecoverable.
        self._protected = protected
        logger.info("selected %d documents for shard %d/%d",
                    len(work), config.shard, config.num_shards)
        return work

    # ------------------------------------------------------------------ main loop

    def run(self) -> RunStats:
        work = self.select_work()
        if work.empty:
            logger.info("nothing to do")
            return self.stats

        # Start the clock at the work, not at the catalog query, or the first reported
        # throughput is meaningless.
        self.stats.started_at = time.perf_counter()

        with (
            tqdm(
                total=len(work),
                unit="doc",
                desc=f"shard {self.config.shard}/{self.config.num_shards}",
                disable=not self.config.show_progress,
                smoothing=0.05,  # multi-day run: a stable average beats a twitchy ETA
            ) as progress,
            # Send log records through tqdm.write, or every INFO line smears the bar.
            logging_redirect_tqdm(),
            ThreadPoolExecutor(self.config.download_workers, thread_name_prefix="download") as readers,
            ThreadPoolExecutor(self.config.upload_workers, thread_name_prefix="upload") as writers,
        ):
            self._progress = progress
            upload_futures = []
            for chunk in self._chunks(work):
                documents = self._download_chunk(chunk, readers)
                if not documents:
                    continue
                for doc_id, justice_kind, frame, corrections in self._classify(documents):
                    upload_futures.append(
                        writers.submit(self._publish, doc_id, justice_kind, frame, corrections)
                    )
                # Keep the upload queue bounded so a slow network cannot balloon memory.
                # Progress is only reported once drained, otherwise the counters lag the
                # in-flight uploads and understate throughput.
                if len(upload_futures) >= self.config.upload_workers * 4:
                    self._drain(upload_futures)
                    upload_futures = []
                    logger.info("progress: %s", self.stats.summary())
            self._drain(upload_futures)
            self._progress = None

        self.bigquery.flush(force=True)
        logger.info("done: %s", self.stats.summary())
        return self.stats

    def _advance(self) -> None:
        """One document reached a terminal state: published, skipped or failed.

        Called exactly once per selected document, so the bar ends full even on a run where
        everything failed.
        """
        with self._progress_lock:
            if self._progress is None:
                return
            self._progress.set_postfix(
                paragraphs=self.stats.paragraphs,
                skipped=self.stats.skipped,
                failed=self.stats.failed,
                refresh=False,
            )
            self._progress.update(1)

    def _chunks(self, work: pd.DataFrame) -> Iterator[pd.DataFrame]:
        size = self.config.documents_per_chunk
        for start in range(0, len(work), size):
            yield work.iloc[start : start + size]

    def _download_chunk(self, chunk: pd.DataFrame, readers: ThreadPoolExecutor) -> list[_Document]:
        futures = {
            readers.submit(self._download, int(row.doc_id), int(row.justice_kind)): int(row.doc_id)
            for row in chunk.itertuples(index=False)
        }
        documents = []
        for future in as_completed(futures):
            doc_id = futures[future]
            try:
                document = future.result()
            except Exception as error:  # one bad document must not stop a multi-day run
                self.stats.failed += 1
                self.run_log.record_failure(doc_id, f"{type(error).__name__}: {error}")
                logger.warning("download failed for %s: %s", doc_id, error)
                self._advance()
                continue
            if document is None:
                self.stats.skipped += 1
                self._advance()
            else:
                documents.append(document)
        return documents

    def _download(self, doc_id: int, justice_kind: int) -> _Document | None:
        document_id = str(doc_id)
        if not self.config.overwrite and self.store.output_exists(justice_kind, document_id):
            return None
        paragraphs = self.store.read_paragraphs(justice_kind, document_id)
        if paragraphs.empty:
            raise ValueError("paragraphs.parquet is empty")
        return _Document(doc_id, justice_kind, paragraphs)

    def _classify(self, documents: list[_Document]):
        """Classify a whole chunk in shared batches, then split results back per document."""
        prepared = [(doc, prepare_document(doc.paragraphs)) for doc in documents]
        contexts, texts = [], []
        for _, frame in prepared:
            contexts.extend(frame["context"].tolist())
            texts.extend(frame["text"].tolist())

        predictions = self.classifier.predict(contexts, texts)

        offset = 0
        for document, frame in prepared:
            size = len(frame)
            raw = predictions[offset : offset + size]
            offset += size

            # The rule is per document and must never see a batch boundary.
            if self.config.use_sequential_rule:
                enforced = enforce_sequential(raw)
                corrections = count_corrections(raw, enforced)
            else:
                enforced = raw
                corrections = 0

            output = frame[
                ["document_id", "paragraph_index", "paragraph_order", "numbered_text", "text"]
            ].copy()
            output["section"] = self.classifier.section_names(enforced)
            yield document.doc_id, document.justice_kind, output[OUTPUT_COLUMNS], corrections

    def _publish(self, doc_id: int, justice_kind: int, frame: pd.DataFrame, corrections: int) -> None:
        if doc_id in self._protected:
            raise RuntimeError(f"refusing to overwrite protected document {doc_id}")

        started = time.perf_counter()
        self.store.write_classification(justice_kind, str(doc_id), frame)
        # Upload first, then record, then BigQuery -- so the flag never outruns the file.
        self.run_log.record_success(
            doc_id,
            n_paragraphs=len(frame),
            n_corrected=corrections,
            seconds=round(time.perf_counter() - started, 3),
        )
        self.bigquery.add(doc_id)

        with self._stats_lock:
            self.stats.documents += 1
            self.stats.paragraphs += len(frame)
            self.stats.corrections += corrections
            if corrections:
                self.stats.documents_with_corrections += 1
        self._advance()

    def _drain(self, futures: list) -> None:
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                # _publish raised before its own tick, so the bar is advanced here instead.
                self.stats.failed += 1
                logger.warning("publish failed: %s", error)
                self._advance()


def summarise_corrections(manifest_path) -> pd.DataFrame:
    """Per-document correction counts from a manifest, for spotting model trouble."""
    records = pd.read_json(manifest_path, lines=True)
    if records.empty:
        return records
    records["correction_rate"] = records["n_corrected"] / records["n_paragraphs"]
    return records.sort_values("correction_rate", ascending=False)
