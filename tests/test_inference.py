"""Tests for the batch inference pipeline.

Runs standalone (``python tests/test_inference.py``) since the project has no pytest, but is
written so pytest would collect it unchanged.
"""

from __future__ import annotations

import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(PACKAGE_ROOT))

from legal_data_doc_parts.inference.__main__ import build_parser  # noqa: E402
from legal_data_doc_parts.inference.catalog import select_work  # noqa: E402
from legal_data_doc_parts.inference.config import PipelineConfig  # noqa: E402
from legal_data_doc_parts.inference.context import build_context, prepare_document  # noqa: E402
from legal_data_doc_parts.inference.progress import BigQueryProgress, RunLog  # noqa: E402
from legal_data_doc_parts.inference import runner  # noqa: E402
from legal_data_doc_parts.inference.runner import RunStats  # noqa: E402
from legal_data_doc_parts.inference.sequential import (  # noqa: E402
    count_corrections,
    enforce_sequential,
)
from legal_data_doc_parts.inference.storage import OUTPUT_COLUMNS  # noqa: E402

DATA_DIR = PACKAGE_ROOT / "legal_data_doc_parts"


# ----------------------------------------------------------------- sequential rule


def test_monotonic_input_is_unchanged():
    labels = np.array([0, 0, 1, 1, 1, 2, 2])
    assert enforce_sequential(labels).tolist() == labels.tolist()


def test_regression_is_pulled_forward():
    # introductory after reasoning becomes reasoning; reasoning after operative becomes operative
    assert enforce_sequential(np.array([0, 1, 0, 2, 1])).tolist() == [0, 1, 1, 2, 2]


def test_leading_operative_pins_the_tail():
    """The known failure mode of the rule, pinned so it cannot change silently."""
    assert enforce_sequential(np.array([2, 0, 0, 1])).tolist() == [2, 2, 2, 2]


def test_skipped_section_is_allowed():
    assert enforce_sequential(np.array([0, 2, 2])).tolist() == [0, 2, 2]


def test_degenerate_inputs():
    assert enforce_sequential(np.array([1])).tolist() == [1]
    assert enforce_sequential(np.array([], dtype=np.int64)).tolist() == []


def test_correction_count():
    raw = np.array([0, 1, 0, 2, 1])
    assert count_corrections(raw, enforce_sequential(raw)) == 2
    monotonic = np.array([0, 1, 2])
    assert count_corrections(monotonic, enforce_sequential(monotonic)) == 0


def test_sequential_rule_is_on_by_default_and_can_be_turned_off():
    assert PipelineConfig(run_id="t").use_sequential_rule is True
    assert PipelineConfig(run_id="t", use_sequential_rule=False).use_sequential_rule is False
    assert build_parser().parse_args([]).no_sequential_rule is False
    assert build_parser().parse_args(["--no-sequential-rule"]).no_sequential_rule is True


def test_summary_does_not_report_zero_corrections_when_the_rule_is_off():
    """0 corrections with the rule off means "not applied", not "model agreed"."""
    assert "sequential rule off" in RunStats(documents=1, paragraphs=10,
                                             sequential_rule=False).summary()
    assert "changed 2 paragraphs" in RunStats(documents=1, paragraphs=10,
                                              corrections=2).summary()


# ----------------------------------------------------------------- context


def test_context_of_first_paragraphs():
    contexts = build_context(["a", "b", "c", "d"])
    assert contexts[0] == ""            # nothing precedes
    assert contexts[1] == "a"           # one predecessor, separator stripped
    assert contexts[2] == "a\n\nb"
    assert contexts[3] == "b\n\nc"


def test_context_matches_training_dataset():
    """The invariant most likely to rot: must reproduce 1.2_embeddings_encodings.ipynb."""
    dataset = DATA_DIR / "legal_paragraphs_dataset_v2.parquet"
    if not dataset.exists():
        print("  (skipped: dataset not present)")
        return

    frame = pd.read_parquet(dataset, columns=["document_id", "paragraph_index", "text", "context"])
    for document_id in frame["document_id"].drop_duplicates().head(25):
        document = frame[frame["document_id"] == document_id]
        expected = document.sort_values("paragraph_index", kind="stable")["context"].tolist()
        actual = build_context(
            document.sort_values("paragraph_index", kind="stable")["text"].tolist()
        )
        assert actual == expected, f"context mismatch for document {document_id}"


def test_prepare_document_sorts_before_building_context():
    shuffled = pd.DataFrame(
        {
            "document_id": ["1"] * 3,
            "paragraph_index": [3, 1, 2],
            "text": ["third", "first", "second"],
        }
    )
    prepared = prepare_document(shuffled)
    assert prepared["paragraph_index"].tolist() == [1, 2, 3]
    assert prepared["context"].tolist() == ["", "first", "first\n\nsecond"]


# ----------------------------------------------------------------- work selection


def test_select_work_applies_protection_shard_and_completion():
    catalog = pd.DataFrame({"doc_id": [1, 2, 3, 4, 5, 6], "justice_kind": [2] * 6})

    assert select_work(catalog, {1, 2}, set())["doc_id"].tolist() == [3, 4, 5, 6]
    assert select_work(catalog, set(), {3})["doc_id"].tolist() == [1, 2, 4, 5, 6]
    # shard k keeps doc_id % num_shards == k
    assert select_work(catalog, set(), set(), shard=0, num_shards=2)["doc_id"].tolist() == [2, 4, 6]
    assert select_work(catalog, set(), set(), shard=1, num_shards=2)["doc_id"].tolist() == [1, 3, 5]
    assert select_work(catalog, set(), set(), limit=2)["doc_id"].tolist() == [1, 2]


def test_shards_partition_the_catalog_exactly():
    catalog = pd.DataFrame({"doc_id": list(range(100)), "justice_kind": [2] * 100})
    seen: list[int] = []
    for shard in range(7):
        seen += select_work(catalog, set(), set(), shard=shard, num_shards=7)["doc_id"].tolist()
    assert sorted(seen) == list(range(100)), "shards must partition without overlap or gaps"


# ----------------------------------------------------------------- output contract


def test_output_columns_match_existing_classification_files():
    assert OUTPUT_COLUMNS == [
        "document_id", "paragraph_index", "paragraph_order", "numbered_text", "text", "section",
    ]


# ----------------------------------------------------------------- run state


def test_run_log_roundtrip():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        log = RunLog(root / "manifest.jsonl", root / "failures.jsonl")
        assert log.completed_ids() == set()

        log.record_success(11, n_paragraphs=3, n_corrected=0)
        log.record_success(22, n_paragraphs=5, n_corrected=1)
        log.record_failure(33, "boom")

        assert log.completed_ids() == {11, 22}
        assert log.failed_ids() == {33}


def test_run_log_survives_a_torn_line():
    """A run killed mid-write leaves a partial line; it must not poison resumption."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "manifest.jsonl"
        manifest.write_text('{"doc_id": 1}\n{"doc_id": 2}\n{"doc_i')
        log = RunLog(manifest, root / "failures.jsonl")
        assert log.completed_ids() == {1, 2}


def test_bigquery_progress_persists_before_flushing():
    """add() must durably queue the id, so a crash cannot lose the intent to mark it."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        progress = BigQueryProgress(
            root / "pending_bq.jsonl", root / "bq.lock", flush_size=1_000_000
        )
        for doc_id in (5, 6, 7):
            progress.add(doc_id)

        assert progress.pending_path.read_text().split() == ["5", "6", "7"]
        # Below the flush threshold and not forced: must not attempt any BigQuery call.
        assert progress.flush() == 0

        # A fresh instance recovers the queue from disk.
        recovered = BigQueryProgress(
            root / "pending_bq.jsonl", root / "bq.lock", flush_size=1_000_000
        )
        assert recovered._pending == [5, 6, 7]


def test_bigquery_progress_disabled_is_inert():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        progress = BigQueryProgress(
            root / "pending_bq.jsonl", root / "bq.lock", enabled=False
        )
        progress.add(1)
        assert progress.flush(force=True) == 0
        assert not progress.pending_path.exists()


# ----------------------------------------------------------------- progress bar


class _FakeBar:
    """Counts what the real tqdm would have drawn."""

    def __init__(self, total=None, **_):
        self.total = total
        self.n = 0
        self.postfix = {}

    def update(self, n=1):
        self.n += n

    def set_postfix(self, refresh=True, **fields):
        self.postfix = fields

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_progress_advances_once_per_document_whatever_the_outcome():
    """Published, skipped and failed documents must each tick exactly once, or a long run's
    bar drifts away from reality."""
    work = pd.DataFrame({"doc_id": [1, 2, 3, 4], "justice_kind": [2] * 4})
    paragraphs = pd.DataFrame(
        {
            "document_id": ["1"] * 3,
            "paragraph_index": [1, 2, 3],
            "paragraph_order": [1, 2, 3],
            "numbered_text": ["1. a", "2. b", "3. c"],
            "text": ["a", "b", "c"],
        }
    )

    class Store:
        def output_exists(self, justice_kind, document_id):
            return document_id == "2"          # already done -> skipped

        def read_paragraphs(self, justice_kind, document_id):
            if document_id == "3":
                raise OSError("boom")          # download failure
            return paragraphs

        def write_classification(self, justice_kind, document_id, frame):
            if document_id == "4":
                raise OSError("upload boom")   # publish failure

    class Classifier:
        def predict(self, contexts, texts):
            return np.zeros(len(texts), dtype=np.int64)

        def section_names(self, labels):
            return ["introductory"] * len(labels)

    bars = []

    def fake_tqdm(**kwargs):
        bars.append(_FakeBar(**kwargs))
        return bars[-1]

    original = runner.tqdm
    runner.tqdm = fake_tqdm
    try:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = object.__new__(runner.Pipeline)
            pipeline.config = PipelineConfig(run_id="t", runs_dir=root, dry_run=True)
            pipeline.store = Store()
            pipeline.classifier = Classifier()
            pipeline.run_log = RunLog(root / "manifest.jsonl", root / "failures.jsonl")
            pipeline.bigquery = BigQueryProgress(
                root / "pending.jsonl", root / "bq.lock", enabled=False
            )
            pipeline.stats = RunStats()
            pipeline._stats_lock = threading.Lock()
            pipeline._progress_lock = threading.Lock()
            pipeline._progress = None
            pipeline._protected = set()
            pipeline.select_work = lambda: work

            stats = pipeline.run()
    finally:
        runner.tqdm = original

    bar = bars[0]
    assert bar.total == len(work)
    assert bar.n == len(work), f"bar ended at {bar.n}/{bar.total}"
    assert (stats.documents, stats.skipped, stats.failed) == (1, 1, 2)
    assert bar.postfix == {"paragraphs": 3, "skipped": 1, "failed": 2}


def _run() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as error:
            failures += 1
            print(f"FAIL {name}: {error}")
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"ERROR {name}: {type(error).__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run())
