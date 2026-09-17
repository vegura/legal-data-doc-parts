"""Static coordinates of the corpus and the tunable knobs of a pipeline run."""

from dataclasses import dataclass
from pathlib import Path

# The package directory, so the CLI works regardless of the caller's cwd.
PACKAGE_DIR = Path(__file__).resolve().parent.parent

PROJECT_ID = "lab-test-project-1-305710"
DOCUMENT_TABLE = f"{PROJECT_ID}.court_data_selected.document_data"

BUCKET_NAME = "court_data_selected_structured"
BLOB_TEMPLATE = "document_text_parsing/info_version_9/{justice_kind}/{document_id}/{name}"
INPUT_BLOB_NAME = "paragraphs.parquet"
OUTPUT_BLOB_NAME = "classification.parquet"

# Best checkpoint of the 5-epoch run (validation macro-F1 0.9068). Neither training run
# exported a final/ directory, so the checkpoint is the artefact.
DEFAULT_MODEL_PATH = PACKAGE_DIR / "model" / "checkpoint-43407"
DEFAULT_RUNS_DIR = PACKAGE_DIR / "runs_selected"

# Sections in document order. The running-max rule in sequential.py depends on the model's
# label ids agreeing with this ordering; classifier.py asserts it at load time.
SECTION_ORDER = ("introductory", "reasoning", "operative")

MAX_SEQUENCE_LENGTH = 512


def blob_path(justice_kind: int | str, document_id: str, name: str) -> str:
    return BLOB_TEMPLATE.format(justice_kind=justice_kind, document_id=document_id, name=name)


@dataclass
class PipelineConfig:
    """One run's configuration. Defaults are tuned for a single GPU box."""

    run_id: str
    model_path: Path = DEFAULT_MODEL_PATH
    runs_dir: Path = DEFAULT_RUNS_DIR

    # Work selection
    shard: int = 0
    num_shards: int = 1
    limit: int | None = None
    retry_failed: bool = False
    overwrite: bool = False

    # Post-processing. Off writes the model's raw argmax, which is what you want when
    # measuring the rule's own effect or when the corpus is not judgment-shaped.
    use_sequential_rule: bool = True

    # Throughput
    download_workers: int = 32
    upload_workers: int = 32
    documents_per_chunk: int = 64
    batch_size: int = 128

    # BigQuery write-back
    update_bigquery: bool = True
    bq_flush_size: int = 10_000

    # Reporting
    show_progress: bool = True

    # Safety
    dry_run: bool = False
    dry_run_dir: Path | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.shard < self.num_shards:
            raise ValueError(f"shard {self.shard} out of range for num_shards {self.num_shards}")
        self.model_path = Path(self.model_path)
        self.runs_dir = Path(self.runs_dir)
        if self.dry_run and self.dry_run_dir is None:
            self.dry_run_dir = self.run_dir / "output"
        # A dry run must never touch BigQuery.
        if self.dry_run:
            self.update_bigquery = False

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_id

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.jsonl"

    @property
    def failures_path(self) -> Path:
        return self.run_dir / "failures.jsonl"

    @property
    def pending_bq_path(self) -> Path:
        return self.run_dir / "pending_bq.jsonl"

    @property
    def catalog_path(self) -> Path:
        return self.runs_dir / "catalog.parquet"

    @property
    def protected_ids_path(self) -> Path:
        """Doc ids of the Lapa-LLM labelled documents, snapshotted before any BQ update."""
        return self.runs_dir / "protected_doc_ids.parquet"

    @property
    def bq_lock_path(self) -> Path:
        return self.runs_dir / "bq_update.lock"
