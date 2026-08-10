"""CLI entrypoint: ``python -m legal_data_doc_parts.inference``."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from legal_data_doc_parts.inference.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_RUNS_DIR,
    PipelineConfig,
)
from legal_data_doc_parts.inference.progress import sync_from_manifest
from legal_data_doc_parts.inference.runner import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m legal_data_doc_parts.inference",
        description="Classify court document paragraphs into introductory / reasoning / operative.",
    )
    parser.add_argument("--run-id", default=None, help="resume an existing run by id")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)

    work = parser.add_argument_group("work selection")
    work.add_argument("--shard", type=int, default=0)
    work.add_argument("--num-shards", type=int, default=1)
    work.add_argument("--limit", type=int, default=None,
                      help="process at most N *outstanding* documents; rerunning continues "
                           "with the next N rather than redoing the same ones")
    work.add_argument("--retry-failed", action="store_true", help="re-queue this run's failures")
    work.add_argument("--overwrite", action="store_true",
                      help="reclassify documents that already have classification.parquet")

    speed = parser.add_argument_group("throughput")
    speed.add_argument("--download-workers", type=int, default=32)
    speed.add_argument("--upload-workers", type=int, default=32)
    speed.add_argument("--documents-per-chunk", type=int, default=64)
    speed.add_argument("--batch-size", type=int, default=128)

    bq = parser.add_argument_group("bigquery")
    bq.add_argument("--no-bq-update", action="store_true",
                    help="do not touch is_classified (for shards that should stay read-only)")
    bq.add_argument("--bq-flush-size", type=int, default=10_000,
                    help="documents per UPDATE; each statement scans the whole 800MB table")
    bq.add_argument("--sync-bq", action="store_true",
                    help="reconcile is_classified from this run's manifest, then exit")

    parser.add_argument("--no-progress", action="store_true",
                        help="suppress the tqdm progress bar (for nohup / captured logs)")
    parser.add_argument("--no-sequential-rule", action="store_true",
                        help="write the model's raw argmax instead of enforcing "
                             "introductory -> reasoning -> operative per document")
    parser.add_argument("--dry-run", action="store_true",
                        help="write output locally under the run directory; never touch GCS or BigQuery")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    run_id = args.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ")
    config = PipelineConfig(
        run_id=run_id,
        model_path=args.model_path,
        runs_dir=args.runs_dir,
        shard=args.shard,
        num_shards=args.num_shards,
        limit=args.limit,
        retry_failed=args.retry_failed,
        overwrite=args.overwrite,
        use_sequential_rule=not args.no_sequential_rule,
        show_progress=not args.no_progress,
        download_workers=args.download_workers,
        upload_workers=args.upload_workers,
        documents_per_chunk=args.documents_per_chunk,
        batch_size=args.batch_size,
        update_bigquery=not args.no_bq_update,
        bq_flush_size=args.bq_flush_size,
        dry_run=args.dry_run,
    )

    if args.sync_bq:
        if not config.manifest_path.exists():
            print(f"no manifest at {config.manifest_path}", file=sys.stderr)
            return 1
        marked = sync_from_manifest(
            config.manifest_path, config.pending_bq_path, config.bq_lock_path,
            batch_size=config.bq_flush_size,
        )
        print(f"marked {marked} documents classified")
        return 0

    logging.getLogger(__name__).info("run %s -> %s", run_id, config.run_dir)
    stats = Pipeline(config).run()
    print(stats.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
