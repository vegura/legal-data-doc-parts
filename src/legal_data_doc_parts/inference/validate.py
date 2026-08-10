"""Golden accuracy check against the Lapa-LLM ground truth.

Runs the real inference path -- context construction, the classifier, the sequential rule --
over documents whose labels we already have, and reports agreement. This is the check that
catches a broken invariant: if context construction or the tokenizer pairing regressed,
agreement collapses well below the model's 0.9068 validation macro-F1.

Reads the local ``legal_paragraphs.parquet``, so it costs nothing and touches no cloud
resource.

    python -m legal_data_doc_parts.inference.validate --documents 200
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from legal_data_doc_parts.inference.classifier import SectionClassifier
from legal_data_doc_parts.inference.config import DEFAULT_MODEL_PATH, PACKAGE_DIR, SECTION_ORDER
from legal_data_doc_parts.inference.context import prepare_document
from legal_data_doc_parts.inference.sequential import count_corrections, enforce_sequential

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = PACKAGE_DIR / "legal_paragraphs.parquet"
SECTION_TO_ID = {section: index for index, section in enumerate(SECTION_ORDER)}


def is_monotonic(labels: np.ndarray) -> bool:
    return bool(np.all(np.diff(labels) >= 0))


def validate(
    n_documents: int = 200,
    model_path: Path = DEFAULT_MODEL_PATH,
    ground_truth_path: Path = GROUND_TRUTH_PATH,
    batch_size: int = 128,
    seed: int = 0,
) -> pd.DataFrame:
    frame = pd.read_parquet(ground_truth_path)
    frame["document_id"] = frame["document_id"].astype(str)

    rng = np.random.default_rng(seed)
    document_ids = frame["document_id"].drop_duplicates()
    chosen = set(rng.choice(document_ids, size=min(n_documents, len(document_ids)), replace=False))
    sample = frame[frame["document_id"].isin(chosen)]
    logger.info("validating on %d documents / %d paragraphs", len(chosen), len(sample))

    classifier = SectionClassifier(model_path, batch_size=batch_size)

    rows = []
    for document_id, document in sample.groupby("document_id", sort=True):
        prepared = prepare_document(document)
        raw = classifier.predict(prepared["context"].tolist(), prepared["text"].tolist())
        enforced = enforce_sequential(raw)
        truth = prepared["section"].map(SECTION_TO_ID).to_numpy()

        rows.append(
            {
                "document_id": document_id,
                "n_paragraphs": len(prepared),
                "n_corrected": count_corrections(raw, enforced),
                "raw_correct": int((raw == truth).sum()),
                "enforced_correct": int((enforced == truth).sum()),
                "truth_is_monotonic": is_monotonic(truth),
                "raw": raw,
                "enforced": enforced,
                "truth": truth,
            }
        )

    return pd.DataFrame(rows)


def report(results: pd.DataFrame) -> str:
    raw = np.concatenate(results["raw"].tolist())
    enforced = np.concatenate(results["enforced"].tolist())
    truth = np.concatenate(results["truth"].tolist())
    names = list(SECTION_ORDER)

    monotonic = results["truth_is_monotonic"]
    lines = [
        f"documents: {len(results)}   paragraphs: {len(truth)}",
        "",
        f"{'':<22}{'accuracy':>10}{'macro-F1':>10}",
        f"{'raw argmax':<22}{accuracy_score(truth, raw):>10.4f}"
        f"{f1_score(truth, raw, average='macro', zero_division=0):>10.4f}",
        f"{'+ sequential rule':<22}{accuracy_score(truth, enforced):>10.4f}"
        f"{f1_score(truth, enforced, average='macro', zero_division=0):>10.4f}",
        "",
        f"sequential rule changed {results['n_corrected'].sum()} paragraphs "
        f"({results['n_corrected'].sum() / len(truth):.2%}) "
        f"in {int((results['n_corrected'] > 0).sum())} documents",
        f"ground truth is itself monotonic in {int(monotonic.sum())}/{len(results)} documents "
        f"({monotonic.mean():.1%})",
    ]

    # Where ground truth is monotonic the rule can only help; where it is not, the rule is
    # structurally unable to reproduce the labels. Splitting the two keeps the claim honest.
    for label, subset in [("truth monotonic", monotonic), ("truth non-monotonic", ~monotonic)]:
        if not subset.any():
            continue
        part = results[subset]
        part_raw = np.concatenate(part["raw"].tolist())
        part_enforced = np.concatenate(part["enforced"].tolist())
        part_truth = np.concatenate(part["truth"].tolist())
        lines.append(
            f"  {label:<22} n={len(part):>4} docs  "
            f"raw {accuracy_score(part_truth, part_raw):.4f} -> "
            f"enforced {accuracy_score(part_truth, part_enforced):.4f}"
        )

    lines += ["", classification_report(truth, enforced, target_names=names, digits=4, zero_division=0)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=int, default=200)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None, help="write per-document results")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s | %(message)s")
    results = validate(
        n_documents=args.documents,
        model_path=args.model_path,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(report(results))
    if args.output:
        results.drop(columns=["raw", "enforced", "truth"]).to_csv(args.output, index=False)
        print(f"\nper-document results -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
