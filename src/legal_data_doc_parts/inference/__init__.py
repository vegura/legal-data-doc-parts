"""Batch section classification of court documents held in Cloud Storage.

Reads each document's ``paragraphs.parquet``, classifies every paragraph into
introductory / reasoning / operative, writes ``classification.parquet`` back beside the
input, and marks the document ``is_classified`` in BigQuery.
"""

from legal_data_doc_parts.inference.config import PipelineConfig
from legal_data_doc_parts.inference.sequential import enforce_sequential

__all__ = ["PipelineConfig", "enforce_sequential"]
