"""Reconstruction of the two-paragraph context the classifier was trained on.

The model is a sentence-pair classifier: the first segment is the two preceding paragraphs,
the second is the paragraph being labelled. This must reproduce
``1.2_embeddings_encodings.ipynb`` exactly, or every prediction degrades silently::

    previous_1 = by_document.shift(1).fillna("")
    previous_2 = by_document.shift(2).fillna("")
    context = (previous_2 + "\\n\\n" + previous_1).str.strip()
"""

from collections.abc import Sequence

import pandas as pd

CONTEXT_SEPARATOR = "\n\n"


def build_context(texts: Sequence[str]) -> list[str]:
    """Context strings for paragraphs already in ascending ``paragraph_index`` order."""
    contexts = []
    for position in range(len(texts)):
        previous_2 = texts[position - 2] if position >= 2 else ""
        previous_1 = texts[position - 1] if position >= 1 else ""
        contexts.append((previous_2 + CONTEXT_SEPARATOR + previous_1).strip())
    return contexts


def prepare_document(paragraphs: pd.DataFrame) -> pd.DataFrame:
    """Sort a document's paragraphs and attach their context.

    Sorting first is not optional: context is positional, so an out-of-order frame produces
    a plausible-looking but wrong pairing.
    """
    ordered = paragraphs.sort_values("paragraph_index", kind="stable").reset_index(drop=True)
    ordered["context"] = build_context(ordered["text"].tolist())
    return ordered
