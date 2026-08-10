"""Model loading and batched paragraph classification."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from legal_data_doc_parts.inference.config import MAX_SEQUENCE_LENGTH, SECTION_ORDER

logger = logging.getLogger(__name__)


class SectionClassifier:
    """Wraps the fine-tuned encoder and reproduces the training-time encoding."""

    def __init__(self, model_path: Path, batch_size: int = 128, device: str | None = None):
        self.batch_size = batch_size
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        # fp16 is a large win on GPU and unsupported on CPU.
        if self.device.type == "cuda":
            model = model.half()
        self.model = model.to(self.device).eval()

        self.id_to_section = {int(i): label for i, label in model.config.id2label.items()}
        self._validate_label_order()
        logger.info(
            "loaded %s on %s (%s labels: %s)",
            model_path, self.device, len(self.id_to_section), self.id_to_section,
        )

    def _validate_label_order(self) -> None:
        """The running-max rule is only correct if ids ascend in document order."""
        actual = tuple(self.id_to_section[i] for i in sorted(self.id_to_section))
        if actual != SECTION_ORDER:
            raise ValueError(
                f"model label order {actual} does not match the sequential order "
                f"{SECTION_ORDER}; the running-max rule would be wrong"
            )

    def predict(self, contexts: Sequence[str], texts: Sequence[str]) -> np.ndarray:
        """Predicted label ids, aligned with the inputs.

        Pairs are bucketed by token length so a batch is not padded out to the longest
        sequence in the whole chunk, then restored to the caller's order.
        """
        if len(contexts) != len(texts):
            raise ValueError(f"length mismatch: {len(contexts)} contexts, {len(texts)} texts")
        if not texts:
            return np.empty(0, dtype=np.int64)

        encoded = self.tokenizer(
            list(contexts),
            list(texts),
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            padding=False,
        )
        # token_type_ids carry the context/paragraph segment boundary this model was trained
        # with, so every field has to survive batching -- not just input_ids.
        fields = list(encoded.keys())

        order = np.argsort([len(ids) for ids in encoded["input_ids"]], kind="stable")
        predictions = np.empty(len(texts), dtype=np.int64)

        with torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                positions = order[start : start + self.batch_size]
                batch = self.tokenizer.pad(
                    {field: [encoded[field][i] for i in positions] for field in fields},
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**batch).logits
                predictions[positions] = logits.argmax(dim=-1).cpu().numpy()

        return predictions

    def section_names(self, label_ids: np.ndarray) -> list[str]:
        return [self.id_to_section[int(i)] for i in label_ids]
