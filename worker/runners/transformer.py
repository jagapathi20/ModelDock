"""Runner for Hugging Face sequence-classification models (the main workload).

One batch goes through three phases, kept as separate methods so the
benchmark can time each one:

    preprocess   texts -> padded token tensors      (tokenizer, CPU, mostly Rust)
    forward      tensors -> logits                  (PyTorch, CPU, multithreaded)
    postprocess  logits -> [{"label", "score"}]     (softmax + argmax, cheap)

``forward`` is the part batching makes cheaper per item: one matrix multiply
over 16 rows uses the CPU's vector units and caches far better than 16 separate
multiplies over 1 row each.
"""

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from worker.runners.base import JsonDict, ModelRunner

# DistilBERT accepts up to 512 tokens. We cap lower on purpose: a batch is
# padded to its *longest* input, so one 512-token input would make every other
# request in its batch pay for 512 tokens. Capping bounds the worst-case batch
# cost. Inputs longer than this are truncated (the start of the text is kept).
DEFAULT_MAX_LENGTH = 128


def configure_cpu_threads(num_threads: int) -> None:
    """Set how many CPU threads this *process* may use for inference.

    D7: processes x threads must not exceed the number of cores, or threads
    fight over cores and everything gets slower. This is process-wide, which
    is why it's a separate function rather than a runner argument: two runners
    in one process can't have different thread counts.

    Call it once, early, before the first inference.
    """
    if num_threads < 1:
        raise ValueError("num_threads must be >= 1")
    # Fast tokenizers have their own Rust thread pool for batch encoding. Left
    # on, it adds threads on top of PyTorch's and breaks the D7 budget. For
    # batches of short texts the speedup is negligible anyway.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Intra-op threads: how many threads one operator (e.g. one matmul) uses.
    torch.set_num_threads(num_threads)


class TransformerClassifierRunner(ModelRunner):
    """Text classification with any ``AutoModelForSequenceClassification``.

    Input:  ``{"text": "the movie was great"}``
    Output: ``{"label": "POSITIVE", "score": 0.9998}``, where ``score`` is
            the softmax probability of the predicted label.
    """

    kind = "transformer_classifier"

    def __init__(self, max_length: int = DEFAULT_MAX_LENGTH) -> None:
        super().__init__()
        self.max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None
        self._id2label: dict[int, str] = {}

    def load(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        # local_files_only=True: fail fast if the directory is incomplete
        # instead of silently reaching out to the Hugging Face Hub.
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True
        )
        # eval() switches off training-only behavior, most importantly dropout,
        # which randomly zeroes activations. Without it, the same input could
        # give different scores on every call, and the response cache (D10)
        # would be wrong.
        self._model.eval()
        self._id2label = {int(k): v for k, v in self._model.config.id2label.items()}
        self._loaded = True

    # --- the three phases -------------------------------------------------

    def preprocess(self, inputs: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        texts = [item["text"] for item in inputs]
        return self._tokenizer(
            texts,
            padding=True,  # pad to the longest text in *this* batch
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(self, encoded: Mapping[str, torch.Tensor]) -> torch.Tensor:
        # inference_mode() turns off autograd bookkeeping (no gradient graph is
        # recorded), which saves memory and time. It's a stricter, faster
        # version of no_grad().
        with torch.inference_mode():
            return self._model(**encoded).logits

    def postprocess(self, logits: torch.Tensor) -> list[JsonDict]:
        probs = torch.softmax(logits, dim=-1)
        scores, label_ids = probs.max(dim=-1)
        return [
            {"label": self._id2label[int(i)], "score": float(s)}
            for i, s in zip(label_ids.tolist(), scores.tolist(), strict=True)
        ]

    # --- public API -------------------------------------------------------

    def predict_batch(self, inputs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
        self._require_loaded()
        if not inputs:
            return []
        return self.postprocess(self.forward(self.preprocess(inputs)))
