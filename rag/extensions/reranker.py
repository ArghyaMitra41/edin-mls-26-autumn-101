"""Cross-encoder reranking stage for the RAG investigation.

The dense retriever (enc_a) scores the query and each record independently.
A cross-encoder reads the query and one record together and outputs a single
relevance score, which is more accurate but costs one transformer forward pass
per (query, record) pair. We therefore only rerank the top-N dense candidates.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np


class CrossEncoderReranker:
    def __init__(self, model_dir: str | Path | None = None, device: str | None = "cuda",
                 max_length: int = 512, batch_size: int = 32, model=None):
        """Load a locally saved sentence-transformers CrossEncoder.

        `model` lets tests pass a stub with a `predict(pairs, ...)` method instead.
        """
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size and max_length must be positive")
        self.batch_size = batch_size
        self.max_length = max_length
        self.name = Path(model_dir).name if model_dir is not None else "stub"
        start = time.perf_counter()
        if model is None:
            if model_dir is None:
                raise ValueError("model_dir is required")
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(str(model_dir), max_length=max_length, device=device,
                                 local_files_only=True)
        self.model = model
        self.load_ms = (time.perf_counter() - start) * 1000.0

    def _scores(self, pairs):
        scores = self.model.predict(pairs, batch_size=self.batch_size,
                                    show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(scores, dtype=np.float32).reshape(-1)

    def warmup(self, n: int = 3):
        """Run a few dummy batches so CUDA start-up is not charged to the first query."""
        pairs = [("warm-up query", "warm-up document text")] * self.batch_size
        for _ in range(n):
            self._scores(pairs)

    def rerank(self, query: str, record_ids: list[str], data, top_k: int):
        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        if len(record_ids) < top_k:
            raise ValueError("need at least top_k candidates to rerank")
        pairs = [(query, data.record_text(rid)) for rid in record_ids]

        start = time.perf_counter()
        scores = self._scores(pairs)  # returns numpy, so GPU work has finished here
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        order = np.argsort(-scores, kind="stable")  # ties keep the dense order
        return {
            "record_ids": [record_ids[int(i)] for i in order[:top_k]],
            "scores": [float(scores[int(i)]) for i in order[:top_k]],
            "full_order": [record_ids[int(i)] for i in order],
            "latency_ms": elapsed_ms,
            "n_candidates": len(record_ids),
        }

    def truncation_stats(self, query: str, record_ids: list[str], data):
        """Fraction of (query, record) pairs longer than max_length tokens."""
        tok = getattr(self.model, "tokenizer", None)
        if tok is None:
            return None
        n_long = 0
        for rid in record_ids:
            n = len(tok(query, data.record_text(rid), truncation=False)["input_ids"])
            n_long += n > self.max_length
        return n_long / len(record_ids) if record_ids else 0.0
