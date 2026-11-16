"""Reranking stage tests with stub models (no downloads, no GPU, no ELM)."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions"))
from reranker import CrossEncoderReranker
from src.rag import RAGPipeline
from src.context_builder import SimpleContextBuilder


class Data:
    def __init__(self, texts):
        self.texts = texts

    def record_text(self, rid):
        return self.texts[rid]


class StubCrossEncoder:
    """Score = number written in the record text."""

    def predict(self, pairs, **kwargs):
        return np.array([float(doc) for _, doc in pairs], dtype=np.float32)


class StubEncoder:
    def encode(self, text):
        return np.ones(2, dtype=np.float32)


class StubRetriever:
    """Dense order r0, r1, r2, ... (dense scores ignore the reranker's opinion)."""

    def __init__(self, ids):
        self.ids = ids

    def retrieve(self, q, k=5):
        return {"record_ids": self.ids[:k], "scores": [0.0] * k, "latency_ms": 0.1}


class StubGenerator:
    def generate(self, prompt):
        return {"answer": "ok", "latency_ms": 1.0}


IDS = [f"r{i}" for i in range(8)]
DATA = Data({f"r{i}": str(s) for i, s in enumerate([1, 5, 2, 8, 3, 7, 0, 6])})


def pipeline(**kw):
    return RAGPipeline(None, StubEncoder(), StubRetriever(IDS), DATA, SimpleContextBuilder(),
                       StubGenerator(), **kw)


class RerankTests(unittest.TestCase):
    def test_rerank_orders_by_score_and_keeps_k(self):
        rr = CrossEncoderReranker(model=StubCrossEncoder())
        out = rr.rerank("q", IDS[:6], DATA, top_k=3)
        self.assertEqual(out["record_ids"], ["r3", "r5", "r1"])
        self.assertEqual(out["full_order"], ["r3", "r5", "r1", "r4", "r2", "r0"])
        self.assertEqual(out["n_candidates"], 6)

    def test_ties_keep_dense_order(self):
        rr = CrossEncoderReranker(model=StubCrossEncoder())
        data = Data({"a": "1", "b": "1", "c": "1"})
        self.assertEqual(rr.rerank("q", ["a", "b", "c"], data, top_k=3)["record_ids"], ["a", "b", "c"])

    def test_too_few_candidates_rejected(self):
        rr = CrossEncoderReranker(model=StubCrossEncoder())
        with self.assertRaises(ValueError):
            rr.rerank("q", IDS[:2], DATA, top_k=5)

    def test_baseline_path_unchanged(self):
        out = pipeline().answer_transcript("q", k=5)
        self.assertEqual(out["record_ids"], IDS[:5])
        self.assertIsNone(out["rerank"])
        self.assertNotIn("rerank_ms", out["timing"])

    def test_pipeline_with_reranker(self):
        rr = CrossEncoderReranker(model=StubCrossEncoder())
        out = pipeline(reranker=rr, candidate_depth=8).answer_transcript("q", k=5)
        self.assertEqual(out["record_ids"], ["r3", "r5", "r7", "r1", "r4"])
        self.assertEqual(len(out["retrieval"]["record_ids"]), 8)
        self.assertIn("rerank_ms", out["timing"])
        self.assertEqual(out["contexts"], ["8", "7", "6", "5", "3"])

    def test_depth_below_k_rejected(self):
        rr = CrossEncoderReranker(model=StubCrossEncoder())
        with self.assertRaises(ValueError):
            pipeline(reranker=rr, candidate_depth=3).answer_transcript("q", k=5)
        with self.assertRaises(ValueError):
            pipeline(reranker=rr)

    def test_row_schema_passes_judge_validation(self):
        from scripts.evaluate_answers import validate_candidates
        out = pipeline().answer_transcript("a question", k=5)
        row = {"query_id": "q1", "split": "dev", "transcript": out["transcript"],
               "answer": out["answer"], "retrieved_record_ids": out["record_ids"]}
        self.assertEqual(validate_candidates([row], {"q1": {}}, "dev"), {"q1"})


if __name__ == "__main__":
    unittest.main()
