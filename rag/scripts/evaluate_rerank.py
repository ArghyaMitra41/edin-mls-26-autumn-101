"""Retrieval-quality sweep for cross-encoder reranking (no ELM calls).

For every query: dense-retrieve the top 100 with enc_a (the baseline ranking),
then for each reranker model and candidate depth N, rerank the top N and keep
the dense order for ranks N+1..100. Metrics are computed on that list, so
Recall@100 is identical to the baseline by construction (a sanity check).

Also reports the oracle Recall@5 for each N: the best Recall@5 any reranker of
the top N could reach (relevant records in the top N, capped at 5).

Example:
  python rag/scripts/evaluate_rerank.py --device cpu --split practice \
      --transcript-cache rag/.cache/practice_transcripts.jsonl \
      --models rag/models/rerankers/minilm-l6 rag/models/rerankers/bge-base \
      --depths 10 20 50 100 --out-dir rag/results/rerank_sweep_practice
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions"))

import argparse
import json
import statistics

import numpy as np

from src.asr_adapter import ASRAdapter
from src.data import CourseData
from src.encoder import LocalEncoder
from src.metrics import recall_at_k, mrr_at_k, ndcg_at_k
from src.retriever import ExactDenseRetriever
from scripts.evaluate_rag import load_cache, append_cache
from reranker import CrossEncoderReranker

DENSE_DEPTH = 100
METRICS = ["Recall@5", "Recall@10", "Recall@20", "Recall@100", "MRR@10", "NDCG@10"]


def metrics(ranked, rel):
    return {
        "Recall@5": recall_at_k(ranked, rel, 5),
        "Recall@10": recall_at_k(ranked, rel, 10),
        "Recall@20": recall_at_k(ranked, rel, 20),
        "Recall@100": recall_at_k(ranked, rel, 100),
        "MRR@10": mrr_at_k(ranked, rel, 10),
        "NDCG@10": ndcg_at_k(ranked, rel, 10),
    }


def oracle_recall5(dense, rel, n):
    if not rel:
        return 0.0
    return min(5, len(set(rel).intersection(dense[:n]))) / len(rel)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * len(xs))) - 1))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["practice", "dev"], default="practice")
    ap.add_argument("--transcript-cache", type=Path, required=True)
    ap.add_argument("--models", type=Path, nargs="+", required=True)
    ap.add_argument("--depths", type=int, nargs="+", default=[10, 20, 50, 100])
    ap.add_argument("--device", default=None, help="enc_a device (README uses cpu)")
    ap.add_argument("--reranker-device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="Smoke test on the first N queries")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    if any(n < 5 or n > DENSE_DEPTH for n in args.depths):
        ap.error(f"--depths must be between 5 and {DENSE_DEPTH}")
    if args.out_dir.exists():
        ap.error("Output directory exists; choose a new one to preserve prior results")

    data_dir = ROOT / "data" / "rag_corpus"
    data = CourseData(data_dir)
    items = data.audio_items(args.split)
    if args.limit:
        items = items[:args.limit]
    qrels = data.load_qrels(args.split)
    encoder = LocalEncoder(ROOT / "models" / "enc_a", device=args.device)
    retriever = ExactDenseRetriever(
        data_dir / "embeddings" / "corpus_enc_a.npy",
        data_dir / "embeddings" / "corpus_enc_a_ids.json",
    )

    # 1. Transcripts and dense top-100 (shared by every configuration).
    cache = load_cache(args.transcript_cache)
    asr = None
    queries = []
    for item in items:
        qid = item["query_id"]
        transcript = cache.get(qid)
        if transcript is None:
            asr = asr or ASRAdapter()
            transcript = asr.transcribe(data_dir / item["audio_file"])
            append_cache(args.transcript_cache, qid, transcript)
        dense = retriever.retrieve(encoder.encode(transcript), k=DENSE_DEPTH)["record_ids"]
        rel = qrels[qid]
        queries.append({"query_id": qid, "transcript": transcript, "dense": dense, "rel": rel,
                        "row": {"query_id": qid, "n_relevant": len(rel),
                                "baseline": metrics(dense, rel),
                                "oracle_Recall@5": {str(n): oracle_recall5(dense, rel, n)
                                                    for n in args.depths}}})
    if asr is not None:
        asr.close()

    summary = {"split": args.split, "queries": len(queries), "dense_depth": DENSE_DEPTH,
               "transcript_cache": str(args.transcript_cache), "reranker_device": args.reranker_device,
               "batch_size": args.batch_size,
               "n_relevant": {"mean": statistics.mean(q["row"]["n_relevant"] for q in queries),
                              "max": max(q["row"]["n_relevant"] for q in queries),
                              "share_more_than_5": float(np.mean([q["row"]["n_relevant"] > 5 for q in queries]))},
               "baseline": {m: float(np.mean([q["row"]["baseline"][m] for q in queries])) for m in METRICS},
               "oracle_Recall@5": {str(n): float(np.mean([q["row"]["oracle_Recall@5"][str(n)] for q in queries]))
                                   for n in args.depths},
               "configs": {}}

    # 2. Rerank with each model at each depth.
    for model_dir in args.models:
        rr = CrossEncoderReranker(model_dir, device=args.reranker_device, batch_size=args.batch_size)
        rr.warmup()
        name = rr.name
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            torch = None
        trunc = [rr.truncation_stats(q["transcript"], q["dense"][:max(args.depths)], data) for q in queries]
        for n in args.depths:
            key = f"{name}@N{n}"
            lat = []
            for i, q in enumerate(queries, start=1):
                out = rr.rerank(q["transcript"], q["dense"][:n], data, top_k=5)
                ranked = out["full_order"] + q["dense"][n:]
                q["row"][key] = {**metrics(ranked, q["rel"]), "rerank_ms": out["latency_ms"]}
                lat.append(out["latency_ms"])
            agg = {m: float(np.mean([q["row"][key][m] for q in queries])) for m in METRICS}
            agg.update({"rerank_ms_mean": statistics.mean(lat), "rerank_ms_median": statistics.median(lat),
                        "rerank_ms_p95": pct(lat, 0.95)})
            summary["configs"][key] = agg
            print(f"{key:<22} R@5 {agg['Recall@5']:.4f}  NDCG@10 {agg['NDCG@10']:.4f}  "
                  f"MRR@10 {agg['MRR@10']:.4f}  rerank median {agg['rerank_ms_median']:.1f} ms")
        summary.setdefault("models", {})[name] = {
            "path": str(model_dir), "load_ms": rr.load_ms,
            "truncated_pair_share": float(np.mean([t for t in trunc if t is not None])) if any(t is not None for t in trunc) else None,
            "peak_gpu_mem_mb": (torch.cuda.max_memory_allocated() / 2**20) if torch is not None and torch.cuda.is_available() else None,
        }
        del rr

    args.out_dir.mkdir(parents=True)
    with (args.out_dir / "per_query.jsonl").open("x", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q["row"], ensure_ascii=False) + "\n")
    with (args.out_dir / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nBaseline:", "  ".join(f"{m} {v:.4f}" for m, v in summary["baseline"].items()))
    print("Oracle Recall@5:", "  ".join(f"N{n} {v:.4f}" for n, v in summary["oracle_Recall@5"].items()))
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
