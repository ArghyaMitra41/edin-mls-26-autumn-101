"""Latency / memory micro-benchmark of the reranking stage (no ELM calls).

For each model and candidate depth N: warm up on the first queries, then rerank
every query's top-N dense candidates `--repeats` times. Reports per-query
latency (mean, std, median, p95), throughput in pairs/s, model load time and
peak GPU memory. The dense candidates are computed once, so only the reranker
is timed.

Example:
  python rag/extensions/bench_rerank.py --device cpu --split dev \
      --transcript-cache rag/.cache/dev_transcripts.jsonl \
      --models rag/models/rerankers/minilm-l6 rag/models/rerankers/bge-base \
      --out rag/results/rerank_bench_dev.json
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions"))

import argparse
import json
import platform
import statistics
import time

from src.data import CourseData
from src.encoder import LocalEncoder
from src.retriever import ExactDenseRetriever
from scripts.evaluate_rag import load_cache
from reranker import CrossEncoderReranker


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, max(0, int(round(q * len(xs))) - 1))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["practice", "dev"], default="dev")
    ap.add_argument("--transcript-cache", type=Path, required=True)
    ap.add_argument("--models", type=Path, nargs="+", required=True)
    ap.add_argument("--depths", type=int, nargs="+", default=[10, 20, 50, 100])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup-queries", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None, help="enc_a device")
    ap.add_argument("--reranker-device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error("Output exists; choose a new filename")

    import torch

    data_dir = ROOT / "data" / "rag_corpus"
    data = CourseData(data_dir)
    cache = load_cache(args.transcript_cache)
    items = [it for it in data.audio_items(args.split) if it["query_id"] in cache]
    if not items:
        ap.error("No cached transcripts for this split; run evaluate_rag.py with --transcript-cache first")
    encoder = LocalEncoder(ROOT / "models" / "enc_a", device=args.device)
    retriever = ExactDenseRetriever(data_dir / "embeddings" / "corpus_enc_a.npy",
                                    data_dir / "embeddings" / "corpus_enc_a_ids.json")
    queries = [(cache[it["query_id"]],
                retriever.retrieve(encoder.encode(cache[it["query_id"]]), k=max(args.depths))["record_ids"])
               for it in items]

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    result = {"split": args.split, "queries": len(queries), "repeats": args.repeats,
              "batch_size": args.batch_size, "reranker_device": args.reranker_device, "gpu": gpu,
              "torch": torch.__version__, "python": platform.python_version(), "models": {}}

    for model_dir in args.models:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        rr = CrossEncoderReranker(model_dir, device=args.reranker_device, batch_size=args.batch_size)
        rr.warmup()
        entry = {"load_ms": rr.load_ms, "depths": {}}
        for n in args.depths:
            for transcript, dense in queries[:args.warmup_queries]:
                rr.rerank(transcript, dense[:n], data, top_k=5)
            lat = []
            wall = time.perf_counter()
            for _ in range(args.repeats):
                for transcript, dense in queries:
                    lat.append(rr.rerank(transcript, dense[:n], data, top_k=5)["latency_ms"])
            wall = time.perf_counter() - wall
            entry["depths"][str(n)] = {
                "mean_ms": statistics.mean(lat), "std_ms": statistics.stdev(lat),
                "median_ms": statistics.median(lat), "p95_ms": pct(lat, 0.95),
                "pairs_per_s": n * len(lat) / (sum(lat) / 1000.0),
                "wall_s": wall,
            }
            d = entry["depths"][str(n)]
            print(f"{rr.name:<12} N={n:<4} median {d['median_ms']:8.2f} ms  p95 {d['p95_ms']:8.2f} ms  "
                  f"{d['pairs_per_s']:8.0f} pairs/s")
        entry["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
        result["models"][rr.name] = entry
        del rr

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
