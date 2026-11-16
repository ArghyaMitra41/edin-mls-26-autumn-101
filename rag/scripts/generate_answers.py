"""Generate answer candidates using the audio-to-answer baseline.

Investigation options (all off by default, so the baseline command is unchanged):
  --transcript-cache  reuse cached ASR transcripts (ASR only runs for missing ones)
  --reranker / --candidate-depth  rerank the top-N dense candidates to the top k
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
from src.asr_adapter import ASRAdapter
from src.context_builder import SimpleContextBuilder
from src.data import CourseData
from src.encoder import LocalEncoder
from src.generator import ELMGenerator
from src.rag import RAGPipeline
from src.retriever import ExactDenseRetriever
from scripts.evaluate_rag import load_cache, append_cache

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["practice", "dev"], required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None, help="Diagnostic subset only")
    ap.add_argument("--transcript-cache", type=Path, default=None,
                    help="Reuse cached transcripts (same format as evaluate_rag.py)")
    ap.add_argument("--reranker", type=Path, default=None, help="Local cross-encoder model dir")
    ap.add_argument("--candidate-depth", type=int, default=None,
                    help="Dense candidates passed to the reranker (N >= k)")
    ap.add_argument("--reranker-device", default="cuda")
    args = ap.parse_args()
    if args.k < 1 or (args.limit is not None and args.limit < 1):
        ap.error("--k and --limit must be positive")
    if (args.reranker is None) != (args.candidate_depth is None):
        ap.error("--reranker and --candidate-depth must be given together")
    if args.candidate_depth is not None and args.candidate_depth < args.k:
        ap.error("--candidate-depth must be at least --k")
    if args.output.exists():
        ap.error("Output exists; choose a new filename to preserve prior results")
    asr = ASRAdapter()
    cache = load_cache(args.transcript_cache) if args.transcript_cache else None
    if cache is None:
        asr.prepare()  # with a cache, ASR is only started if a transcript is missing
    reranker = None
    if args.reranker is not None:
        sys.path.insert(0, str(ROOT / "extensions"))
        from reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(args.reranker, device=args.reranker_device)
        reranker.warmup()
    data_dir = ROOT / "data/rag_corpus"
    data = CourseData(data_dir)
    pipeline = RAGPipeline(asr, LocalEncoder(ROOT / "models/enc_a", device=args.device),
        ExactDenseRetriever(data_dir / "embeddings/corpus_enc_a.npy", data_dir / "embeddings/corpus_enc_a_ids.json"),
        data, SimpleContextBuilder(), ELMGenerator(),
        reranker=reranker, candidate_depth=args.candidate_depth)
    items = data.audio_items(args.split)
    if args.limit:
        items = items[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        for item in items:
            qid = item["query_id"]
            audio = data_dir / item["audio_file"]
            if cache is None:
                result = pipeline.answer_audio(audio, k=args.k)
            else:
                transcript = cache.get(qid)
                if transcript is None:
                    transcript = asr.transcribe(audio)
                    append_cache(args.transcript_cache, qid, transcript)
                    cache[qid] = transcript
                result = pipeline.answer_transcript(transcript, k=args.k)
            row = {"query_id": qid, "split": args.split, "audio_file": item["audio_file"],
                   "transcript": result["transcript"], "answer": result["answer"],
                   "retrieved_record_ids": result["record_ids"], "timing": result["timing"]}
            if args.transcript_cache or reranker is not None:
                row.update({"transcript_source": "cache" if cache is not None else "asr",
                            "reranker": reranker.name if reranker else None,
                            "candidate_depth": args.candidate_depth,
                            "prompt_chars": result["prompt_chars"]})
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(qid)

if __name__ == "__main__":
    main()
