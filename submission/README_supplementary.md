# Supplementary code

This folder accompanies the report *Spoken-question RAG on our own CUDA kernels: does
cross-encoder reranking pay for itself?*

Paths are relative to the root of the supplied course repository. The folder holds only the
files we wrote or changed, plus the result files behind the report's tables and figures. All
other supplied code, data and models are unchanged and are not included. To run anything,
copy this folder over a clone of the course repository. You do not need to run anything to
check the numbers: the result files are here.

## Our code

| File | Status | What it does | Report |
|---|---|---|---|
| `hw1-asr/csrc/task1_rope.cu` | supplied, blanks completed | RoPE | §2.1, §4.1–4.2 |
| `hw1-asr/csrc/task2_reduce.cuh`, `task2_reduce.cu` | supplied, blanks completed | warp and block reductions, RMSNorm, softmax | §2.1, §4.1–4.2 |
| `hw1-asr/csrc/task3_gemm.cu` | supplied, blanks completed | tiled GEMM with fused bias and GELU | §2.1, §4.1–4.2 |
| `hw1-asr/csrc/task4_attention.cu` | supplied, blanks completed | FlashAttention-style attention (GQA, causal mask) | §2.1, §4.1–4.2 |
| `rag/src/rag.py` | changed | optional reranking stage between search and prompt; `answer_transcript()` runs everything after ASR | §2.2 |
| `rag/scripts/generate_answers.py` | changed | adds `--transcript-cache`, `--reranker`, `--candidate-depth` and `--reranker-device`; records `rerank_ms` and `prompt_chars` | §2.2, §3 |
| `rag/src/evaluation/answer_judge.py` | changed | when the judge returns malformed JSON, reads the three scores with a regular expression (`v0.3.1`); valid replies are scored as before | §2.2 |
| `rag/extensions/reranker.py` | new | `CrossEncoderReranker`: scores the top-N (question, record) pairs in batches of 32 and keeps the best k; ties keep the dense order | §2.2 |
| `rag/scripts/evaluate_rerank.py` | new | retrieval sweep over models and N, with the oracle Recall@5 (no ELM calls) | §4.3, Fig. 2, Table 5 |
| `rag/extensions/bench_rerank.py` | new | reranker latency, throughput, load time and peak GPU memory (no ELM calls) | §4.3, Table 2 |
| `rag/extensions/paired_analysis.py` | new | paired bootstrap confidence intervals and sign-flip tests on the judge scores, and rank-1 counts | §4.4–4.5, Table 3 |
| `rag/tests/test_rerank.py` | new | 7 unit tests with a stub cross-encoder: ordering, ties, invalid depths, baseline path unchanged, output schema | §2.2 |
| `cross.py` | new | downloads the two cross-encoders into `rag/models/rerankers/` | — |
| `changes_to_supplied_files.patch` | — | the exact diff of the 8 supplied files above marked "blanks completed" or "changed" | — |

We also changed `rag/.gitignore` so that result files are kept and downloaded models are
ignored. That file is not included.

## Result files

| File | What it holds | Report |
|---|---|---|
| `hw1-asr/hw1_gpu.txt`, `hw1_make_test.txt` | `nvidia-smi -L` and `make test` output | §4.1, App. A |
| `hw1-asr/hw1_profile.txt`, `hw1_profile_pytorch.txt` | `./run.sh --profile`, on our kernels and with `--cuda none` (library path shortened) | Tables 1 and 4 |
| `rag/results/rerank_sweep_{practice,dev}/` | `evaluate_rerank.py` sweeps: `summary.json` (means) and `per_query.jsonl` | Fig. 2, Table 5 |
| `rag/results/dev_baseline_retrieval.txt` | the supplied `evaluate_rag.py` on dev; it matches the sweep's baseline row | Table 3 |
| `rag/results/rerank_bench_dev.json` | `bench_rerank.py` on dev | Table 2 |
| `rag/results/dev_{baseline,rerank_minilm_N20,rerank_bge_N10}.jsonl` | per question: transcript, retrieved records, answer, timings, prompt size | Table 3 |
| `rag/results/dev_*_judge/` | judge scores: `answer_judge_summary.json` and `answer_judge_per_query.jsonl` | Table 3, §4.4 |
| `rag/results/baseline_practice.jsonl`, `baseline_judge/` | practice-split baseline answers and judge scores | App. B |
| `rag/results/sample_answers.jsonl`, `sample_judge/` | 5 practice questions answered and judged a second time (judge-noise check) | §4.4 |

## Dependencies

- The supplied environments: `hw1-asr/setup_env.sh` (conda environment `mls-asr`) and
  `rag/requirements.txt` (the RAG virtual environment). We added no packages.
- The course data from `rag-assets.zip`, installed as described in `rag/README.md`. An ELM API
  key in `rag/.env` is needed only to generate and judge answers.
- Two public cross-encoders from Hugging Face: `cross-encoder/ms-marco-MiniLM-L-6-v2` and
  `BAAI/bge-reranker-base`. Run `python cross.py` from the repository root on a machine with
  internet access. It saves them as `rag/models/rerankers/minilm-l6` and `rag/models/rerankers/bge-base`.
- Our numbers come from one NVIDIA GeForce RTX 2080 Ti with CUDA 12.8.

## Commands

Run on a GPU node after the setup in `hw1-asr/README.md` and `rag/README.md`.

HW1, from `hw1-asr/` (conda environment `mls-asr`):

```bash
make test
./run.sh --profile
./run.sh --cuda none --profile
```

RAG, from the repository root (RAG environment active and `RAG_ASR_PYTHON` set, as in `rag/README.md`):

```bash
python -m unittest discover -s rag/tests -p "test_rerank.py" -v

# Run ASR once per split; every later command reads the cached transcripts
python rag/scripts/evaluate_rag.py --device cpu --split dev \
    --transcript-cache rag/.cache/dev_transcripts.jsonl

# Retrieval sweep and cost benchmark (no ELM calls)
python rag/scripts/evaluate_rerank.py --device cpu --split dev \
    --transcript-cache rag/.cache/dev_transcripts.jsonl \
    --models rag/models/rerankers/minilm-l6 rag/models/rerankers/bge-base \
    --depths 10 20 50 100 --out-dir rag/results/rerank_sweep_dev
python rag/extensions/bench_rerank.py --device cpu --split dev \
    --transcript-cache rag/.cache/dev_transcripts.jsonl \
    --models rag/models/rerankers/minilm-l6 rag/models/rerankers/bge-base \
    --out rag/results/rerank_bench_dev.json

# Answers and judge scores (ELM calls). The baseline is the same command
# without --reranker and --candidate-depth; MiniLM used --candidate-depth 20.
python rag/scripts/generate_answers.py --device cpu --split dev \
    --transcript-cache rag/.cache/dev_transcripts.jsonl \
    --reranker rag/models/rerankers/bge-base --candidate-depth 10 \
    --output rag/results/dev_rerank_bge_N10.jsonl
python rag/scripts/evaluate_answers.py --split dev \
    --candidates rag/results/dev_rerank_bge_N10.jsonl \
    --out-dir rag/results/dev_rerank_bge_N10_judge

# Paired analysis (standard library only; runs on the included result files)
python rag/extensions/paired_analysis.py rag/results
```

The scripts refuse to overwrite existing outputs, so use new output names when re-running.
`--device cpu` keeps the query encoder on the CPU, as in the supplied guide. The rerankers
run on the GPU (`--reranker-device`, default `cuda`). The practice split uses the same
commands with `--split practice` and its own transcript cache.
