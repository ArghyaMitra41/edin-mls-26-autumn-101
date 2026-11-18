#!/usr/bin/env bash
# Build submission/supplementary_code.zip. It holds our new and changed files, the result
# files behind the report, a README, and a patch of the supplied files we changed.
#
#   bash submission/build_supplementary_zip.sh
#
# Before writing the zip, the script checks every file for UUN-like strings, personal
# paths, API keys and the words of your git user name. It stops if anything matches.
# To check more words as well (for example a teammate's surname or the Group ID):
#
#   EXTRA_TERMS="surname groupid" bash submission/build_supplementary_zip.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/submission/supplementary_code.zip"
BASE=6bbdfbc   # last commit of the supplied course repository
cd "$ROOT"

FILES=(
  hw1-asr/csrc/task1_rope.cu
  hw1-asr/csrc/task2_reduce.cuh
  hw1-asr/csrc/task2_reduce.cu
  hw1-asr/csrc/task3_gemm.cu
  hw1-asr/csrc/task4_attention.cu
  hw1-asr/hw1_gpu.txt
  hw1-asr/hw1_make_test.txt
  hw1-asr/hw1_profile.txt
  hw1-asr/hw1_profile_pytorch.txt
  cross.py
  rag/src/rag.py
  rag/src/evaluation/answer_judge.py
  rag/scripts/generate_answers.py
  rag/scripts/evaluate_rerank.py
  rag/extensions/reranker.py
  rag/extensions/bench_rerank.py
  rag/extensions/paired_analysis.py
  rag/tests/test_rerank.py
  rag/results/dev_baseline_retrieval.txt
  rag/results/rerank_sweep_dev/summary.json
  rag/results/rerank_sweep_dev/per_query.jsonl
  rag/results/rerank_sweep_practice/summary.json
  rag/results/rerank_sweep_practice/per_query.jsonl
  rag/results/rerank_bench_dev.json
  rag/results/dev_baseline.jsonl
  rag/results/dev_rerank_minilm_N20.jsonl
  rag/results/dev_rerank_bge_N10.jsonl
  rag/results/dev_baseline_judge/answer_judge_summary.json
  rag/results/dev_baseline_judge/answer_judge_per_query.jsonl
  rag/results/dev_rerank_minilm_N20_judge/answer_judge_summary.json
  rag/results/dev_rerank_minilm_N20_judge/answer_judge_per_query.jsonl
  rag/results/dev_rerank_bge_N10_judge/answer_judge_summary.json
  rag/results/dev_rerank_bge_N10_judge/answer_judge_per_query.jsonl
  rag/results/baseline_practice.jsonl
  rag/results/baseline_judge/answer_judge_summary.json
  rag/results/baseline_judge/answer_judge_per_query.jsonl
  rag/results/sample_answers.jsonl
  rag/results/sample_judge/answer_judge_summary.json
  rag/results/sample_judge/answer_judge_per_query.jsonl
)

# Remove any old zip first, so a failed check never leaves a stale zip to upload.
rm -f "$OUT"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
DEST="$STAGE/supplementary_code"

for f in "${FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "Missing file: $f" >&2
    exit 1
  fi
  mkdir -p "$DEST/$(dirname "$f")"
  cp "$f" "$DEST/$f"
done

# run.sh prints the absolute library path, which contains the home directory.
P="$DEST/hw1-asr/hw1_profile.txt"
sed -E 's#^Library: .*/hw1-asr/#Library: hw1-asr/#' "$P" > "$P.tmp"
mv "$P.tmp" "$P"

cp submission/README_supplementary.md "$DEST/README.md"
git --no-pager diff --no-color --no-ext-diff --diff-filter=M "$BASE" -- \
  hw1-asr/csrc rag/src rag/scripts > "$DEST/changes_to_supplied_files.patch"

# Anonymity and credential check. If anything matches, stop without writing a zip.
PATTERNS='(^|[^A-Za-z0-9])s[0-9]{7}([^0-9]|$)|/home/|/Users/|@ed\.ac\.uk|ELM_API_KEY *= *[^ ]|sk-[A-Za-z0-9]{16,}'
TERMS="$(git config user.name 2>/dev/null || true) ${EXTRA_TERMS:-}"
for w in $TERMS; do
  if [ "${#w}" -ge 3 ]; then
    PATTERNS="$PATTERNS|$w"
  fi
done
set +e
grep -r -n -I -i -E "$PATTERNS" "$DEST"
status=$?
set -e
if [ "$status" -eq 0 ]; then
  echo "Anonymity check FAILED: the lines above contain a name, UUN, personal path or key. No zip written." >&2
  exit 1
elif [ "$status" -ne 1 ]; then
  echo "Anonymity check could not run (grep exit status $status). No zip written." >&2
  exit 1
fi

(cd "$STAGE" && zip -r -X -q "$OUT" supplementary_code)
unzip -l "$OUT"
echo "Anonymity check passed. Wrote submission/supplementary_code.zip ($(du -h "$OUT" | cut -f1 | tr -d ' '))."
