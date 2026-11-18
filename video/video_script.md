# Coursework video: recording script

What is in this folder:

- `video_script.md` (this file): what to say, what to show, every command, and how to edit and check the video.
- `slides.html`: the slide deck.
  - Open it in Chrome or Safari and press `f` for full screen.
  - `→` or space moves to the next slide, `←` goes back.
  - The mouse pointer hides itself after 2 seconds.
  - Optional: copy the demo question's audio next to it as `question.wav`, and a player appears on slide 8 (see 1.3). Never commit that file; `video/.gitignore` already ignores it.

**Target length: 9:30 (the limit is 10:00).**
- **Part A, 0:00–7:30**: presentation and live demonstration (at most 8 minutes).
- **Part B, 7:30–9:30**: Question 1 and Question 2, about one minute each, as the course asks.

The narration (the "Say" blocks) is written for the markers who watch the video. It runs at about 150 words per minute. Read it in your own voice, and change words that feel unnatural, but keep the numbers.

Sounds to agree on:
- **enc_a** is "enc A"; **bge** is "B-G-E".
- **Recall@5** is "recall at five".
- **GELU** is "gee-loo"; **RoPE** is "rope".

---

## At a glance

| # | Time | Screen | What happens |
|---|---|---|---|
| 1 | 0:00–0:15 | Slide 1 | Title and our question |
| 2 | 0:15–0:45 | Slide 2 | The pipeline; supplied vs ours |
| 3 | 0:45–1:25 | Slide 3 | The four kernel groups |
| 4 | 1:25–1:50 | Terminal | `make test`: 33 of 33 pass |
| 5 | 1:50–2:25 | Terminal | `./run.sh --profile`: PASS, linear is 83% of GPU time |
| 6 | 2:25–2:55 | Slide 4 | 3× slower than PyTorch, and why |
| 7 | 2:55–3:25 | Slide 5 | Where RAG loses quality; the question |
| 8 | 3:25–3:55 | Slide 6 | How the reranking stage works |
| 9 | 3:55–4:15 | Slide 7 | How we measured |
| 10 | 4:15–5:00 | Slide 8, then terminal | Live run 1: baseline, from audio to answer |
| 11 | 5:00–5:40 | Terminal | Live run 2: same question with the bge reranker |
| 12 | 5:40–6:05 | Slide 9 | Retrieval vs N, with the oracle |
| 13 | 6:05–6:35 | Slide 10 | Answer quality and cost |
| 14 | 6:35–7:05 | Slide 11 | Why reranking does not help |
| 15 | 7:05–7:30 | Slide 12 | Conclusion and limitations |
| Q1 | 7:30–8:30 | Slide 13 | Question 1: our main design choice |
| Q2 | 8:30–9:30 | Slide 14, terminal, slide 14 | Question 2: how we checked our work |

The times are for the edited video, after the long waits are cut (model loading, and the reranker plus the ELM calls). If your first full edit runs over 9:45, see [If you run long](#if-you-run-long).

---

## 1. Before you record

### 1.1 Anonymity checklist (from the submission requirements)

- [ ] No webcam. No names, UUNs, email addresses or group ID, on screen or spoken.
- [ ] Turn on Do Not Disturb. Quit Mail, Messages, Teams, Slack and WhatsApp.
- [ ] Show the slides only in full screen (`f`). The browser address bar would show `file:///Users/<your Mac user name>/...`.
- [ ] Show the terminal only in full screen (Terminal: View → Enter Full Screen), with one tab. This hides the window title, which can show your login or the `ssh` command.
- [ ] Start recording after you have logged in to the cluster: the `ssh` lines contain your UUN.
- [ ] Set a neutral prompt (`export PS1='$ '`, in 1.3). The default cluster prompt shows your UUN.
- [ ] Do not run `check_setup.py` on camera: it prints your home path. `run.sh` also prints a path with your UUN; the command in segment 5 replaces it with `~`.
- [ ] Never show `rag/.env`.
- [ ] Export the file as `video.mp4` and check its metadata (section 5).

### 1.2 Screen and OBS set-up (from the video production guide)

1. OBS → Tools → Auto-Configuration Wizard → "Optimize just for recording". Keep the canvas at your screen's shape; 30 fps is enough.
2. Add a source: **macOS Screen Capture**, method **Display Capture**, and **untick "Show cursor"**.
3. Settings → Audio: choose your microphone and check that its meter moves. Mute Desktop Audio, except for the one take where you play the question audio (segment 10, optional).
4. Terminal font: about 100 columns should fill the screen width (around 22–26 pt on a laptop). Check that `tput cols` prints 100–110 and that the `make test` lines do not wrap.
5. Make a 10-second test recording of the terminal and of slide 10. Play it back and check that the smallest text is readable and the voice is clear.
6. Record each **take** (section 2) as a separate file. You join them in Shotcut.

### 1.3 Prepare the cluster session (off camera)

Run these before you start recording. Book the same GPU type as in the report, so the numbers on screen match.

```bash
# on your Mac (off camera: these lines contain your UUN)
ssh <UUN>@ssh.inf.ed.ac.uk
ssh mlp.inf.ed.ac.uk

# on mlp: book an RTX 2080 Ti for 90 minutes
srun -p Teaching --gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=2 --mem=16G --time=01:30:00 --pty bash

# on the GPU node
export PATH=/opt/cuda-12.8.0/bin:$PATH
conda activate mls-asr
cd ~/edin-mls-26-autumn-101
make -C hw1-asr lib

# neutral prompt and window title
export PS1='$ '
unset PROMPT_COMMAND
printf '\033]0;terminal\007'
```

**Optional, for slide 8:** copy the question audio to your Mac and save it as `video/question.wav`. Run this off camera, in a second Mac terminal, from the repository folder on your Mac. It hops through the gateway in one command. If it fails, skip the audio: it is optional.

```bash
scp -J <UUN>@ssh.inf.ed.ac.uk \
  <UUN>@mlp.inf.ed.ac.uk:edin-mls-26-autumn-101/rag/data/rag_corpus/audio/dev/q_33fd7baba78a1358.wav \
  video/question.wav
```

### 1.4 The display function `show` (paste once, before the RAG takes)

`run_audio_query.py` prints full forum posts and an audio path that contains your UUN, and `generate_answers.py` writes JSON lines. `show` only formats a saved result. It prints:
- the transcript;
- the 5 record IDs, marking those listed as relevant in the dev relevance labels (qrels);
- Recall@5;
- the first 10 lines of the answer;
- the timings and the prompt size.

It never prints paths. Run it from the repository root.

```bash
show() {  # usage: show FILE QUERY_ID SPLIT  (prints one result; relevant posts marked from the qrels)
python - "$@" <<'PY'
import csv, json, sys, textwrap
path, qid, split = sys.argv[1:4]
text = open(path, encoding="utf-8").read()
try:
    rows = [json.loads(text)]
except json.JSONDecodeError:
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
r = next(row for row in rows if row.get("query_id") == qid)
ids = r.get("record_ids") or r["retrieved_record_ids"]
with open(f"rag/data/rag_corpus/qrels_{split}.tsv", encoding="utf-8-sig", newline="") as f:
    rel = {x["record_id"] for x in csv.DictReader(f, delimiter="\t")
           if x["query_id"] == qid and float(x.get("score") or 1) > 0}
how = f"reranker {r['reranker']} over the top {r['candidate_depth']}" if r.get("reranker") else "baseline (no reranker)"
src = "made by ASR in this run" if "asr_ms" in r["timing"] else "read from the cache"
print(f"Run:        {how}; transcript {src}")
print(f"Transcript: {r['transcript']}\n")
print(f"Top {len(ids)} posts given to the language model:")
for i, rid in enumerate(ids, 1):
    print(f"  {i}. {rid}   {'relevant' if rid in rel else ''}")
hits = [i for i, rid in enumerate(ids, 1) if rid in rel]
first = f", first at rank {hits[0]}" if hits else ""
recall = len(hits) / len(rel) if rel else 0.0
print(f"Relevant posts for this question: {len(rel)}. In the top {len(ids)}: {len(hits)} "
      f"(Recall@{len(ids)} = {recall:.2f}){first}.\n")
lines = [w for para in r["answer"].splitlines() for w in (textwrap.wrap(para, 92) or [""])]
print("Answer:")
for line in lines[:10]:
    print("  " + line)
if len(lines) > 10:
    print(f"  ... ({len(lines) - 10} more lines)")
print("\nTiming (ms): " + " | ".join(f"{k.removesuffix('_ms')} {v:.0f}" for k, v in r["timing"].items()))
print(f"Prompt:      {r['prompt_chars']} characters")
PY
}
```

### 1.5 Rehearse each part just before you record it (off camera)

Work in this order, so you never have to switch back to an earlier environment:

1. **HW1** (conda `mls-asr` only):
   - `cd hw1-asr`, then run `make test` and the `run.sh` line from segment 5 once. This builds the test binaries and warms the file cache, so the model loads faster on camera.
   - Then `clear`, and record T2 and T3.
2. **RAG:**
   - Run the set-up block at the start of segment 10 and paste `show` (1.4).
   - Run both demo runs and the Question 2 commands once. Write down the numbers you will read aloud: the ASR time in run 1 and the rerank time in run 2. Also note which post in run 2 is marked `relevant` at rank 4 or 5 (expected: one of `r_546ed46a6b5df9b9` and `r_fc9e0bfa7e1ce5f2`).
   - Then delete the demo outputs, because `generate_answers.py` refuses to overwrite a file, and clear the screen. After that, record T5, T6 and the terminal clip of Question 2.

   ```bash
   rm -f /tmp/video_base.json /tmp/video_bge.jsonl /tmp/video_r1.log /tmp/video_r2.log
   clear
   ```

3. The slide takes (T1, T4, T7, T8 and the slide clips of T9) do not need the cluster; record them whenever you like.

The rehearsal and the recording together use 6 ELM generation calls and no judge calls. If your `srun` session ends, repeat the steps in 1.3.

---

## 2. Takes (record each as its own file)

| Take | Screen | Segments |
|---|---|---|
| T1 | Slides 1–3 | 1, 2, 3 |
| T2 | Terminal | 4 (`make test`) |
| T3 | Terminal | 5 (`run.sh --profile`) |
| T4 | Slides 4–8 | 6, 7, 8, 9, and the first sentence of 10 |
| T5 | Terminal | rest of 10 (run 1) |
| T6 | Terminal | 11 (run 2) |
| T7 | Slides 9–12 | 12, 13, 14, 15 |
| T8 | Slide 13 | Question 1 |
| T9 | Slide 14, then terminal, then slide 14 | Question 2 (three short clips) |

The HW1 takes (T2, T3) run in the `mls-asr` environment. Before T5, switch to the RAG set-up in segment 10.

---

## 3. Part A: presentation and demonstration (0:00–7:30)

### Segment 1: Title (0:00–0:15) · Slide 1

**Say** (35 words):
> Our system answers spoken questions about personal-finance forum posts. The speech recogniser runs on CUDA kernels we wrote ourselves. Our investigation asks whether a cross-encoder reranker improves the answers enough to pay for its cost.

### Segment 2: The system (0:15–0:45) · Slide 2

**Say** (68 words):
> A spoken question goes into the ASR model, which runs on our kernels. A small encoder, enc_a, turns the transcript into a vector, and exact search finds the closest of 24,000 forum posts. A language model, EuroLLM-22B, then answers from the question and five posts. Blue is our work: the kernels, and a new reranking stage between search and prompt. The baseline is the same pipeline without it.

### Segment 3: The kernels (0:45–1:25) · Slide 3

**Say** (91 words):
> We wrote four kernel groups. RoPE uses one thread per pair of numbers, with one launch for both queries and keys. The reductions use warp shuffles, then shared memory across warps; softmax keeps a running maximum and sum, so it reads each row twice, not three times. The GEMM works on 16 by 16 tiles in shared memory, loads the transposed weights in a coalesced way, and fuses bias and GELU into the store. Attention never stores the score matrix; it keeps a running maximum, sum and output for each query.

### Segment 4: `make test` (1:25–1:50) · Terminal (take T2)

**Do:** from `hw1-asr` (`cd hw1-asr`, off camera), clear the screen, then type:
```bash
make test
```
**Edit:** if the run takes more than about 5 seconds, cut the middle and add the caption **"Waiting time shortened"** for 2 seconds.

**Point at:**
- the four `ALL PASS` lines;
- `audio encoder (partial, 32/64) … 490 GB/s (80% of peak)` near the top: scroll up to it while you say the RoPE sentence, then scroll back down;
- `encoder q_proj (1500x1280x1280) … 1939 GFLOP/s`, with `cuBLAS 13195` on the line below it.

**Say**, once the output has finished (50 words):
> make test checks every kernel against a CPU reference. All 33 cases pass on our RTX 2080 Ti. RoPE on the encoder shape reaches about 490 gigabytes per second, which is 80 percent of peak bandwidth. Our GEMM on the encoder's q-projection reaches about 1.9 teraflops, against 13.2 for cuBLAS.

### Segment 5: The whole model (1:50–2:25) · Terminal (take T3)

**Do:** clear the screen, then type the line below. The `sed` replaces your home folder with `~` in the library path that `run.sh` prints.
```bash
./run.sh --profile 2>&1 | sed "s#$HOME#~#g"
```
**Say**, while it starts (11 words):
> Now the whole model, with all four tasks on our kernels.

**Edit:** the output appears all at once, after the model has loaded (about 75–90 s). Cut the wait and add the caption **"Waiting time shortened (model loading)"** for 3 seconds.

**Point at:** `Transcription:` and `Accuracy: 100.0%`, `Status: PASS`, the `Time:` line, then the `linear` row of the profile (about 83%).

**Say**, after the output (55 words):
> The transcription matches the reference exactly, and the status is PASS. One transcription of the 3.5-second clip takes about 3.2 seconds. The profile shows where the time goes: the linear layers, which are our GEMM, take 83 percent of GPU time, and attention most of the rest. RoPE and RMSNorm take only a few milliseconds.

If the live time differs a little from 3.2 s, say what is on screen; the measured numbers are in the report.

### Segment 6: ASR results (2:25–2:55) · Slide 4

**Say** (65 words):
> Against the supplied PyTorch version, we are three times slower: 3.2 seconds against 1.07. The gap is all in the GEMM; our RoPE and RMSNorm are in fact much faster, because PyTorch builds them from many small operations. In our GEMM, each multiply-add needs two shared-memory reads, so shared memory is the limit. cuBLAS reuses values from registers instead, and that is our next step.

### Segment 7: Where RAG loses quality (2:55–3:25) · Slide 5

**Say** (72 words):
> For RAG, we first looked for where quality is lost. On the practice questions, 77 percent of relevant posts are in enc_a's top 100, but only 42 percent reach the top 5 that the model sees. And answers score 0.14 when none of the top 5 is relevant, against 0.47 otherwise. So we asked: does reranking the top N with a cross-encoder give better evidence and better answers, and at what cost?

### Segment 8: The reranking stage (3:25–3:55) · Slide 6

**Say** (64 words):
> enc_a encodes the question and each post separately. A cross-encoder reads them together, which usually ranks better but costs one transformer pass per post. So our stage takes the top N from exact search, scores the N pairs on the GPU, and keeps the best five, with ties in enc_a's order. Without a reranker the original code runs, and k is 5 either way.

### Segment 9: How we measured (3:55–4:15) · Slide 7

**Say** (42 words):
> We tested two public rerankers: MiniLM, with 22 million parameters, and bge-base, with 278 million, at N from 10 to 100. We chose N on the 350 practice questions and report on the 125 dev questions, all with the same cached transcripts.

### Segment 10: Live run 1, the baseline from audio (4:15–5:00) · Slide 8, then terminal (end of T4, then T5)

**On slide 8, say** (8 words):
> Now the system running, on dev question 2.

Optional: click play on the slide's audio player and let the question play (about 6 s).

**Before T5 (off camera):** switch the terminal to the RAG set-up, paste the `show` function from 1.4, and clear the screen:
```bash
cd ~/edin-mls-26-autumn-101
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"
source "$HOME/mls-rag-env/bin/activate"
rm -f /tmp/video_base.json /tmp/video_bge.jsonl /tmp/video_r1.log /tmp/video_r2.log
clear
```

**Do (T5):** type the first command, paste the second:
```bash
export ASR_CUDA=all
python rag/scripts/run_audio_query.py --device cpu --split dev \
    --query-id q_33fd7baba78a1358 > /tmp/video_base.json 2> /tmp/video_r1.log
```
**Say**, while it runs (14 words):
> Run 1 is the baseline, from audio to answer, with ASR on our kernels.

**Edit:** cut the wait (ASR model loading, then one ELM call; about 2 minutes) and add the caption **"Waiting time shortened (model loading and answer generation)"**. Then type:
```bash
show /tmp/video_base.json q_33fd7baba78a1358 dev
```
**Expected:**

```text
Run:        baseline (no reranker); transcript made by ASR in this run
Transcript: Why might a bank prefer a lower all cash offer over a higher offer through conventional lending?

Top 5 posts given to the language model:
  1. r_25dd7f9bd795774e   relevant
  2. r_ebb4bae4b3af65ff
  3. r_a09d83608eae6749
  4. r_27ccf35eab559e86
  5. r_cdd23a3f18f9e5b8
Relevant posts for this question: 6. In the top 5: 1 (Recall@5 = 0.17), first at rank 1.
```
The answer and timings follow. The answer text changes from run to run (temperature 0.1).

**Say**, after the output (54 words; «X» is the ASR time on your screen):
> Our kernels transcribed: "Why might a bank prefer a lower all-cash offer over a higher offer through conventional lending?" Search returned these five posts. Six posts are labelled relevant for this question, and one of them is in the top 5, at rank 1. ASR took about «X» seconds, and search a few milliseconds.

**If the output differs:**
- If the transcript or the five IDs differ from the expected output, read what is on screen and do not claim "rank 1". Say only how many relevant posts are in the top 5.
- If generation is slow, keep the cut: the generation time depends on the shared ELM service.

### Segment 11: Live run 2, our reranker (5:00–5:40) · Terminal (take T6)

**Do:** clear the screen and paste:
```bash
python rag/scripts/generate_answers.py --device cpu --split dev --limit 2 \
    --transcript-cache rag/.cache/dev_transcripts.jsonl \
    --reranker rag/models/rerankers/bge-base --candidate-depth 10 \
    --output /tmp/video_bge.jsonl 2> /tmp/video_r2.log
```
**Say**, while it runs (35 words):
> Run 2 is our change: the bge reranker over the top 10. It reuses the cached transcript, so the text is identical. It answers the first two dev questions, and we look at the second.

`generate_answers.py` has no option to pick a single question, so `--limit 2` reaches dev question 2. It prints the two question IDs as it finishes them.

**Edit:** cut the wait (loading the reranker, then two ELM calls) and add **"Waiting time shortened"**. Then type:
```bash
show /tmp/video_bge.jsonl q_33fd7baba78a1358 dev
```
**Expected** (the same transcript as run 1):

```text
Run:        reranker bge-base over the top 10; transcript read from the cache
...
Top 5 posts given to the language model:
  1. r_a09d83608eae6749
  2. r_25dd7f9bd795774e   relevant
  3. r_cdd23a3f18f9e5b8
  4. r_546ed46a6b5df9b9          (one of ranks 4 and 5 is marked relevant)
  5. r_fc9e0bfa7e1ce5f2
Relevant posts for this question: 6. In the top 5: 2 (Recall@5 = 0.33), first at rank 2.
```

**Point at:**
- the `rerank` value in the timing line;
- rank 1, which was rank 3 in run 1 and is not relevant;
- rank 2, which was rank 1 in run 1;
- the relevant post at rank 4 or 5, which was not in run 1's top 5.

**Say**, after the output (52 words; «X» is the rerank time on your screen):
> Reranking took about «X» milliseconds. The top 5 now holds two relevant posts instead of one: the reranker pulled one up from below rank 5. But it also put a post that is not relevant at rank 1, above the one that was first. Across the dev set we see both effects.

### Segment 12: Retrieval results (5:40–6:05) · Slide 9 (take T7 starts)

**Say** (63 words):
> Across all 125 dev questions, the grey line is the oracle: the best any reordering of the top N could do. It rises from 0.47 to 0.71. Both rerankers do the opposite: Recall@5 falls as N grows, and the larger bge model falls fastest. Only at N of 10 are they slightly above the baseline, and on the practice split even that disappears.

### Segment 13: Answers and cost (6:05–6:35) · Slide 10

**Say** (73 words):
> For answers, we ran the baseline and each model's chosen setting through EuroLLM and the judge. bge at N of 10 raised the mean score from 0.314 to 0.349, but the 95 percent interval runs from minus 0.02 to plus 0.09, so we cannot claim a gain. MiniLM was slightly below. The cost: 61 to 111 milliseconds per question, up to 1.6 gigabytes of GPU memory, and 8 to 21 percent longer prompts.

### Segment 14: Why reranking does not help (6:35–7:05) · Slide 11

**Say** (80 words):
> Better evidence does help: where bge raised Recall@5, the judge score rose by 0.18 on average. But the rerankers rarely improve the evidence. The baseline has a relevant post at rank 1 for 47 dev questions, and bge keeps only 28 of them. The likely cause is a domain mismatch: these models were trained on web search or general data, not on long forum posts and spoken questions. Prompt length, truncation, ASR errors and judge noise do not explain it.

### Segment 15: Conclusion (7:05–7:30) · Slide 12

**Say** (56 words):
> To sum up: our kernels are correct, and the memory-bound ones are fast, but the simple GEMM makes ASR three times slower than PyTorch. Off-the-shelf reranking costs time, memory and prompt length, and gives no measurable answer gain on this corpus. The main limits are 125 dev questions, one judgement per answer, and one GPU type.

---

## 4. Part B: the two questions (the final two minutes)

Say "Question 1" and "Question 2" clearly; the slides show the same labels. If the video ends at 9:30, Question 1 starts at 7:30.

### Question 1: our main design choice (7:30–8:30) · Slide 13 (take T8)

**Say** (147 words):
> Question 1: our main design choice. We added reranking as a separate stage between search and the prompt, and kept k fixed at 5, so the model always sees exactly five posts. Only which five, and their order, can change. We also gave every run the same cached transcripts. We chose this because the practice data showed that the loss was in the ordering: 77 percent of relevant posts were in the top 100, but only 42 percent in the top 5, and answers without relevant evidence scored 0.14 against 0.47. A larger k would also bring in more relevant posts, but every prompt would get longer, so we could not tell better evidence from more text. An approximate index would only speed up a search that already takes 5 milliseconds. Holding k and the transcripts fixed is what lets us attribute any change to the reranker.

### Question 2: how we checked our work (8:30–9:30) · Slide 14, terminal, slide 14 (take T9)

Record three short clips and join them.

**Clip 1, slide 14 (about 3 s). Say** (7 words):
> Question 2: how we checked our work.

**Clip 2, terminal (about 35 s).** Run this from the repository root, with the RAG environment active:
```bash
python -m unittest discover -s rag/tests -p "test_rerank.py" -v
```
**Say** (48 words):
> First, unit tests with stub models. A fake cross-encoder scores each post by a number written in its text, so we know the exact order it must return. One test checks that without a reranker, the pipeline still returns the retriever's top 5, as the original code did.

Then clear the screen and run:
```bash
grep '"Recall@100"' rag/results/rerank_sweep_dev/summary.json
tail -n 6 rag/results/dev_baseline_retrieval.txt
grep -A7 '"baseline"' rag/results/rerank_sweep_dev/summary.json
```

**Point at:**
- the nine identical `0.7244…` lines;
- the supplied `evaluate_rag.py` output (`Recall@5 0.379497 … MRR@10 0.459708`) next to our sweep's baseline (`0.3794974…`, `0.4597079…`).

**Say** (37 words):
> Second, checks on the real runs. Reranking only reorders the top N, so Recall@100 must not change, and it is 0.724 in every configuration. Our new sweep script also reproduces the supplied evaluation script's baseline numbers exactly.

**Clip 3, slide 14 (about 20 s). Say** (47 words):
> Third, an example: we tested the judge before trusting it. We answered five practice questions twice with identical inputs, and four of the five scores changed, one by 0.5. That is why we compare paired means over all 125 questions, with bootstrap intervals, and not single answers.

---

## 5. Edit, export and check (Shotcut, and HandBrake only if needed)

1. In Shotcut, create a project and drag the takes onto the timeline in order: T1 … T9.
2. Trim the start and end of each take. To remove a mistake, split on both sides of it, delete it, and close the gap.
3. **Cuts during waiting time:**
   - Split before and after the wait, and delete the middle.
   - Add a Text filter ("Text: Simple") on the clip after the cut, showing "Waiting time shortened" for 2–3 seconds.
   - Keep the command line and the final output visible.
4. Check the timing on the timeline:
   - Question 1 should start about 2:00 before the end, so the two answers fill the final two minutes.
   - The whole video must be **10:00 or less**.
5. Export: File → Export → **H.264 High Profile** (MP4, H.264 video, AAC audio), with the file name `video.mp4`. Then watch the whole export once, checking for gaps, missing audio and cuts in the wrong place.
6. Metadata: in a Mac terminal, run `mdls -name kMDItemTitle -name kMDItemAuthors -name kMDItemDurationSeconds video.mp4`.
   - No name may appear.
   - The duration must be at most 600 seconds.
   - If a title shows your name, rename the Shotcut project (for example to "video") and export again.
7. Only if the file is larger than the upload limit on Learn:
   - Use HandBrake, preset **Fast 1080p30**, with a new output name.
   - Encode a short preview and check that the smallest terminal text is still readable.
   - Raise RF by 2 if the file is still too large. Keep the original export.

### Final checks before submitting

- [ ] At most 10:00, with a real system run (both live RAG runs and the HW1 runs are in it).
- [ ] The final two minutes answer both questions, introduced as "Question 1" and "Question 2".
- [ ] The narration is audible throughout; terminal text and slide text are readable in the exported file.
- [ ] No names, UUNs, group ID, faces, home paths or API keys; neutral file name `video.mp4`; metadata checked.
- [ ] Every cut in a wait is marked "Waiting time shortened".
- [ ] Uploaded through the route on Learn; after upload, play the submitted file to confirm it is the right version.

### AI declaration (add to the report's final page)

The report's AI declaration does not yet mention the video. Suggested bullet for `report/v2/report.tex`, in the itemize list of the AI declaration:

```latex
\item Video: it drafted the narration script and the slides for the video, and suggested the commands for the live demonstration, including a small shell function that formats the demo output. We checked the content, ran the demonstration and recorded the narration ourselves.
```

---

## If you run long

Cut in this order, and never cut the live runs or the two questions:

1. Skip the question audio on slide 8 (saves about 6 s).
2. In segment 4, do not scroll up; say only "All 33 cases pass on our RTX 2080 Ti" and the GEMM sentence (about 8 s).
3. Shorten segment 9 to: "We chose N on the 350 practice questions and report on the 125 dev questions." (about 8 s)
4. Drop the last sentence of segment 14 (about 5 s).
5. In segment 3, keep only the GEMM and attention sentences (about 15 s).

---

## Appendix: every command in order

```bash
# ---- off camera: log in and book the GPU (contains your UUN) ----
ssh <UUN>@ssh.inf.ed.ac.uk
ssh mlp.inf.ed.ac.uk
srun -p Teaching --gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=2 --mem=16G --time=01:30:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH
conda activate mls-asr
cd ~/edin-mls-26-autumn-101
make -C hw1-asr lib
export PS1='$ '
unset PROMPT_COMMAND
printf '\033]0;terminal\007'
cd hw1-asr
clear

# ---- T2 (on camera) ----
make test

# ---- T3 (on camera) ----
./run.sh --profile 2>&1 | sed "s#$HOME#~#g"

# ---- off camera: switch to the RAG set-up, then paste show() from 1.4 ----
cd ~/edin-mls-26-autumn-101
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"
source "$HOME/mls-rag-env/bin/activate"
rm -f /tmp/video_base.json /tmp/video_bge.jsonl /tmp/video_r1.log /tmp/video_r2.log
clear

# ---- T5 (on camera): run 1 ----
export ASR_CUDA=all
python rag/scripts/run_audio_query.py --device cpu --split dev \
    --query-id q_33fd7baba78a1358 > /tmp/video_base.json 2> /tmp/video_r1.log
show /tmp/video_base.json q_33fd7baba78a1358 dev

# ---- T6 (on camera): run 2 ----
python rag/scripts/generate_answers.py --device cpu --split dev --limit 2 \
    --transcript-cache rag/.cache/dev_transcripts.jsonl \
    --reranker rag/models/rerankers/bge-base --candidate-depth 10 \
    --output /tmp/video_bge.jsonl 2> /tmp/video_r2.log
show /tmp/video_bge.jsonl q_33fd7baba78a1358 dev

# ---- T9 clip 2 (on camera): Question 2 evidence ----
python -m unittest discover -s rag/tests -p "test_rerank.py" -v
clear
grep '"Recall@100"' rag/results/rerank_sweep_dev/summary.json
tail -n 6 rag/results/dev_baseline_retrieval.txt
grep -A7 '"baseline"' rag/results/rerank_sweep_dev/summary.json

# ---- when done ----
exit
```
