# CLAUDE.md — Coursework plan for INFR11269 Machine Learning Systems (Autumn 2026)

This file is a step-by-step plan for the whole coursework in this repository. It is
written in plain language, in the order you should do things. Each phase ends with
a **"Done when"** checklist so you know when to move on.

> Claude Code also reads this file at the start of every session. The section
> [Notes for AI assistants](#notes-for-ai-assistants-working-in-this-repo) at the end
> lists the rules that any AI helper must follow in this repo.

---

## 0. The big picture (read this first)

The coursework has **four parts**, done in this order:

| # | Part | Folder | What you actually produce |
|---|------|--------|---------------------------|
| 1 | CUDA tutorial (practice, not graded on its own) | `cuda-tutorial/` | Nothing to hand in — you learn the CUDA you need |
| 2 | Homework 1: ASR kernels | `hw1-asr/` | 5 completed CUDA files + test/profile output + answers to ≥4 guide questions |
| 3 | RAG systems investigation | `rag/` | One precise experiment comparing a change against the baseline (quality **and** cost) |
| 4 | Report | `report/` | A ≤6-page PDF (ICLR template) covering parts 2 and 3 |

**In one sentence:** you write the GPU code that makes a speech-recognition model work,
then plug that model into a "listen to a question → search documents → answer" system,
change one part of that system, measure what happens, and write it up.

### What you hand in at the end

1. **A code zip** containing your 5 edited kernel files:
   `hw1-asr/csrc/task1_rope.cu`, `task2_reduce.cuh`, `task2_reduce.cu`, `task3_gemm.cu`, `task4_attention.cu`
   (plus your RAG investigation code, e.g. changes in `rag/src/` or `rag/extensions/` —
   confirm against the course submission brief).
2. **The report PDF**, which must contain:
   - output of `make test` and `./run.sh --profile` (same GPU node) and the GPU name (`nvidia-smi -L`) — main body or appendix;
   - answers to **at least one question from each of the 4 task guides** (4+ answers) in the **main body**;
   - your RAG systems investigation (question, baseline, change, controls, quality + cost results, analysis, limitations);
   - the **AI declaration** and **group information** sections from the template.

### Rough time budget

| Phase | Typical effort |
|-------|---------------|
| Accounts, cluster access, CUDA tutorial | 0.5–1 day |
| HW1 setup | 1–2 hours (mostly downloading) |
| HW1 tasks 1–4 | 3–6 days (task 1 is short, task 4 is the longest) |
| RAG setup + baseline runs | 0.5–1 day |
| Systems investigation | 4–7 days |
| Report | 2–4 days |

Start the report skeleton early and fill it in as you go — it is much easier than
writing everything at the end.

---

## Phase 1 — Accounts and things to request early

Do these **on day one**, because some of them take time to be approved.

1. **DICE account** — you need your University UUN and an Informatics DICE account.
2. **Network access** — the cluster only accepts connections from campus, a DICE
   machine, or the **Informatics VPN**. Install and test the VPN now.
3. **Teaching cluster access** — check you can reach `mlp.inf.ed.ac.uk` (Phase 2).
4. **ELM API key** — apply for your own University ELM key. The RAG part *cannot*
   generate or judge answers without it. Never commit it to git.
5. **Group** — if you are in a group, agree who does what and get your Group ID
   (it goes on the last page of the report).
6. **Overleaf account** — for writing the report.

**Done when:** you have VPN working, a DICE login, and an ELM key request submitted.

---

## Phase 2 — Log in to the cluster and do the CUDA tutorial

### 2.1 Understand the two kinds of machine

- **Head node** (`mlp.inf.ed.ac.uk`): where you log in, edit files, use git, download
  things. **It has no GPU.** Running GPU code here fails with "No GPU found".
- **Compute node**: a machine with a GPU that Slurm (the job scheduler) lends you for
  a limited time. You get one with `srun`. GPU nodes may have **no internet**.

Think of it like a library: the head node is the front desk, the compute node is a
study room you book for an hour.

### 2.2 Log in (two hops if you're off campus)

```bash
ssh s2916110@ssh.inf.ed.ac.uk     # hop 1: the gateway
ssh mlp.inf.ed.ac.uk              # hop 2: the cluster head node
```

Example: if your UUN is `s2412345`, the first command is `ssh s2412345@ssh.inf.ed.ac.uk`.

### 2.3 Clone the repo **on the cluster** (not just on your Mac)

```bash
git clone https://github.com/ed-aisys/edin-mls-26-autumn.git
cd edin-mls-26-autumn
```

> Your Mac copy of this repo is fine for reading and editing, but **everything runs on
> the cluster**. Either edit on the cluster (e.g. VS Code Remote-SSH) or push/pull your
> own private git copy between the two. Don't edit two copies separately and lose work.

### 2.4 Book a small GPU and build the examples

```bash
# Run these on mlp (the head node). Check first: `hostname` must show mlp,
# not ssh.inf (the gateway) or your laptop — `srun` only exists on mlp.
cd ~/edin-mls-26-autumn/cuda-tutorial
srun -p Teaching --gres=gpu:1 --cpus-per-task=1 --mem=2G --time=00:40:00 --pty bash
# --- now you are on a compute node ---
hostname                                 # which machine you got
nvidia-smi -L                            # which GPU you got
export PATH=/opt/cuda-12.8.0/bin:$PATH   # make nvcc (the CUDA compiler) available
nvcc --version
make
cd 1-first-kernel && ./hello
```

What the `srun` flags mean: `-p Teaching` = the course's partition; `--gres=gpu:1` =
one GPU; `--mem=2G` = 2 GB of normal RAM; `--time=00:40:00` = 40 minutes;
`--pty bash` = give me an interactive shell.

**Always type `exit` when finished** so the GPU goes back to the pool.

### 2.5 Work through the lessons in order

| Lesson | What to learn | Why it matters for HW1 |
|--------|---------------|------------------------|
| `1-first-kernel` | grid / block / thread, `__global__` | Every task |
| `2-vectoradd` | global thread index `blockIdx.x * blockDim.x + threadIdx.x`, bounds check, memory copies | Task 1 is basically a fancier vector add |
| `3-streams` | streams and events | Background only |
| `4-shared-memory` | `__shared__` arrays, reuse of loaded data | Tasks 2, 3, 4 |
| `5-tiled-gemm` | tiles, `__syncthreads()` barriers, timing | **Task 3 starts from this exact code** |
| `6-optional-thrust-atomics` | atomics, races | Optional; helps with Task 2 "Try this" #4 |

For each lesson: read the README, read the `.cu` file, run it, then change one thing
(e.g. the block size) and predict what happens before running again.

**Example of the core idea (vector add):** with 1000 elements and 256 threads per
block you launch `ceil(1000/256) = 4` blocks = 1024 threads. Threads 1000–1023 have
no element, so the kernel must start with `if (i >= n) return;`. You will use this
pattern in Task 1.

**Done when:** you can explain, without looking, what `blockIdx`, `threadIdx`,
`__shared__` and `__syncthreads()` do, and you have run `5-tiled-gemm`.

---

## Phase 3 — Homework 1 setup (once)

### 3.1 Run the setup script **on the head node** (it needs internet)

```bash
cd ~/edin-mls-26-autumn/hw1-asr
bash setup_env.sh
```

It installs Miniconda (if you don't have conda), creates a conda environment called
`mls-asr` (Python 3.11, PyTorch 2.10, transformers 5.0) and downloads the 4.5 GB
model. It needs ~20 GB free in your cluster home. If it stops halfway, just run it again.

### 3.2 Your daily "start work" routine (do this every session)

```bash
srun -p Teaching --gres=gpu:1 --cpus-per-task=2 --mem=16G --time=01:00:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH
conda activate mls-asr
cd ~/edin-mls-26-autumn/hw1-asr
nvidia-smi -L
```

Note `--mem=16G` here — the model needs much more RAM than the tutorial did.

### 3.3 Pick ONE GPU type for all your timing experiments

Different GPUs have very different speeds. If you time version A on an RTX 2080 Ti and
version B on an H200 slice, the comparison is meaningless. Choose one and always ask
for it by name, e.g.:

```bash
srun -p Teaching --gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=2 --mem=16G --time=01:00:00 --pty bash
```

| `nvidia-smi -L` shows | SMs | Memory | `--gres` name | Build folder |
|---|---:|---:|---|---|
| RTX 2080 Ti | 68 | 11 GB | `nvidia_geforce_rtx_2080_ti` | `build/sm_75` |
| RTX A6000 | 84 | 48 GB | `nvidia_rtx_a6000` | `build/sm_86` |
| H200 MIG 1g.18gb | 16 | 18 GB | `h200_1g.18gb` | `build/sm_90` |
| H200 MIG 3g.71gb | 60 | 71 GB | `h200_3g.71gb` | `build/sm_90` |

Write down which one you chose — it goes in the report's methodology section.
Look up its **peak memory bandwidth** (GB/s) now; you need it for Task 1/2 questions
(the tests also print a "peak").

### 3.4 Check the setup works before touching any code

```bash
make                    # must finish with no errors (kernels compile but do nothing yet)
./run.sh --cuda none    # runs the model with PyTorch only
```

It should end with `Transcription: Concord returned to its place amidst the tents.`,
`Accuracy: 100.0%` and `Status: PASS`. **Save this output** — it is your PyTorch
baseline timing for later comparison.

**Done when:** `./run.sh --cuda none` prints `Status: PASS`.

---

## Phase 4 — Homework 1: the four kernel tasks

### 4.0 How every task works (same loop each time)

1. Read the guide in `hw1-asr/docs/` **fully** before coding.
2. Open the task file in `hw1-asr/csrc/`, find `BLANK x.y`, replace `// YOUR CODE HERE`.
   Each blank is 1–15 lines. Don't touch the code around the blanks (except tuning constants).
3. Build and test:
   ```bash
   make            # fix every compiler error
   make test1      # (test2, test3, test4 for later tasks) → want "ALL PASS"
   ./run.sh --cuda task1          # the real model with your kernel on
   ```
4. When the test passes, run the guide's **"Try this"** experiments and **save the
   numbers** in your experiment log (see [Appendix B](#appendix-b--experiment-log-template)).
5. Draft answers to the guide's **"Questions for the report"** while it's fresh.
6. Commit your work (to your own private copy — never a public repo).

How to read a failing test line:

```
text decoder (GQA 16/4, full)   FAIL  max_err=2.39e+00
    160556 of 163840 checked values are wrong. First: index 2048, got -0.0918819, expected -0.22623
```
"`got 0`" usually means a blank is still empty or you write to the wrong place.
"First: index 2048" tells you where to start looking — work out which row/column that is.

Debugging tools (use on the **tests**, not the whole model, and add `small`):

```bash
ASR_DEBUG=1 ./run.sh                                         # errors reported by the op that caused them
compute-sanitizer build/sm_75/test_task3_gemm small          # out-of-bounds accesses, with line numbers
compute-sanitizer --tool racecheck build/sm_75/test_task3_gemm small   # shared-memory races
```
(Replace `sm_75` with your GPU's folder.)

---

### Task 1 — RoPE (warm-up) · `csrc/task1_rope.cu` · guide `docs/task1-rope.md`

**What it does, simply:** before attention, each query/key vector gets "rotated" by an
angle that depends on the word's position, so the model can tell position 3 from
position 30. Numbers are rotated in pairs: element `i` pairs with element `i + half`.

**Worked example (from the guide):** `head_dim = 4`, `half = 2`, `inv_freq = [1.0, 0.01]`,
token at position 3. Pair 0 = `(v[0], v[2])` rotates by `3 × 1.0 = 3` radians; pair 1 =
`(v[1], v[3])` by `3 × 0.01 = 0.03` radians. `[1, 0, 0, 0]` becomes `[cos3, 0, sin3, 0] = [-0.990, 0, 0.141, 0]`.

**Who does what:** one thread per pair. All of `q`'s pairs come first, then `k`'s.
With `seq_len=3, num_q_heads=2, num_k_heads=1, half=2` there are `3×2×2 = 12` q-pairs
and `3×1×2 = 6` k-pairs = 18 threads. Thread 7 → q, row 3 (token 1, head 1), pair 1.
Thread 13 → it's past the 12 q-pairs, so it works on k, and *inside k* it is pair number 1 → k row 0, pair 1.

| Blank | Plain-English job |
|-------|-------------------|
| 1.1 | Compute this thread's global index (like vector add); threads with no pair stop. |
| 1.2 | If the index is past q's pairs, switch to `k` (pointer, head count) and re-count the index from 0 inside k. |
| 1.3 | Turn the index into: which row, which pair in that row, and the token position. Hint: each row has `half` pairs; rows go head-by-head inside a token, and positions restart at 0 for each batch element. |
| 1.4 | Apply the rotation formula. Read both old values into variables **before** writing either one. |

Test: `make test1` → `Task 1 (RoPE): ALL PASS`, then `./run.sh --cuda task1`.

Try-this to run and log: `ROPE_THREADS` = 32/64/128/256/512/1024; why tiny cases are slow;
`--use_fast_math` (remember `make clean` afterwards).

Report questions (pick ≥1): bytes moved + % of peak bandwidth; cos/sin table vs compute; why one launch for q and k.
*How to count bytes (example method):* every element the kernel touches is read once
(4 bytes) and written once (4 bytes), plus the small `inv_freq` reads. Count how many
elements of q and k are inside the rotated part for the encoder shape, multiply by 8,
then divide by the measured time to get GB/s and compare with your GPU's peak.

---

### Task 2 — Reductions · `csrc/task2_reduce.cuh` (part A) + `csrc/task2_reduce.cu` (part B) · guide `docs/task2-reduce.md`

**What it does, simply:** a reduction turns many numbers into one (sum, max). Now
threads must cooperate. Two levels:
- **Warp level** (32 threads): lanes swap values directly with `__shfl_xor_sync`, in 5 steps (offset 16, 8, 4, 2, 1). After that every lane holds the total. No shared memory needed.
- **Block level** (e.g. 256 threads = 8 warps): each warp reduces, lane 0 of each warp writes to a small `__shared__` array, `__syncthreads()`, then one more warp reduction. `block_reduce_sum` is given as a complete worked example — copy its *shape* for `block_reduce_max`.

**Butterfly example with 8 lanes holding a..h:** after offset 4 lane 0 holds `a+e`;
after offset 2 it holds `a+e+c+g`; after offset 1 every lane holds `a+b+…+h`.

**Online softmax trick (`MaxSum`):** summarise a set of numbers as `{m = max, l = Σ exp(x−m)}`.
Two summaries merge as `m = max(m1,m2)`, `l = l1·exp(m1−m) + l2·exp(m2−m)`.
Example: `{1,3}` → `{3, 1.135}`, `{2}` → `{2, 1}`; merged → `{3, 1.135 + 0.368} = {3, 1.503}`,
same as computing `{1,3,2}` directly. This lets softmax read the row **once** to get both max and sum.

| Blank | File | Plain-English job |
|-------|------|-------------------|
| 2.1 | `.cuh` | One butterfly step of the warp **sum**. |
| 2.2 | `.cuh` | Same for **max**. |
| 2.3 | `.cuh` | Block max — like `block_reduce_sum`, but missing warps must contribute `-INFINITY`, not 0 (think: a row of all-negative numbers). |
| 2.4 | `.cuh` | `maxsum_merge`. Careful: merging two empty summaries `{-inf,0}` must give `{-inf,0}`, not `nan` (because `-inf − (-inf) = nan`). |
| 2.5 | `.cu` | RMSNorm: `y = x / sqrt(mean(x²) + eps) * weight`. Copy the loop shape of the given `layernorm_kernel`. |
| 2.6 | `.cu` | Softmax pass 1: one read of the row → `row_max` and `row_sum` using `MaxSum`. |
| 2.7 | `.cu` | Softmax pass 2: `y = exp(x − row_max) / row_sum`. |

Golden rule: every thread of the block must call `block_reduce_*` (they contain barriers).
Never put them inside an `if` that only some threads enter.

Test: `make test2`, then `./run.sh --cuda task1,task2`.

Try-this to log: `REDUCE_THREADS` sweep; why the 59,264-wide vocab row is slow (1 block → 1 SM busy);
remove the final `__syncthreads()` and run racecheck (put it back!); `atomicAdd` version of RMSNorm.

Report questions (pick ≥1): an input that breaks max with 0-padding (e.g. a row of all `-5`);
`expf(100.0f)` overflow and why subtracting the max is safe; how many global reads per element;
the two-warp race story.

---

### Task 3 — Tiled GEMM (the linear layer) · `csrc/task3_gemm.cu` · guide `docs/task3-gemm.md`

**What it does, simply:** `y = act(x · Wᵀ + bias)`. This is ~all the model's arithmetic,
so it decides overall speed. Start from `cuda-tutorial/5-tiled-gemm/gemm.cu` — open it side by side.

**Key differences from the lecture:**
1. **W is stored transposed** as `W[N, K]` (one row per output). So `y[m][n] = Σ_k x[m][k] · W[n][k]`.
   Load the W tile so that neighbouring threads read **neighbouring addresses** (`k` varies with `threadIdx.x`) — this is "coalesced" and fast.
2. `ws[TILE][TILE+1]` — the `+1` avoids shared-memory bank conflicts (already in the code).
3. **Epilogue (fusion):** add `bias[col]` if `bias != nullptr`, apply `gelu()` if `act == ACT_GELU`, then store — only for `(row, col)` inside y.

**Example of why tiling helps:** with `TILE = 16`, each value loaded into shared memory
is used by 16 threads, so global memory is read 16× less than the naive kernel.

| Blank | Plain-English job |
|-------|-------------------|
| 3.1 | Load your element of the x tile into `xs`; 0 if outside x. |
| 3.2 | Load your element of the W tile so `ws[n][k] = W[blockIdx.x*TILE + n][k0 + k]`; 0 if outside; coalesced. |
| 3.3 | Add `TILE` products to `acc`. |
| 3.4 | Bias → GELU (if asked) → store (if inside y). |

Rule: **no early `return`** — every thread, even one outside y, must reach both `__syncthreads()`.
Outside elements are simply loaded as 0.

Test: `make test3` (prints your GFLOP/s vs naive vs cuBLAS), then `./run.sh --cuda task1,task2,task3`.
Expect: several × faster than naive, still several × slower than cuBLAS — that's normal.

Try-this to log: `TILE` = 8/16/32; remove `+1`; uncoalesced W load (watch `lm_head`);
remove second barrier + racecheck; compute `lm_head` GB/s (it reads a 485 MB weight matrix).
GFLOP/s = `2·M·N·K / time`.

Report questions (pick ≥1): global loads naive vs tiled for 1024³; one thing cuBLAS does better
(e.g. register tiling — each thread computes several outputs; vectorised loads; double buffering);
bytes saved by fusing GELU into `fc1` (`1500 × 5120` floats, read+write); why all threads must reach both barriers.

---

### Task 4 — Attention (FlashAttention-style) · `csrc/task4_attention.cu` · guide `docs/task4-attention.md`

**What it does, simply:** each position looks at all others and takes a weighted
average of their `v` vectors. Weights = softmax of `q·k × scale`. Your kernel never
stores the big score matrix — it walks the keys 32 at a time and keeps a running summary
`m` (max), `l` (sum of exps), `acc` (weighted sum of values).

**Worked example (from the guide):** scores `1, 2 | 3, 0`, values `10, 20 | 30, 40`, tiles of 2.
- After tile 1: `m = 2`, `l = 1.368`, `acc = 23.68`.
- Tile 2 has a bigger max 3 → `corr = exp(2−3) = 0.368` shrinks the old `l` and `acc`.
- After tile 2: `l = 1.553`, `acc = 40.70`, so `out = 40.70 / 1.553 = 26.21`. Check it with a calculator — it equals the plain softmax answer.

**Layout of work:** one block = 4 warps = 4 queries. Lane `r` scores key `j0 + r`; then
the lanes switch to "dimension" jobs (with `head_dim = 128`, each lane owns 4 dims:
lane 0 owns 0, 32, 64, 96). The `p` values must be passed between lanes (`__shfl_sync`).

Special cases:
- **GQA** (decoder): 16 query heads share 4 K/V heads → heads 0–3 use K/V head 0, 4–7 use 1, … i.e. a group size of `num_heads / num_kv_heads`.
- **Causal mask** (decoder): query `i` may only see keys `≤ i` (use the given `last_key`); masked score = `-INFINITY` so `p = 0`.
- All warps must loop to `block_last_key` (not their own `last_key`), because the loop has barriers.

| Blank | Plain-English job |
|-------|-------------------|
| 4.1 | Which K/V head this query head reads (GQA). |
| 4.2 | All 128 threads copy 32 keys + 32 values into `k_tile` (row pitch `head_dim+1`) and `v_tile`; 0 past `seq_k`. |
| 4.3 | This lane's score for key `j0 + lane`, or `-INFINITY` if masked/out of range. |
| 4.4 | Update `m`, `l`, `acc` (rescale with `corr`), compute `p`. Reuse your Task 2 warp reductions. Beware `-inf − (-inf)`. |
| 4.5 | Add `p`-weighted values to `acc`. |
| 4.6 | Write `acc / l` to `out`. |

Test: `make test4`. The "peaky scores" case fails if the `corr` rescaling is wrong.
Then `./run.sh` (all four tasks) → `Status: PASS`.

Try-this to log: `ATTN_WARPS` = 1/2/4/8/16; `k_pitch = head_dim` (bank conflicts);
compare with PyTorch attention: `./run.sh --cuda task1,task2,task3 --profile`.

Report questions (pick ≥1): size of the full score matrix (`20 × 1500 × 1500 × 4 bytes`) vs shared memory per block;
why equal tile iterations; `corr` when the max doesn't change (= 1) and when `m_new` is `-inf`;
how many times each key is loaded with `ATTN_WARPS = 4`; why `acc` has a fixed size (registers vs local memory).

---

## Phase 5 — Finish HW1 and collect the hand-in output

On **one GPU node of your chosen type**, in one session:

```bash
nvidia-smi -L                       | tee hw1_gpu.txt
make test                           | tee hw1_make_test.txt
./run.sh --profile                  | tee hw1_profile.txt
./run.sh --cuda none --profile      | tee hw1_profile_pytorch.txt   # for comparison in the report
```

Alternative (batch job, no interactive shell): `sbatch run_asr.sbatch`, then
`squeue -u $USER` until it's gone, then `cat asr-JOB_ID.out`.

What to look at in the profile: which operation takes most time (it will be `linear`,
task 3), and how your total compares to PyTorch/cuBLAS. That ratio is a good sentence
for the report.

Before you move on, check the **rules**:
- Only the 5 task files are changed (`git status` / `git diff --stat`). If you edited
  `ops.py`, `model.py`, `benchmark.py`, `provided.cu` or the tests while experimenting, restore them: `git checkout -- <file>`.
- No cuBLAS / cuDNN / CUTLASS / Thrust / PyTorch doing the work inside your kernels.

**Done when:** `make test` is all PASS, `./run.sh` prints `Status: PASS`, and the three output files are saved.

---

## Phase 6 — RAG setup (once)

**What RAG is, simply:** someone *speaks* a question → your ASR model turns it into
text → a small encoder model turns the text into a vector → we find the 5 corpus
records whose vectors are most similar → we send the question + those records to the
ELM language model → it writes an answer. Code: `rag/src/rag.py` shows the whole chain.

```
audio ─▶ ASR (your kernels) ─▶ transcript ─▶ enc_a encoder ─▶ vector
      ─▶ ExactDenseRetriever (top-k of 24,000 records) ─▶ SimpleContextBuilder (prompt)
      ─▶ ELMGenerator (answer)
```

All commands below run from the **repository root** (`~/edin-mls-26-autumn`).

### 6.1 Create the RAG Python environment (head node)

```bash
conda activate mls-asr
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"
python -m venv "$HOME/mls-rag-env"
source "$HOME/mls-rag-env/bin/activate"
python -m pip install -r rag/requirements.txt
cp rag/.env.example rag/.env
```

Then edit `rag/.env` and put your key on the `ELM_API_KEY=` line. Leave the other
settings alone for now. `.env` is git-ignored — keep it that way.

### 6.2 Download and install the course data (head node)

`rag-assets.zip` is stored with **Git LFS**. If it is only ~134 bytes, you have the
pointer file, not the real archive (this is the case in the Mac copy of this repo).

```bash
git lfs install
git lfs pull --include="rag-assets.zip"
ls -lh rag-assets.zip                      # should now be large, not 134 bytes
mkdir -p "$HOME/mls-rag-assets"
unzip rag-assets.zip -d "$HOME/mls-rag-assets"
python rag/scripts/install_assets.py --from-dir "$HOME/mls-rag-assets"   # → "Course assets verified"
```

You now have 24,000 corpus records, 350 **practice** and 125 **dev** audio queries,
relevance judgements (qrels: "which records are correct for which query"), enc_a and
enc_b corpus embeddings, and the enc_a encoder.

### 6.3 Daily RAG start routine (GPU node)

```bash
# after srun with --mem=16G (Phase 3.2)
conda activate mls-asr
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"
source "$HOME/mls-rag-env/bin/activate"
export PATH=/opt/cuda-12.8.0/bin:$PATH
cd ~/edin-mls-26-autumn
make -C hw1-asr lib
python rag/scripts/check_setup.py          # all PASS
python -m unittest discover -s rag/tests -v
```

**Done when:** `check_setup.py` reports PASS for corpus, manifest, embeddings, enc_a, ELM config and ASR path.

---

## Phase 7 — Run the RAG baseline and record its numbers

The **baseline** is the system exactly as supplied. You need its numbers because your
investigation is a comparison *against* it.

### 7.1 One query, end to end

```bash
export ASR_CUDA=none                                   # PyTorch ASR first
python rag/scripts/run_audio_query.py --device cpu
export ASR_CUDA=all                                    # now your kernels
python rag/scripts/run_audio_query.py --device cpu
```

Read the output: transcript, retrieved record IDs, evidence, answer, and timings
(`asr_ms`, `embedding_ms`, `retrieval_ms`, `generation_ms`, `end_to_end_ms`).
Ask yourself: is the answer actually supported by the evidence? Other options:
`--query-id <id>`, `--split dev`, `--k 10`.

### 7.2 Retrieval quality (no ELM cost)

```bash
python rag/scripts/evaluate_rag.py --device cpu --split practice \
    --transcript-cache rag/.cache/practice_transcripts.jsonl
```

Prints Recall@5/10/20/100, MRR@10 and NDCG@10.
- *Recall@10 = 0.6* means: on average, 60% of the relevant records appear in the top 10.
- *MRR@10* rewards putting the first correct record near the top (rank 1 → 1.0, rank 2 → 0.5, …).

The transcript cache means ASR runs only once; later runs reuse the same transcripts,
so you compare retrieval changes fairly. Use a **new** cache file only if you change ASR.

### 7.3 Answer quality (costs ELM requests — start small)

```bash
python rag/scripts/generate_answers.py --device cpu --split practice --limit 5 --output rag/results/sample_answers.jsonl
python rag/scripts/evaluate_answers.py --split practice --candidates rag/results/sample_answers.jsonl --allow-subset --dry-run
python rag/scripts/evaluate_answers.py --split practice --candidates rag/results/sample_answers.jsonl --allow-subset --out-dir rag/results/sample_judge
```

The judge gives correctness, grounding and completeness, each 0–4; normalised score =
sum / 12. Example: scores 3, 4, 2 → 9/12 = 0.75.

Then do a **full baseline run** (remove `--limit` and `--allow-subset`, use new
filenames such as `rag/results/baseline_practice.jsonl` and `rag/results/baseline_judge/`).
Output files are never overwritten — the scripts refuse — so name each run clearly.

> Note: `generate_answers.py` re-runs ASR for every query (it doesn't use the
> transcript cache). If your investigation is about retrieval/context/generation, consider
> adding a cached-transcript option in your own code so ASR noise and time don't vary between runs.

**Done when:** you have baseline retrieval metrics, baseline judge scores, and baseline timings saved in your log.

---

## Phase 8 — The systems investigation (the most important part)

### 8.1 Choose ONE precise question

A good question names **one change**, **one quality measure** and **one cost measure**.

| Weak | Strong |
|------|--------|
| "Make RAG better" | "How does retrieval depth k ∈ {1, 3, 5, 10, 20} trade answer quality against prompt length and generation latency?" |
| "Use FAISS" | "Does an HNSW approximate index match exact search's Recall@10 on 24k records, and what do we gain in search latency vs lose in build time and memory?" |
| "Try a new LLM" | ❌ Not allowed — replacing the language model alone is not a systems investigation. |

Directions the course suggests (pick one, go deep rather than wide):

1. **Retrieval depth / context size** — vary `--k`. Quality: Recall@k, judge score. Cost: prompt characters/tokens, `generation_ms`, ELM requests.
2. **ANN index vs exact search** — replace `ExactDenseRetriever` with e.g. FAISS IVF/HNSW (`pip install faiss-cpu` in the RAG env). Quality: Recall@k relative to exact. Cost: query latency, index build time, memory. Sweep a parameter (e.g. `nprobe` or `efSearch`) to get a trade-off curve.
3. **Batching** — encode many queries at once (`LocalEncoder.encode_many` already exists) or batch retrieval as a matrix multiply. Cost: throughput (queries/s), latency per query.
4. **Reranking** — retrieve 50, rerank to top 5 with a stronger model. Quality: MRR/NDCG, judge score. Cost: extra latency.
5. **Context selection** — dedupe, truncate, or order evidence in `context_builder.py`. Quality: grounding score. Cost: prompt length.
6. **Dynamic updates** — add/delete records without a full re-index; measure update cost and query quality.
7. **Representation migration (enc_b)** — the enc_b corpus vectors exist but its query encoder isn't bundled. Equal dimensions do **not** mean compatible spaces; you'd need to learn/justify a mapping (e.g. a linear map fitted on paired enc_a/enc_b document vectors). Measure quality vs migration cost.
8. **ASR-side** (allowed alternative focus) — e.g. a faster GEMM, KV cache, or encoder truncation of the 30 s padding; measure ASR accuracy (WER / downstream recall) vs latency. Follow the ASR requirement files if you go this way.

### 8.2 Plan the experiment before running it

Write down (this becomes your Methodology section):
- **Baseline**: supplied pipeline, enc_a, exact retrieval, k = 5, ELM model from `.env`.
- **What changes**: exactly one thing (plus the parameter you sweep).
- **What stays fixed (controls)**: same split, same cached transcripts, same GPU type, same ELM model and judge settings, same k unless k is the variable.
- **Quality metric(s)**: Recall@k / MRR@10 / NDCG@10 and/or judge score.
- **Cost metric(s)**: latency (ms, mean and median/p95), memory, build time, prompt size, ELM calls.
- **Repetitions**: time each configuration several times (e.g. 5 runs), report mean ± std; warm up first.
- **Budget**: how many ELM calls will this need? (e.g. 5 k-values × 125 dev queries = 625 generation calls + 625 judge calls). Use the practice split while developing; use dev for final numbers.

### 8.3 Implement

- Put new code in `rag/extensions/` (e.g. `rag/extensions/faiss_retriever.py`) or modify `rag/src/` — both are allowed.
- Keep the baseline runnable side by side (a flag or separate script), so you can always re-run "before" and "after".
- Record timings with `time.perf_counter()` around exactly the part you're measuring, as `rag.py` already does.

### 8.4 Run, record, plot

- Every run: save its output file, the exact command, git commit hash, GPU, date (Appendix B template).
- Make 1–3 clear figures/tables, e.g. a quality-vs-latency curve with one point per configuration and the baseline marked.

### 8.5 Explain *why*

The marks are in the explanation, not just the numbers. Example:
"Recall@10 rose from k=5 to k=10, but the judge's grounding score fell, because the
extra documents were mostly off-topic and the model quoted them. Generation latency
grew roughly linearly with prompt length." Negative results are fine if you explain them.
Also consider other explanations (ASR transcript errors, judge noise — try `--repeats`).

**Done when:** you have a table/figure of quality vs cost for baseline and your change, and a paragraph explaining the mechanism.

---

## Phase 9 — Write the report

### 9.1 Set up

Upload `report/report_template_overleaf.zip` to Overleaf (New Project → Upload Project),
set `report.tex` as the main document. (Local alternative: `report/latex_template/`.)

**Do NOT change:** the style file, margins, font size, line spacing, the `\author{Anonymous coursework submission}` line,
the `\lhead{...}` line, keep `\iclrfinalcopy` and the abstract **commented out**, and keep References **before** the Appendix.

### 9.2 Suggested structure (≤ 6 pages main body; references and appendix don't count)

| Section | What goes in it | Rough length |
|---------|-----------------|--------------|
| 1. Introduction | System context, your question, why it matters, headline result | ~0.5 page |
| 2. System & Technical Approach | Brief: your HW1 kernels (design choices, tuning constants chosen) + the RAG baseline + your change | ~1.5 pages |
| 3. Experimental Methodology | Data split, GPU, software versions, metrics, controls, repetitions | ~0.75 page |
| 4. Results & Analysis | HW1: profile summary + ≥4 guide answers (one per task, ideally in one subsection). RAG: quality + cost table/figure and the *why* | ~2.5 pages |
| 5. Discussion & Conclusion | Trade-offs, limitations, what you'd do next | ~0.5 page |
| References | Cite FlashAttention, RoPE, FAISS/HNSW etc. if used | (free) |
| Appendix | Full `make test` / profile output, extra Try-this results | (free, may not be read) |
| AI declaration | Say exactly how you used AI tools (including Claude Code) | required |
| Group information | Group ID, names, UUNs | required |

Tip: reviewers may not read the appendix — anything you need for your argument must be in the main body.

### 9.3 Check the PDF

Download the compiled PDF, then from the repo root:

```bash
python -m pip install -r report/format_checker/requirements.txt
python report/format_checker/check_report_format.py /path/to/report.pdf
```

Fix every **FAIL**. Open the PDF and inspect every **WARN** (it checks page size, margins,
10pt font, header text, line numbers, and the 6-page body limit).

**Done when:** the checker shows no FAIL and the main body is ≤ 6 pages.

---

## Phase 10 — Final submission checklist

- [ ] `make test` → all four tasks ALL PASS on your chosen GPU
- [ ] `./run.sh` → `Status: PASS` with all tasks on your kernels
- [ ] Only the 5 task files differ from the original in `hw1-asr/` (`git diff --stat`)
- [ ] No library (cuBLAS, cuDNN, CUTLASS, Thrust, PyTorch) used inside your kernels
- [ ] Code zip contains the 5 kernel files (+ RAG investigation code as the brief requires)
- [ ] No `rag/.env` / API key anywhere in the zip or the report
- [ ] Report includes `make test` + `./run.sh --profile` output + GPU name
- [ ] Report main body answers ≥1 question from **each** of the 4 task guides
- [ ] Report has question, baseline, change, controls, quality **and** cost results, analysis, limitations
- [ ] AI declaration filled in honestly; group information filled in
- [ ] Format checker: no FAIL; main body ≤ 6 pages
- [ ] Submitted following the course submission brief (deadline, file names, where to upload)

---

## Appendix A — Quick troubleshooting

| You see | Cause | Fix |
|---------|-------|-----|
| `'srun' was not found` / `command not found: srun` | You're on the gateway (`ssh.inf`) or your laptop, not `mlp` | Run `hostname`; then `ssh mlp.inf.ed.ac.uk` and retry. Still missing on `mlp` → ask for Teaching cluster access |
| `No GPU found` / `nvidia-smi` fails | You're on the head node | `srun ... --pty bash` first |
| `nvcc: command not found` | PATH not set | `export PATH=/opt/cuda-12.8.0/bin:$PATH` |
| `No module named 'torch'` | Env not active | `conda activate mls-asr` |
| Test says `got 0` | Blank empty / wrong write index | Re-check that blank's index maths |
| `illegal memory access` | Index past array end | `compute-sanitizer build/sm_XX/test_taskN_... small` |
| Test hangs | `__syncthreads()` reached by only some threads | Move barriers out of thread-dependent `if`/`return` |
| `nan` in output | `exp` overflow, ÷0, or `-inf − (-inf)` | Handle the empty/`-INFINITY` cases (Tasks 2.4, 4.4) |
| Tests pass, transcription wrong | Shape the tests missed | `./run.sh --cuda task1`, then `task1,task2`, … to find the culprit |
| `no kernel image is available` | Built for another GPU | `make` on the node you run on |
| `CUDA out of memory` | GPU too small/busy | `nvidia-smi`; ask for another GPU |
| `srun` waits forever | Partition busy | Wait, or drop the specific GPU name; `sinfo -p Teaching` |
| `unzip`: not a zip archive | LFS pointer file | `git lfs install && git lfs pull --include="rag-assets.zip"` |
| ELM auth error | Key/endpoint wrong | Check `rag/.env` |
| ASR worker can't import torch | `RAG_ASR_PYTHON` unset | Re-run the `export RAG_ASR_PYTHON=...` line |
| "Output exists" | Scripts never overwrite | Use a new output filename |

## Appendix B — Experiment log template

Keep one line per run in a spreadsheet or `notes/experiments.md` (not handed in):

```
| date       | git commit | GPU          | what changed            | command                                   | result                          |
|------------|------------|--------------|-------------------------|-------------------------------------------|---------------------------------|
| 2026-11-10 | a1b2c3d    | RTX 2080 Ti  | TILE=16 (default)       | make test3                                | enc fc1: 1.9 ms, 1030 GFLOP/s   |
| 2026-11-10 | a1b2c3d    | RTX 2080 Ti  | TILE=32                 | make test3                                | enc fc1: ...                    |
| 2026-11-20 | d4e5f6a    | RTX 2080 Ti  | RAG baseline k=5, dev   | generate_answers.py --split dev --k 5 ... | judge 0.xx, gen_ms mean ...     |
```
(The numbers above are placeholders to show the format — fill in your own.)

## Appendix C — Daily command cheat sheet

```bash
# log in
ssh s2916110@ssh.inf.ed.ac.uk ; ssh mlp.inf.ed.ac.uk
# GPU shell (same GPU type every time for timing)
srun -p Teaching --gres=gpu:nvidia_geforce_rtx_2080_ti:1 --cpus-per-task=2 --mem=16G --time=01:00:00 --pty bash
export PATH=/opt/cuda-12.8.0/bin:$PATH
conda activate mls-asr
# HW1
cd ~/edin-mls-26-autumn/hw1-asr && make && make test1 && ./run.sh --cuda task1
# RAG
cd ~/edin-mls-26-autumn
export RAG_ASR_PYTHON="$(python -c 'import sys; print(sys.executable)')"
source "$HOME/mls-rag-env/bin/activate"
make -C hw1-asr lib && python rag/scripts/check_setup.py
# when done
exit
```

---

## Notes for AI assistants working in this repo

- **Where code runs:** everything needing a GPU (`make`, tests, `run.sh`, RAG with ASR) must run on a Slurm compute node of the Edinburgh teaching cluster, not on this Mac or the head node. Downloads (`setup_env.sh`, `git lfs pull`, `pip install`) happen on the head node.
- **Hand-in constraints for `hw1-asr/`:** only `csrc/task1_rope.cu`, `csrc/task2_reduce.cuh`, `csrc/task2_reduce.cu`, `csrc/task3_gemm.cu`, `csrc/task4_attention.cu` may differ in the final submission. Don't leave edits in `ops.py`, `model.py`, `benchmark.py`, `provided.cu` or `tests/`. Kernels must be hand-written CUDA — never use cuBLAS/cuDNN/CUTLASS/Thrust/PyTorch for task work.
- **Kernel rules:** every thread of a block must reach every `__syncthreads()` (no thread-dependent early returns before barriers); masked/missing values use `-INFINITY` for max-type reductions; avoid `-inf − (-inf)`.
- **RAG:** run scripts from the repository root. Never print, commit or copy `rag/.env` / `ELM_API_KEY`. Scripts refuse to overwrite outputs — use new filenames. ELM calls cost quota: test with `--limit` / `--dry-run` first. Replacing only the LLM is not a valid investigation.
- **Report:** don't alter the ICLR template's style, geometry, font size, header line, anonymous author line, or review mode; keep references before the appendix; main body ≤ 6 pages.
- **Academic integrity:** the course requires that all AI assistance be explicitly acknowledged in the report's AI declaration. Prefer explaining concepts and debugging the student's own code over writing the graded blanks wholesale, and keep a note of what help was given so the student can declare it accurately.
